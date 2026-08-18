"""Level-specific empirical-Bayes shrinkage configuration and summaries."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .results import TreeNBResult


class ShrinkagePrior(str, Enum):
    """Supported priors for species-by-taxonomy coefficients."""

    GAUSSIAN = "gaussian"
    LAPLACE = "laplace"
    SPIKE_SLAB = "spike_slab"


@dataclass(frozen=True)
class EmpiricalBayesConfig:
    """Configure pilot-gene empirical-Bayes learning in biological log-effect units.

    The learned scale is shared by every coefficient and gene within one
    ``species_tax_<level>`` family. Gaussian scales are standard deviations;
    Laplace scales are mean absolute deviations. Design columns must therefore
    retain a common log-effect interpretation across taxonomy levels.
    """

    prior: ShrinkagePrior
    pilot_genes: int = 256
    pilot_max_iter: int = 300
    update_interval: int = 20
    warmup_iterations: int = 100
    initial_scale: float = 0.5
    min_scale: float = 0.02
    max_scale: float = 3.0
    damping: float = 0.5
    spike_scale: float = 0.02
    initial_inclusion: float = 0.1

    def __post_init__(self) -> None:
        if self.pilot_genes < 1:
            raise ValueError("pilot_genes must be positive.")
        if self.pilot_max_iter < 1:
            raise ValueError("pilot_max_iter must be positive.")
        if self.update_interval < 1:
            raise ValueError("update_interval must be positive.")
        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations cannot be negative.")
        if not 0.0 < self.min_scale <= self.initial_scale <= self.max_scale:
            raise ValueError("Require min_scale <= initial_scale <= max_scale.")
        if not 0.0 < self.damping <= 1.0:
            raise ValueError("damping must lie in (0, 1].")
        if not 0.0 < self.spike_scale < self.initial_scale:
            raise ValueError("spike_scale must be positive and below initial_scale.")
        if not 0.0 < self.initial_inclusion < 1.0:
            raise ValueError("initial_inclusion must lie in (0, 1).")


def evolutionary_burden(result: TreeNBResult) -> pd.DataFrame:
    """Summarize invariant fitted evolutionary variance by taxonomy level.

    Gaussian burden is ``tau**2`` and Laplace burden is ``2*b**2``.
    Spike-and-slab burden is the mixture second moment. The values are
    comparable only for orthogonalized fits with the same coefficient coding.
    Ordinary fixed-L1 fits report their fitted contribution burden but have no
    learned prior moment or inclusion rate.
    """
    if not result.orthogonal_tree:
        raise ValueError(
            "Evolutionary burden requires orthogonal_tree=True so levels are comparable."
        )
    if result.designs is None:
        raise ValueError("Evolutionary burden requires keep_design_artifacts=True.")
    is_empirical_bayes = (
        result.shrinkage_prior is not None and result.shrinkage_scales is not None
    )
    prior = ShrinkagePrior(result.shrinkage_prior) if is_empirical_bayes else None
    families = (
        sorted(result.shrinkage_scales)
        if result.shrinkage_scales is not None
        else sorted(
            family
            for family in result.coefficients
            if family.startswith("species_tax_")
        )
    )
    rows: list[dict[str, float | str]] = []
    for family in families:
        if family not in result.designs:
            raise ValueError(f"Evolutionary burden is missing the design for {family}.")
        scale = float("nan")
        coefficient_second_moment = float("nan")
        inclusion_rate = float("nan")
        prior_predictive_burden = float("nan")
        if prior is ShrinkagePrior.GAUSSIAN:
            assert result.shrinkage_scales is not None
            scale = float(result.shrinkage_scales[family])
            coefficient_second_moment = scale**2
        elif prior is ShrinkagePrior.LAPLACE:
            assert result.shrinkage_scales is not None
            scale = float(result.shrinkage_scales[family])
            coefficient_second_moment = 2.0 * scale**2
        elif prior is ShrinkagePrior.SPIKE_SLAB:
            assert result.shrinkage_scales is not None
            scale = float(result.shrinkage_scales[family])
            if result.shrinkage_inclusion is None:
                raise ValueError("Spike-and-slab result is missing inclusion rates.")
            inclusion = result.shrinkage_inclusion[family]
            spike_scale = float(result.diagnostics["spike_scale"])
            coefficient_second_moment = (
                inclusion * scale**2 + (1.0 - inclusion) * spike_scale**2
            )
            inclusion_rate = float(inclusion)
        design = result.designs[family]
        design_energy = float((design**2).sum(axis=1).mean())
        if np.isfinite(coefficient_second_moment):
            prior_predictive_burden = coefficient_second_moment * design_energy
        fitted_contribution = design @ result.coefficients[family]
        burden = float((fitted_contribution**2).mean())
        rows.append(
            {
                "family": family,
                "level": family.removeprefix("species_tax_"),
                "prior": prior.value if prior is not None else "fixed_l1",
                "scale": scale,
                "coefficient_second_moment": float(coefficient_second_moment),
                "design_energy_per_group": design_energy,
                "variance_burden": float(burden),
                "prior_predictive_burden": float(prior_predictive_burden),
                "inclusion_rate": inclusion_rate,
            }
        )
    return pd.DataFrame(rows)
