"""Leakage-free tuning comparison for fixed-L1 and empirical-Bayes fits."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import anndata as ad
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse
from scipy.special import gammaln

from .eb_calibration import (
    EffectRegime,
    KnownTruthSimulation,
    _method_config,
    make_known_truth_simulation,
)
from .model import _build_design_matrices, fit_tree_nb
from .pseudobulk import aggregate_chunk, build_pseudobulk
from .results import TreeNBResult
from .species_tree import build_species_tree_design
from .taxonomy_tree import build_taxonomy_tree_from_obs

type FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class L1TuningComparisonResult:
    """Validation curves and invariant reconstruction metrics."""

    tuning_curve: pd.DataFrame
    metrics: pd.DataFrame
    method_comparison: pd.DataFrame
    selected_lambdas: dict[str, float]
    n_simulations: int


def _donor_split(
    *, adata: ad.AnnData, rng: np.random.Generator
) -> tuple[ad.AnnData, ad.AnnData]:
    """Hold out two donors per species without using synthetic truth."""
    obs = cast(pd.DataFrame, adata.obs)
    donor_species = obs[["donor", "species"]].astype(str).drop_duplicates()
    validation: set[str] = set()
    for _, frame in donor_species.groupby("species", sort=True):
        donors = np.asarray(sorted(frame["donor"]), dtype=object)
        chosen = rng.choice(donors, size=2, replace=False)
        validation.update(str(value) for value in chosen)
    validation_mask = obs["donor"].astype(str).isin(validation).to_numpy()
    return adata[~validation_mask].copy(), adata[validation_mask].copy()


def _mean_nb_nll(*, validation: ad.AnnData, result: TreeNBResult) -> float:
    """Score unseen donors after integrating fitted donor effects by their mean."""
    taxonomy_cols = ("Class", "Subclass", "Group")
    pb = build_pseudobulk(
        validation.obs,
        taxonomy_col="Group",
        species_col="species",
        donor_col="donor",
        min_cells_per_pseudobulk=2,
    )
    tree = build_taxonomy_tree_from_obs(validation.obs, list(taxonomy_cols))
    species_design = build_species_tree_design(
        "(Human,Macaque,Mouse);", ["Human", "Macaque", "Mouse"]
    )
    designs, _, _ = _build_design_matrices(
        pb,
        tree,
        species_design,
        list(taxonomy_cols),
        "species",
        None,
        "donor",
        orthogonal_tree=result.orthogonal_tree,
    )
    if result.intercept is None or result.log_theta_baseline is None:
        raise RuntimeError("Tuning requires retained intercept and dispersion estimates.")
    X = cast(Any, validation.X)
    counts = cast(FloatArray, aggregate_chunk(X, pb.cell_to_group))
    if sparse.issparse(X):
        cell_totals = cast(FloatArray, np.asarray(X.sum(axis=1)).ravel())
    else:
        cell_totals = cast(FloatArray, np.asarray(X).sum(axis=1))
    library_sizes = cast(
        FloatArray, np.asarray(pb.cell_to_group.T @ cell_totals).ravel()
    )
    eta = np.log(library_sizes[:, None] + 1e-8) + result.intercept[None, :]
    for family, design in designs.items():
        if family == "donor":
            continue
        beta = result.coefficients[family]
        if design.shape[1] != beta.shape[0]:
            raise RuntimeError(f"Validation design mismatch for {family}.")
        eta += design @ beta
    if result.designs is not None and "donor" in result.designs:
        donor_mean = (
            result.designs["donor"] @ result.coefficients["donor"]
        ).mean(axis=0)
        eta += donor_mean[None, :]
    mu = np.exp(np.clip(eta, -20.0, 20.0))
    theta = np.exp(np.clip(result.log_theta_baseline, -5.0, 10.0))[None, :]
    nll = (
        -gammaln(counts + theta)
        + gammaln(theta)
        + gammaln(counts + 1.0)
        - theta * np.log(theta / (theta + mu) + 1e-8)
        - counts * np.log(mu / (theta + mu) + 1e-8)
    )
    return float(nll.mean())


def _fit_l1(
    *,
    simulation: KnownTruthSimulation,
    adata: ad.AnnData,
    penalty: float,
    orthogonal_tree: bool,
    max_iter: int,
) -> TreeNBResult:
    """Fit a fixed-L1 model under one basis and penalty."""
    return fit_tree_nb(
        adata,
        taxonomy_cols=list(simulation.taxonomy_cols),
        species_col="species",
        species_tree=simulation.species_tree,
        batch_col=None,
        donor_col="donor",
        min_cells_per_pseudobulk=2,
        global_lambda=penalty,
        max_iter=max_iter,
        progress=False,
        refit_support=False,
        orthogonal_tree=orthogonal_tree,
    )


def _fit_eb(
    *, simulation: KnownTruthSimulation, method: str, max_iter: int
) -> TreeNBResult:
    """Fit one empirical-Bayes comparator on the full simulation."""
    config = _method_config(
        method=method,
        pilot_genes=simulation.adata.n_vars,
        max_iter=max_iter,
    )
    if config is None:
        raise ValueError("An empirical-Bayes method is required.")
    return fit_tree_nb(
        simulation.adata,
        taxonomy_cols=list(simulation.taxonomy_cols),
        species_col="species",
        species_tree=simulation.species_tree,
        batch_col=None,
        donor_col="donor",
        min_cells_per_pseudobulk=2,
        global_lambda=0.05,
        max_iter=max_iter,
        progress=False,
        refit_support=False,
        orthogonal_tree=True,
        empirical_bayes=config,
    )


def _invariant_metrics(
    *,
    simulation_index: int,
    regime: EffectRegime,
    method: str,
    simulation: KnownTruthSimulation,
    result: TreeNBResult,
) -> list[dict[str, float | int | str]]:
    """Score family contributions in observation space, independent of basis."""
    if result.designs is None:
        raise RuntimeError("Invariant scoring requires retained designs.")
    records: list[dict[str, float | int | str]] = []
    fitted_strength: dict[str, FloatArray] = {}
    true_strength: dict[str, FloatArray] = {}
    for family, truth in simulation.coefficients.items():
        true_effect = simulation.designs[family] @ truth
        fitted_effect = result.designs[family] @ result.coefficients[family]
        difference = fitted_effect - true_effect
        true_scale = float(np.sqrt(np.mean(np.square(true_effect))))
        fitted_scale = float(np.sqrt(np.mean(np.square(fitted_effect))))
        correlation = float(
            np.corrcoef(true_effect.ravel(), fitted_effect.ravel())[0, 1]
        )
        fitted_strength[family] = np.sqrt(np.mean(np.square(fitted_effect), axis=0))
        true_strength[family] = np.sqrt(np.mean(np.square(true_effect), axis=0))
        records.append(
            {
                "simulation": simulation_index,
                "regime": regime.value,
                "method": method,
                "family": family,
                "level": family.removeprefix("species_tax_"),
                "effect_rmse": float(np.sqrt(np.mean(np.square(difference)))),
                "normalized_effect_rmse": float(
                    np.sqrt(np.mean(np.square(difference))) / true_scale
                ),
                "effect_correlation": correlation,
                "magnitude_ratio": fitted_scale / true_scale,
                "true_burden": true_scale**2,
                "estimated_burden": fitted_scale**2,
                "burden_ratio": (fitted_scale / true_scale) ** 2,
            }
        )
    localization = float("nan")
    if regime is EffectRegime.SPARSE_LARGE:
        families = tuple(simulation.coefficients)
        correct: list[bool] = []
        for gene in range(simulation.adata.n_vars):
            true_values = np.asarray([true_strength[f][gene] for f in families])
            if np.count_nonzero(true_values > 1e-8) != 1:
                continue
            fitted_values = np.asarray([fitted_strength[f][gene] for f in families])
            correct.append(int(np.argmax(fitted_values)) == int(np.argmax(true_values)))
        localization = float(np.mean(correct))
    for record in records:
        record["level_localization_rate"] = localization
    return records


def run_l1_tuning_comparison(
    *,
    n_simulations: int = 10,
    penalties: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05, 0.09, 0.15, 0.30),
    random_state: int = 2601,
    max_iter: int = 200,
) -> L1TuningComparisonResult:
    """Tune L1 by held-out-donor likelihood, then compare invariant recovery."""
    if n_simulations < 1:
        raise ValueError("n_simulations must be positive.")
    if not penalties or any(value <= 0.0 for value in penalties):
        raise ValueError("penalties must contain only positive values.")
    rng = np.random.default_rng(random_state)
    simulations: list[tuple[EffectRegime, KnownTruthSimulation]] = []
    tuning_records: list[dict[str, float | int | str | bool]] = []
    for simulation_index in range(n_simulations):
        for regime in EffectRegime:
            simulation = make_known_truth_simulation(
                seed=int(rng.integers(0, np.iinfo(np.int32).max)),
                regime=regime,
            )
            simulations.append((regime, simulation))
            training, validation = _donor_split(adata=simulation.adata, rng=rng)
            for basis, orthogonal in (("legacy", False), ("orthogonal", True)):
                for penalty in penalties:
                    fit = _fit_l1(
                        simulation=simulation,
                        adata=training,
                        penalty=penalty,
                        orthogonal_tree=orthogonal,
                        max_iter=max_iter,
                    )
                    tuning_records.append(
                        {
                            "simulation": simulation_index,
                            "regime": regime.value,
                            "basis": basis,
                            "orthogonal_tree": orthogonal,
                            "penalty": penalty,
                            "validation_nll": _mean_nb_nll(
                                validation=validation, result=fit
                            ),
                        }
                    )
    tuning_curve = pd.DataFrame(tuning_records)
    pooled = (
        tuning_curve.groupby(["basis", "penalty"], as_index=False)["validation_nll"]
        .mean()
        .sort_values(["basis", "validation_nll", "penalty"])
    )
    selected_lambdas = {
        str(basis): float(frame.iloc[0]["penalty"])
        for basis, frame in pooled.groupby("basis", sort=True)
    }
    metric_records: list[dict[str, float | int | str]] = []
    for index, (regime, simulation) in enumerate(simulations):
        simulation_index = index // len(EffectRegime)
        fit_specs = (
            ("legacy_l1_historical", False, 0.09),
            ("legacy_l1_tuned", False, selected_lambdas["legacy"]),
            ("orthogonal_l1_fixed", True, 0.05),
            ("orthogonal_l1_tuned", True, selected_lambdas["orthogonal"]),
        )
        for method, orthogonal, penalty in fit_specs:
            fit = _fit_l1(
                simulation=simulation,
                adata=simulation.adata,
                penalty=penalty,
                orthogonal_tree=orthogonal,
                max_iter=max_iter,
            )
            metric_records.extend(
                _invariant_metrics(
                    simulation_index=simulation_index,
                    regime=regime,
                    method=method,
                    simulation=simulation,
                    result=fit,
                )
            )
        for prior in ("gaussian", "laplace", "spike_slab"):
            fit = _fit_eb(
                simulation=simulation, method=prior, max_iter=max_iter
            )
            metric_records.extend(
                _invariant_metrics(
                    simulation_index=simulation_index,
                    regime=regime,
                    method=f"{prior}_eb",
                    simulation=simulation,
                    result=fit,
                )
            )
    metrics = pd.DataFrame(metric_records)
    method_comparison = (
        metrics.assign(
            log_burden_error=lambda frame: np.abs(np.log(frame["burden_ratio"])),
            correlation_error=lambda frame: 1.0 - frame["effect_correlation"],
        )
        .groupby("method", as_index=False)[
            [
                "normalized_effect_rmse",
                "log_burden_error",
                "correlation_error",
                "level_localization_rate",
            ]
        ]
        .mean()
    )
    method_comparison["reconstruction_score"] = method_comparison[
        ["normalized_effect_rmse", "log_burden_error", "correlation_error"]
    ].sum(axis=1)
    method_comparison = method_comparison.sort_values(
        "reconstruction_score"
    ).reset_index(drop=True)
    return L1TuningComparisonResult(
        tuning_curve=tuning_curve,
        metrics=metrics,
        method_comparison=method_comparison,
        selected_lambdas=selected_lambdas,
        n_simulations=n_simulations,
    )
