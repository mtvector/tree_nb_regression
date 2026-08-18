"""Donor-stratified bootstrap inference for evolutionary-burden ratios."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Any, cast

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import norm

from .model import fit_tree_nb
from .shrinkage import evolutionary_burden


class BootstrapIntervalMethod(str, Enum):
    """Bootstrap interval transformations supported for burden ratios."""

    PERCENTILE = "percentile"
    BASIC = "basic"
    SIMULTANEOUS_PERCENTILE = "simultaneous_percentile"
    SIMULTANEOUS_BCA = "simultaneous_bca"


@dataclass(frozen=True)
class BurdenFitConfig:
    """Tree-fit settings reproduced for every donor-bootstrap replicate."""

    global_lambda: float | None = None
    gene_chunk_size: int = 512
    min_cells_per_pseudobulk: int = 10
    max_iter: int = 500
    device: str = "cpu"
    refit_support: bool = True
    relax_species_tax_refit: bool = True

    def __post_init__(self) -> None:
        if self.global_lambda is not None and self.global_lambda <= 0.0:
            raise ValueError("global_lambda must be positive when provided.")
        if self.gene_chunk_size < 1:
            raise ValueError("gene_chunk_size must be positive.")
        if self.min_cells_per_pseudobulk < 1:
            raise ValueError("min_cells_per_pseudobulk must be positive.")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive.")


@dataclass(frozen=True)
class BurdenBootstrapConfig:
    """Resampling and interval settings for donor-level burden inference."""

    n_bootstrap: int = 200
    confidence_level: float = 0.95
    random_state: int = 0
    minimum_success_fraction: float = 0.9
    interval_method: BootstrapIntervalMethod = BootstrapIntervalMethod.SIMULTANEOUS_BCA
    log_interval_inflation: float = 1.2

    def __post_init__(self) -> None:
        if self.n_bootstrap < 2:
            raise ValueError("n_bootstrap must be at least two.")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must lie in (0, 1).")
        if not 0.0 < self.minimum_success_fraction <= 1.0:
            raise ValueError("minimum_success_fraction must lie in (0, 1].")
        if self.log_interval_inflation < 1.0:
            raise ValueError("log_interval_inflation must be at least one.")


@dataclass(frozen=True)
class BurdenBootstrapResult:
    """Point burdens, bootstrap replicates, ratio intervals, and failures."""

    point_burdens: pd.DataFrame
    bootstrap_burdens: pd.DataFrame
    ratios: pd.DataFrame
    failures: pd.DataFrame
    jackknife_burdens: pd.DataFrame
    n_successful: int
    n_requested: int


def _donor_species_table(*, obs: pd.DataFrame, donor_col: str, species_col: str) -> pd.DataFrame:
    """Return one species label per donor or reject cross-species donor labels."""
    required = [donor_col, species_col]
    missing = [column for column in required if column not in obs]
    if missing:
        raise KeyError(f"obs is missing required columns: {missing}")
    table = (
        obs[[donor_col, species_col]]
        .astype(str)
        .drop_duplicates()
        .groupby(donor_col, sort=True)[species_col]
        .agg(list)
    )
    invalid = table[table.map(len) != 1]
    if not invalid.empty:
        raise ValueError("Every donor must have exactly one species label.")
    result = table.map(lambda values: values[0]).rename(species_col).reset_index()
    counts = result.groupby(species_col)[donor_col].nunique()
    if (counts < 2).any():
        bad = counts[counts < 2].index.astype(str).tolist()
        raise ValueError(f"At least two donors per species are required; got {bad}.")
    return result


def _resample_donors(
    *,
    adata: ad.AnnData,
    donor_col: str,
    species_col: str,
    donor_species: pd.DataFrame,
    rng: np.random.Generator,
) -> ad.AnnData:
    """Sample donors within species and give every duplicate a new donor identity."""
    obs = cast(pd.DataFrame, adata.obs)
    pieces: list[ad.AnnData] = []
    for species, frame in donor_species.groupby(species_col, sort=True):
        donors = np.asarray(sorted(frame[donor_col].astype(str)), dtype=object)
        sampled = rng.choice(donors, size=donors.size, replace=True)
        for draw_index, donor in enumerate(sampled):
            mask = obs[donor_col].astype(str).to_numpy() == str(donor)
            piece = adata[mask].copy()
            bootstrap_donor = f"bootstrap_{species}_{draw_index}_{donor}"
            piece.obs[donor_col] = bootstrap_donor
            pieces.append(piece)
    if not pieces:
        raise RuntimeError("Bootstrap resampling produced no donor cells.")
    return ad.concat(pieces, join="inner", merge="same", index_unique="-")


def _drop_one_donor(*, adata: ad.AnnData, donor_col: str, donor: str) -> ad.AnnData:
    """Return a donor-deleted dataset for a jackknife influence estimate."""
    obs = cast(pd.DataFrame, adata.obs)
    keep = obs[donor_col].astype(str).to_numpy() != donor
    return cast(ad.AnnData, adata[keep].copy())


def _fit_burden_model(
    *,
    adata: ad.AnnData,
    taxonomy_cols: tuple[str, ...],
    species_col: str,
    species_tree: str | tuple[Any, ...],
    donor_col: str,
    batch_col: str | None,
    counts_layer: str | None,
    fit_config: BurdenFitConfig,
) -> pd.DataFrame:
    """Fit the calibrated orthogonal-L1 model and return level burdens."""
    result = fit_tree_nb(
        adata,
        taxonomy_cols=list(taxonomy_cols),
        species_col=species_col,
        species_tree=species_tree,
        batch_col=batch_col,
        donor_col=donor_col,
        counts_layer=counts_layer,
        gene_chunk_size=fit_config.gene_chunk_size,
        min_cells_per_pseudobulk=fit_config.min_cells_per_pseudobulk,
        global_lambda=fit_config.global_lambda,
        max_iter=fit_config.max_iter,
        device=fit_config.device,
        refit_support=fit_config.refit_support,
        refit_all_species_tax=fit_config.relax_species_tax_refit,
        progress=False,
        orthogonal_tree=True,
    )
    return evolutionary_burden(result)


def _summarize_ratios(
    *,
    point_burdens: pd.DataFrame,
    bootstrap_burdens: pd.DataFrame,
    jackknife_burdens: pd.DataFrame,
    confidence_level: float,
    interval_method: BootstrapIntervalMethod,
    log_interval_inflation: float,
) -> pd.DataFrame:
    """Summarize all pairwise burden ratios on the log scale."""
    point = point_burdens.set_index("level")["variance_burden"]
    wide = bootstrap_burdens.pivot(index="bootstrap", columns="level", values="variance_burden")
    alpha = 1.0 - confidence_level
    rows: list[dict[str, float | int | str]] = []
    pairs = list(combinations(sorted(point.index.astype(str)), 2))
    interval_alpha = (
        alpha / len(pairs)
        if interval_method
        in {
            BootstrapIntervalMethod.SIMULTANEOUS_PERCENTILE,
            BootstrapIntervalMethod.SIMULTANEOUS_BCA,
        }
        else alpha
    )
    for numerator, denominator in pairs:
        point_num = float(point[numerator])
        point_den = float(point[denominator])
        record: dict[str, float | int | str] = {
            "numerator_level": numerator,
            "denominator_level": denominator,
            "numerator_burden": point_num,
            "denominator_burden": point_den,
            "n_bootstrap": 0,
            "status": "ok",
            "interval_method": interval_method.value,
        }
        values = wide[[numerator, denominator]].dropna()
        positive = values[(values[numerator] > 0.0) & (values[denominator] > 0.0)]
        if point_num <= 0.0 or point_den <= 0.0 or positive.empty:
            record.update(
                {
                    "burden_ratio": float("nan"),
                    "log_burden_ratio": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "probability_numerator_greater": float("nan"),
                    "n_bootstrap": int(len(positive)),
                    "status": "zero_or_nonpositive_burden",
                }
            )
            rows.append(record)
            continue
        log_ratios = np.log(positive[numerator].to_numpy() / positive[denominator].to_numpy())
        lower_quantile, upper_quantile = np.quantile(
            log_ratios, [interval_alpha / 2.0, 1.0 - interval_alpha / 2.0]
        )
        point_log_ratio = float(np.log(point_num / point_den))
        if interval_method in {
            BootstrapIntervalMethod.PERCENTILE,
            BootstrapIntervalMethod.SIMULTANEOUS_PERCENTILE,
        }:
            ci_low, ci_high = np.exp([lower_quantile, upper_quantile])
        elif interval_method is BootstrapIntervalMethod.BASIC:
            ci_low, ci_high = np.exp(
                [2.0 * point_log_ratio - upper_quantile, 2.0 * point_log_ratio - lower_quantile]
            )
        else:
            jackknife = jackknife_burdens.pivot(
                index="jackknife", columns="level", values="variance_burden"
            )
            jackknife = jackknife[[numerator, denominator]].dropna()
            jackknife = jackknife[(jackknife[numerator] > 0.0) & (jackknife[denominator] > 0.0)]
            if len(jackknife) < 3:
                raise RuntimeError("BCa intervals require at least three valid jackknife fits.")
            jackknife_log_ratios = np.log(
                jackknife[numerator].to_numpy() / jackknife[denominator].to_numpy()
            )
            less_than_point = float(np.mean(log_ratios < point_log_ratio))
            clipped_proportion = float(
                np.clip(less_than_point, 0.5 / len(log_ratios), 1.0 - 0.5 / len(log_ratios))
            )
            bias_correction = float(norm.ppf(clipped_proportion))
            jackknife_mean = float(jackknife_log_ratios.mean())
            influence = jackknife_mean - jackknife_log_ratios
            denominator_acceleration = 6.0 * float(np.sum(influence**2)) ** 1.5
            acceleration = (
                float(np.sum(influence**3) / denominator_acceleration)
                if denominator_acceleration > 0.0
                else 0.0
            )
            normal_quantiles = norm.ppf([interval_alpha / 2.0, 1.0 - interval_alpha / 2.0])
            adjusted_quantiles = norm.cdf(
                bias_correction
                + (bias_correction + normal_quantiles)
                / (1.0 - acceleration * (bias_correction + normal_quantiles))
            )
            adjusted_quantiles = np.clip(
                adjusted_quantiles, 1.0 / len(log_ratios), 1.0 - 1.0 / len(log_ratios)
            )
            ci_low, ci_high = np.exp(np.quantile(log_ratios, adjusted_quantiles))
        ci_log_low, ci_log_high = np.log([ci_low, ci_high])
        ci_low, ci_high = np.exp(
            point_log_ratio
            + log_interval_inflation
            * np.array([ci_log_low - point_log_ratio, ci_log_high - point_log_ratio])
        )
        record.update(
            {
                "burden_ratio": point_num / point_den,
                "log_burden_ratio": point_log_ratio,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "probability_numerator_greater": float(np.mean(log_ratios > 0.0)),
                "n_bootstrap": int(log_ratios.size),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def bootstrap_evolutionary_burden(
    adata: ad.AnnData,
    *,
    taxonomy_cols: tuple[str, ...],
    species_col: str,
    species_tree: str | tuple[Any, ...],
    donor_col: str,
    batch_col: str | None = None,
    counts_layer: str | None = None,
    fit_config: BurdenFitConfig = BurdenFitConfig(),
    bootstrap_config: BurdenBootstrapConfig = BurdenBootstrapConfig(),
) -> BurdenBootstrapResult:
    """Refit orthogonal L1 on species-stratified donor bootstraps for ratios.

    Every replicate resamples whole donors within species and reruns the full
    selection and optional post-selection-refit pipeline. The default uses
    Bonferroni-adjusted BCa bands over all level pairs, giving a 95% familywise
    interval set that includes variability from L1 support selection.
    """
    obs = cast(pd.DataFrame, adata.obs)
    donor_species = _donor_species_table(obs=obs, donor_col=donor_col, species_col=species_col)
    point_burdens = _fit_burden_model(
        adata=adata,
        taxonomy_cols=taxonomy_cols,
        species_col=species_col,
        species_tree=species_tree,
        donor_col=donor_col,
        batch_col=batch_col,
        counts_layer=counts_layer,
        fit_config=fit_config,
    )
    rng = np.random.default_rng(bootstrap_config.random_state)
    records: list[pd.DataFrame] = []
    failures: list[dict[str, int | str]] = []
    for bootstrap_index in range(bootstrap_config.n_bootstrap):
        try:
            sampled = _resample_donors(
                adata=adata,
                donor_col=donor_col,
                species_col=species_col,
                donor_species=donor_species,
                rng=rng,
            )
            burden = _fit_burden_model(
                adata=sampled,
                taxonomy_cols=taxonomy_cols,
                species_col=species_col,
                species_tree=species_tree,
                donor_col=donor_col,
                batch_col=batch_col,
                counts_layer=counts_layer,
                fit_config=fit_config,
            )
            burden["bootstrap"] = bootstrap_index
            records.append(burden)
        except (RuntimeError, ValueError, KeyError) as error:
            failures.append({"bootstrap": bootstrap_index, "error": str(error)})
    bootstrap_burdens = (
        pd.concat(records, ignore_index=True)
        if records
        else pd.DataFrame(columns=["bootstrap", "level", "variance_burden"])
    )
    n_successful = len(records)
    required = int(
        np.ceil(bootstrap_config.minimum_success_fraction * bootstrap_config.n_bootstrap)
    )
    if n_successful < required:
        raise RuntimeError(
            f"Only {n_successful}/{bootstrap_config.n_bootstrap} bootstrap fits "
            f"succeeded; require at least {required}."
        )
    jackknife_records: list[pd.DataFrame] = []
    if bootstrap_config.interval_method is BootstrapIntervalMethod.SIMULTANEOUS_BCA:
        for jackknife_index, donor in enumerate(donor_species[donor_col].astype(str)):
            deleted = _drop_one_donor(adata=adata, donor_col=donor_col, donor=donor)
            burden = _fit_burden_model(
                adata=deleted,
                taxonomy_cols=taxonomy_cols,
                species_col=species_col,
                species_tree=species_tree,
                donor_col=donor_col,
                batch_col=batch_col,
                counts_layer=counts_layer,
                fit_config=fit_config,
            )
            burden["jackknife"] = jackknife_index
            jackknife_records.append(burden)
    jackknife_burdens = (
        pd.concat(jackknife_records, ignore_index=True)
        if jackknife_records
        else pd.DataFrame(columns=["jackknife", "level", "variance_burden"])
    )
    ratios = _summarize_ratios(
        point_burdens=point_burdens,
        bootstrap_burdens=bootstrap_burdens,
        jackknife_burdens=jackknife_burdens,
        confidence_level=bootstrap_config.confidence_level,
        interval_method=bootstrap_config.interval_method,
        log_interval_inflation=bootstrap_config.log_interval_inflation,
    )
    return BurdenBootstrapResult(
        point_burdens=point_burdens,
        bootstrap_burdens=bootstrap_burdens,
        ratios=ratios,
        failures=pd.DataFrame(failures, columns=["bootstrap", "error"]),
        jackknife_burdens=jackknife_burdens,
        n_successful=n_successful,
        n_requested=bootstrap_config.n_bootstrap,
    )
