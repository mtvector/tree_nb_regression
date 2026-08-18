"""Donor-honest post-selection intervals for species-by-taxonomy contrasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse
from scipy.stats import t as student_t

from .inference import add_bh_qvalues
from .model import fit_tree_nb
from .results import TreeNBResult
from .shrinkage import EmpiricalBayesConfig

type FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DonorSelectionConfig:
    """Configuration for the independent tree-model selection stage."""

    global_lambda: float = 0.05
    gene_chunk_size: int = 512
    max_iter: int = 500
    min_cells_per_pseudobulk: int = 10
    device: str = "cpu"
    empirical_bayes: EmpiricalBayesConfig | None = None
    localize_with_training_contrast: bool = True


@dataclass
class DonorHonestContrastResult:
    """Selected contrasts, held-out intervals, and split provenance."""

    intervals: pd.DataFrame
    selection_result: TreeNBResult
    training_donors: tuple[str, ...]
    inference_donors: tuple[str, ...]
    contrast_scale: str = "log donor-level counts per million"

    @property
    def discoverable(self) -> pd.DataFrame:
        """Return intervals that are estimable from held-out donors."""
        return self.intervals[self.intervals["status"] == "ok"].copy()


def _validate_donor_species(*, obs: pd.DataFrame, donor_col: str, species_col: str) -> pd.DataFrame:
    """Return one species label per donor or raise for cross-species donors."""
    required = [donor_col, species_col]
    missing = [column for column in required if column not in obs.columns]
    if missing:
        raise KeyError(f"obs is missing required columns: {missing}")
    donor_species = (
        obs[[donor_col, species_col]]
        .astype(str)
        .drop_duplicates()
        .groupby(donor_col, sort=True)[species_col]
        .agg(list)
    )
    invalid = donor_species[donor_species.map(len) != 1]
    if not invalid.empty:
        raise ValueError(
            "Every donor must belong to exactly one species for donor-honest "
            f"splitting; invalid donors: {invalid.index.tolist()}"
        )
    return donor_species.map(lambda values: values[0]).rename(species_col).reset_index()


def _stratified_donor_split(
    *,
    donor_species: pd.DataFrame,
    donor_col: str,
    species_col: str,
    train_fraction: float,
    random_state: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split donors independently within species for selection and inference."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1.")
    rng = np.random.default_rng(random_state)
    training: list[str] = []
    inference: list[str] = []
    for _, sub in donor_species.groupby(species_col, sort=True):
        donors = np.asarray(sorted(sub[donor_col].astype(str)), dtype=object)
        if donors.size < 2:
            raise ValueError(
                "Each species needs at least two donors to form independent "
                "selection and inference sets."
            )
        n_training = int(np.floor(train_fraction * donors.size))
        n_training = min(max(n_training, 1), donors.size - 1)
        selected = rng.choice(donors, size=n_training, replace=False)
        training.extend(str(donor) for donor in selected)
        inference.extend(str(donor) for donor in donors if donor not in set(selected))
    return tuple(sorted(training)), tuple(sorted(inference))


def _selected_species_tax_candidates(*, result: TreeNBResult, threshold: float) -> pd.DataFrame:
    """Materialize interaction coefficients selected on training donors."""
    if result.species_tax_meta is None:
        return pd.DataFrame(
            columns=[
                "family",
                "index",
                "gene_index",
                "gene",
                "level",
                "node_id",
                "node_label",
                "species",
                "n_species_at_node",
                "selection_beta",
            ]
        )
    rows: list[dict[str, Any]] = []
    for family, metadata in result.species_tax_meta.items():
        coefficients = result.coefficients[family]
        selected = np.abs(coefficients) > threshold
        for row in metadata.itertuples(index=False):
            column_index = int(row.col_index)
            for gene_index in np.flatnonzero(selected[column_index]):
                rows.append(
                    {
                        "family": family,
                        "index": column_index,
                        "gene_index": int(gene_index),
                        "gene": result.gene_names[int(gene_index)],
                        "level": str(row.level),
                        "node_id": str(row.node_id),
                        "node_label": str(row.node_label),
                        "species": str(row.species),
                        "n_species_at_node": int(row.n_species_at_node),
                        "selection_beta": float(coefficients[column_index, gene_index]),
                    }
                )
    return pd.DataFrame(rows)


def _donor_log_cpm(
    *,
    adata: Any,
    cell_mask: NDArray[np.bool_],
    donor_col: str,
    species_col: str,
    gene_index: int,
    counts_layer: str | None,
    pseudocount: float,
) -> pd.DataFrame:
    """Aggregate a node's cells to donor-level log counts-per-million values."""
    obs = adata.obs.reset_index(drop=True)
    selected_obs = obs.loc[cell_mask, [donor_col, species_col]].astype(str)
    if selected_obs.empty:
        return pd.DataFrame(columns=[donor_col, species_col, "log_cpm"])
    X = adata.layers[counts_layer] if counts_layer is not None else adata.X
    X_node = X[cell_mask]
    if sparse.issparse(X_node):
        library_sizes = np.asarray(X_node.sum(axis=1)).ravel()
        gene_counts = np.asarray(X_node[:, gene_index].toarray()).ravel()
    else:
        dense_node = np.asarray(X_node)
        library_sizes = dense_node.sum(axis=1)
        gene_counts = dense_node[:, gene_index]
    frame = selected_obs.copy()
    frame["library_size"] = library_sizes
    frame["gene_count"] = gene_counts
    donor_sums = frame.groupby([donor_col, species_col], sort=True)[
        ["library_size", "gene_count"]
    ].sum()
    log_cpm = np.log(
        (donor_sums["gene_count"] + pseudocount)
        / (donor_sums["library_size"] + pseudocount)
        * 1_000_000.0
    )
    return log_cpm.rename("log_cpm").reset_index()


def _welch_contrast(
    *,
    donor_values: pd.DataFrame,
    donor_col: str,
    species_col: str,
    target_species: str,
    expected_species: Sequence[str],
    confidence_level: float,
) -> dict[str, float | int | str]:
    """Estimate one species-versus-mean-of-others donor-level contrast."""
    available = set(donor_values[species_col].astype(str))
    required = set(expected_species)
    if not required.issubset(available):
        return {"status": "species_absent_at_node"}
    means: dict[str, float] = {}
    variances: dict[str, float] = {}
    sizes: dict[str, int] = {}
    for species in expected_species:
        values = donor_values.loc[donor_values[species_col] == species, "log_cpm"].to_numpy(
            dtype=float
        )
        if values.size < 2:
            return {"status": "insufficient_inference_donors"}
        means[species] = float(np.mean(values))
        variances[species] = float(np.var(values, ddof=1))
        sizes[species] = int(values.size)
    other_species = [species for species in expected_species if species != target_species]
    if not other_species:
        return {"status": "no_comparison_species"}
    other_mean = float(np.mean([means[species] for species in other_species]))
    estimate = means[target_species] - other_mean
    variance_terms = [variances[target_species] / sizes[target_species]]
    variance_terms.extend(
        variances[species] / (sizes[species] * len(other_species) ** 2) for species in other_species
    )
    variance = float(np.sum(variance_terms))
    if not np.isfinite(variance) or variance <= 0.0:
        return {"status": "degenerate_donor_variance"}
    numerator = variance**2
    denominator = (variance_terms[0] ** 2) / (sizes[target_species] - 1)
    denominator += sum(
        term**2 / (sizes[species] - 1)
        for term, species in zip(variance_terms[1:], other_species, strict=True)
    )
    if denominator <= 0.0:
        return {"status": "degenerate_donor_variance"}
    degrees_freedom = numerator / denominator
    standard_error = float(np.sqrt(variance))
    critical = float(student_t.ppf((1.0 + confidence_level) / 2.0, degrees_freedom))
    statistic = estimate / standard_error
    p_value = float(2.0 * student_t.sf(abs(statistic), degrees_freedom))
    return {
        "status": "ok",
        "estimate": estimate,
        "se": standard_error,
        "df": degrees_freedom,
        "ci_lo": estimate - critical * standard_error,
        "ci_hi": estimate + critical * standard_error,
        "statistic": statistic,
        "p": p_value,
        "n_target_donors": sizes[target_species],
        "n_other_donors": int(sum(sizes[species] for species in other_species)),
    }


def _score_training_candidates(
    *,
    adata: Any,
    candidates: pd.DataFrame,
    taxonomy_cols: tuple[str, ...],
    donor_col: str,
    species_col: str,
    counts_layer: str | None,
    pseudocount: float,
) -> pd.DataFrame:
    """Score candidates by their selection-donor contrast.

    This ranking uses only the selection donors, so it cannot invalidate the
    held-out interval or its p-value. It makes node localization match the
    donor-level estimand rather than a post-refit coefficient magnitude.
    """
    obs = adata.obs.reset_index(drop=True)
    species = tuple(sorted(obs[species_col].astype(str).unique()))
    ancestor_by_level = {
        level: obs[list(taxonomy_cols[: index + 1])].astype(str).agg("/".join, axis=1)
        for index, level in enumerate(taxonomy_cols)
    }
    scores: list[float] = []
    for candidate in candidates.to_dict("records"):
        node_mask = (
            ancestor_by_level[str(candidate["level"])] == str(candidate["node_id"])
        ).to_numpy()
        values = _donor_log_cpm(
            adata=adata,
            cell_mask=node_mask,
            donor_col=donor_col,
            species_col=species_col,
            gene_index=int(candidate["gene_index"]),
            counts_layer=counts_layer,
            pseudocount=pseudocount,
        )
        contrast = _welch_contrast(
            donor_values=values,
            donor_col=donor_col,
            species_col=species_col,
            target_species=str(candidate["species"]),
            expected_species=species,
            confidence_level=0.95,
        )
        scores.append(abs(float(contrast.get("estimate", 0.0))))
    ranked = candidates.assign(selection_contrast_score=scores)
    keys = ["family", "level", "species", "gene"]
    ranked["selection_contrast_rank"] = (
        ranked.groupby(keys)["selection_contrast_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return ranked


def donor_honest_intervals(
    adata: Any,
    *,
    taxonomy_cols: Sequence[str],
    species_col: str,
    species_tree: str | tuple[Any, ...],
    donor_col: str,
    counts_layer: str | None = None,
    batch_col: str | None = None,
    selection: DonorSelectionConfig = DonorSelectionConfig(),
    train_fraction: float = 0.5,
    random_state: int = 0,
    selection_threshold: float = 0.01,
    confidence_level: float = 0.95,
    pseudocount: float = 0.5,
) -> DonorHonestContrastResult:
    """Select tree interactions on training donors and infer held-out contrasts.

    The estimand is a species' mean donor-level log counts-per-million within a
    taxonomy node minus the equally weighted mean across the other species at
    that node. Selection and inference use disjoint whole-donor sets.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1.")
    if selection_threshold < 0.0:
        raise ValueError("selection_threshold must be nonnegative.")
    if pseudocount <= 0.0:
        raise ValueError("pseudocount must be positive.")
    taxonomy_columns = tuple(taxonomy_cols)
    if not taxonomy_columns:
        raise ValueError("taxonomy_cols cannot be empty.")
    required_columns = [*taxonomy_columns, species_col, donor_col]
    missing_columns = [column for column in required_columns if column not in adata.obs]
    if missing_columns:
        raise KeyError(f"obs is missing required columns: {missing_columns}")
    if counts_layer is not None and counts_layer not in adata.layers:
        raise KeyError(f"counts layer '{counts_layer}' was not found.")

    donor_species = _validate_donor_species(
        obs=adata.obs, donor_col=donor_col, species_col=species_col
    )
    training_donors, inference_donors = _stratified_donor_split(
        donor_species=donor_species,
        donor_col=donor_col,
        species_col=species_col,
        train_fraction=train_fraction,
        random_state=random_state,
    )
    donor_series = adata.obs[donor_col].astype(str)
    training_data = adata[donor_series.isin(training_donors).to_numpy()].copy()
    inference_mask = donor_series.isin(inference_donors).to_numpy()
    selection_result = fit_tree_nb(
        training_data,
        taxonomy_cols=list(taxonomy_columns),
        species_col=species_col,
        species_tree=species_tree,
        batch_col=batch_col,
        donor_col=donor_col,
        counts_layer=counts_layer,
        gene_chunk_size=selection.gene_chunk_size,
        min_cells_per_pseudobulk=selection.min_cells_per_pseudobulk,
        global_lambda=selection.global_lambda,
        max_iter=selection.max_iter,
        device=selection.device,
        refit_support=True,
        fit_dispersion_tree=False,
        progress=False,
        keep_design_artifacts=False,
        empirical_bayes=selection.empirical_bayes,
    )
    candidates = _selected_species_tax_candidates(
        result=selection_result, threshold=selection_threshold
    )
    if not candidates.empty and selection.localize_with_training_contrast:
        candidates = _score_training_candidates(
            adata=training_data,
            candidates=candidates,
            taxonomy_cols=taxonomy_columns,
            donor_col=donor_col,
            species_col=species_col,
            counts_layer=counts_layer,
            pseudocount=pseudocount,
        )
    if candidates.empty:
        candidates["status"] = pd.Series(dtype=str)
        candidates["q"] = pd.Series(dtype=float)
        return DonorHonestContrastResult(
            intervals=candidates,
            selection_result=selection_result,
            training_donors=training_donors,
            inference_donors=inference_donors,
        )

    obs = adata.obs.reset_index(drop=True)
    all_species = tuple(sorted(donor_species[species_col].astype(str).unique()))
    ancestor_by_level = {
        level: obs[list(taxonomy_columns[: index + 1])].astype(str).agg("/".join, axis=1)
        for index, level in enumerate(taxonomy_columns)
    }
    rows: list[dict[str, Any]] = []
    for candidate in candidates.to_dict("records"):
        node_mask = (ancestor_by_level[candidate["level"]] == candidate["node_id"]).to_numpy()
        donor_values = _donor_log_cpm(
            adata=adata,
            cell_mask=node_mask & inference_mask,
            donor_col=donor_col,
            species_col=species_col,
            gene_index=int(candidate["gene_index"]),
            counts_layer=counts_layer,
            pseudocount=pseudocount,
        )
        contrast = _welch_contrast(
            donor_values=donor_values,
            donor_col=donor_col,
            species_col=species_col,
            target_species=str(candidate["species"]),
            expected_species=all_species,
            confidence_level=confidence_level,
        )
        rows.append({**candidate, **contrast})
    intervals = pd.DataFrame(rows)
    intervals["contrast_id"] = (
        intervals["gene"].astype(str)
        + "|"
        + intervals["node_id"].astype(str)
        + "|"
        + intervals["species"].astype(str)
    )
    intervals["q"] = np.nan
    estimable = intervals["status"] == "ok"
    if estimable.any():
        adjusted = add_bh_qvalues(intervals.loc[estimable], within=())
        intervals.loc[estimable, "q"] = adjusted["q"]
    return DonorHonestContrastResult(
        intervals=intervals,
        selection_result=selection_result,
        training_donors=training_donors,
        inference_donors=inference_donors,
    )
