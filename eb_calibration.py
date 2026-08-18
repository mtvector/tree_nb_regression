"""Known-truth calibration for level-specific empirical-Bayes shrinkage."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .calibration import run_donor_honest_calibration
from .honest_inference import DonorSelectionConfig
from .model import _build_design_matrices, fit_tree_nb
from .pseudobulk import build_pseudobulk
from .shrinkage import EmpiricalBayesConfig, ShrinkagePrior, evolutionary_burden
from .species_tree import build_species_tree_design
from .taxonomy_tree import build_taxonomy_tree_from_obs


class EffectRegime(str, Enum):
    """Synthetic evolutionary-effect distributions used for calibration."""

    DENSE_SMALL = "dense_small"
    SPARSE_LARGE = "sparse_large"


@dataclass(frozen=True)
class KnownTruthSimulation:
    """Synthetic counts and coefficients in the fitted orthogonal basis."""

    adata: ad.AnnData
    coefficients: dict[str, np.ndarray]
    designs: dict[str, np.ndarray]
    taxonomy_cols: tuple[str, ...]
    species_tree: str


@dataclass(frozen=True)
class EmpiricalBayesCalibrationResult:
    """Coefficient, localization, and level-burden calibration metrics."""

    metrics: pd.DataFrame
    donor_honest_metrics: pd.DataFrame
    method_comparison: pd.DataFrame
    recommended_prior: str
    n_simulations: int


def _balanced_taxonomy() -> tuple[tuple[str, str, str], ...]:
    """Return a balanced three-level taxonomy with eight terminal groups."""
    return tuple(
        (cell_class, f"{cell_class}{subclass}", f"{cell_class}{subclass}{group}")
        for cell_class in ("A", "B")
        for subclass in ("1", "2")
        for group in ("a", "b")
    )


def _plant_level_coefficients(
    *,
    designs: dict[str, np.ndarray],
    metadata: dict[str, pd.DataFrame],
    node_groups: dict[str, list[list[int]]],
    n_genes: int,
    regime: EffectRegime,
) -> dict[str, np.ndarray]:
    """Plant sum-to-zero effects with distinct burdens at every level."""
    magnitudes = {"Class": 0.25, "Subclass": 0.45, "Group": 0.75}
    planted: dict[str, np.ndarray] = {}
    for level, magnitude in magnitudes.items():
        family = f"species_tax_{level}"
        beta = np.zeros((designs[family].shape[1], n_genes), dtype=np.float64)
        meta = metadata[family]
        groups = node_groups[family]
        if regime is EffectRegime.DENSE_SMALL:
            active_groups = groups
            active_genes = range(n_genes)
            effect = magnitude
        else:
            active_groups = groups[:1]
            level_index = ("Class", "Subclass", "Group").index(level)
            active_genes = range(level_index * 2, min(level_index * 2 + 2, n_genes))
            effect = 2.0 * magnitude
        for group_index, columns in enumerate(active_groups):
            species = meta.set_index("col_index").loc[columns, "species"].astype(str)
            human_columns = [column for column in columns if species.loc[column] == "Human"]
            other_columns = [column for column in columns if species.loc[column] != "Human"]
            node_sign = 1.0 if group_index % 2 == 0 else -1.0
            for gene in active_genes:
                # Pair positive and negative gene effects so each species-node
                # perturbation changes composition without masquerading as a
                # library-size shift.
                sign = node_sign * (1.0 if gene % 2 == 0 else -1.0)
                if human_columns:
                    beta[human_columns[0], gene] = sign * effect
                if other_columns:
                    beta[other_columns, gene] = -sign * effect / len(other_columns)
        planted[family] = beta
    return planted


def _canonicalize_coefficients(
    *,
    coefficients: dict[str, np.ndarray],
    designs: dict[str, np.ndarray],
    node_groups: dict[str, list[list[int]]],
) -> dict[str, np.ndarray]:
    """Choose the minimum-norm sum-to-zero coordinates for each fitted effect."""
    canonical: dict[str, np.ndarray] = {}
    for family, beta in coefficients.items():
        X = designs[family]
        constraints = np.zeros(
            (len(node_groups[family]), X.shape[1]), dtype=np.float64
        )
        for row_index, columns in enumerate(node_groups[family]):
            constraints[row_index, columns] = 1.0
        augmented_design = np.vstack([X, 1_000.0 * constraints])
        augmented_effect = np.vstack(
            [X @ beta, np.zeros((constraints.shape[0], beta.shape[1]))]
        )
        canonical[family] = np.linalg.lstsq(
            augmented_design, augmented_effect, rcond=None
        )[0]
    return canonical


def make_known_truth_simulation(
    *,
    seed: int,
    regime: EffectRegime,
    n_genes: int = 12,
    n_donors_per_species: int = 6,
    cells_per_pseudobulk: int = 6,
) -> KnownTruthSimulation:
    """Simulate donor-correlated counts from the model's orthogonal basis."""
    if n_genes < 6:
        raise ValueError("n_genes must be at least six.")
    if n_donors_per_species < 2:
        raise ValueError("n_donors_per_species must be at least two.")
    if cells_per_pseudobulk < 2:
        raise ValueError("cells_per_pseudobulk must be at least two.")
    rng = np.random.default_rng(seed)
    taxonomy_cols = ("Class", "Subclass", "Group")
    species = ("Human", "Macaque", "Mouse")
    rows: list[dict[str, str]] = []
    for species_name in species:
        for donor_index in range(n_donors_per_species):
            for path in _balanced_taxonomy():
                for _ in range(cells_per_pseudobulk):
                    rows.append(
                        {
                            **dict(zip(taxonomy_cols, path, strict=True)),
                            "species": species_name,
                            "donor": f"{species_name}_d{donor_index}",
                            "batch": f"batch_{donor_index % 2}",
                        }
                    )
    obs = pd.DataFrame(rows)
    placeholder = ad.AnnData(
        X=sparse.csr_matrix((len(obs), n_genes), dtype=np.int64),
        obs=obs,
        var=pd.DataFrame(index=[f"gene_{index}" for index in range(n_genes)]),
    )
    pb = build_pseudobulk(
        obs,
        taxonomy_col="Group",
        species_col="species",
        batch_col="batch",
        donor_col="donor",
        min_cells_per_pseudobulk=cells_per_pseudobulk,
    )
    tree = build_taxonomy_tree_from_obs(obs, list(taxonomy_cols))
    species_tree = "(Human,Macaque,Mouse);"
    species_design = build_species_tree_design(species_tree, list(species))
    designs, metadata, node_groups = _build_design_matrices(
        pb,
        tree,
        species_design,
        list(taxonomy_cols),
        "species",
        "batch",
        "donor",
        orthogonal_tree=True,
    )
    coefficients = _plant_level_coefficients(
        designs=designs,
        metadata=metadata,
        node_groups=node_groups,
        n_genes=n_genes,
        regime=regime,
    )
    coefficients = _canonicalize_coefficients(
        coefficients=coefficients,
        designs=designs,
        node_groups=node_groups,
    )
    eta_group = np.zeros((pb.n_groups, n_genes), dtype=np.float64)
    for family, beta in coefficients.items():
        eta_group += designs[family] @ beta
    cell_to_group = pb.cell_to_group.tocoo()
    group_for_cell = np.full(len(obs), -1, dtype=np.int64)
    group_for_cell[cell_to_group.row] = cell_to_group.col
    donor_effects = {
        donor: rng.normal(0.0, 0.12, size=n_genes)
        for donor in obs["donor"].astype(str).unique()
    }
    counts = np.zeros((len(obs), n_genes), dtype=np.int64)
    for cell_index, row in obs.iterrows():
        group_index = group_for_cell[cell_index]
        eta = eta_group[group_index] + donor_effects[str(row["donor"])]
        unnormalized = np.exp(np.clip(eta, -5.0, 5.0))
        mean = 1_500.0 * unnormalized / unnormalized.sum()
        theta = 20.0
        rate = rng.gamma(shape=theta, scale=mean / theta)
        counts[cell_index] = rng.poisson(rate)
    placeholder.X = sparse.csr_matrix(counts)
    return KnownTruthSimulation(
        adata=placeholder,
        coefficients=coefficients,
        designs={
            family: design
            for family, design in designs.items()
            if family.startswith("species_tax_")
        },
        taxonomy_cols=taxonomy_cols,
        species_tree=species_tree,
    )


def _method_config(
    *, method: str, pilot_genes: int, max_iter: int
) -> EmpiricalBayesConfig | None:
    """Build a reproducible prior configuration for one benchmark method."""
    if method != "l1":
        prior = ShrinkagePrior(method)
        return EmpiricalBayesConfig(
            prior=prior,
            pilot_genes=pilot_genes,
            pilot_max_iter=max_iter,
            warmup_iterations=(
                max(10, 3 * max_iter // 4)
                if prior is ShrinkagePrior.SPIKE_SLAB
                else max(10, max_iter // 5)
            ),
            update_interval=10,
        )
    return None


def _fit_method(
    *, simulation: KnownTruthSimulation, method: str, max_iter: int
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Fit one shrinkage method and return coefficients and burdens."""
    config = _method_config(
        method=method,
        pilot_genes=simulation.adata.n_vars,
        max_iter=max_iter,
    )
    result = fit_tree_nb(
        simulation.adata,
        taxonomy_cols=list(simulation.taxonomy_cols),
        species_col="species",
        species_tree=simulation.species_tree,
        batch_col="batch",
        donor_col="donor",
        min_cells_per_pseudobulk=2,
        global_lambda=0.05,
        max_iter=max_iter,
        progress=False,
        refit_support=False,
        orthogonal_tree=True,
        empirical_bayes=config,
    )
    if method == "l1":
        if result.designs is None:
            raise RuntimeError("Calibration fit did not retain design artifacts.")
        burdens = {
            family: float(
                np.mean(np.square(result.designs[family] @ beta))
            )
            for family, beta in result.coefficients.items()
            if family.startswith("species_tax_")
        }
    else:
        burden_frame = evolutionary_burden(result)
        burdens = dict(
            zip(
                burden_frame["family"].astype(str),
                burden_frame["variance_burden"].astype(float),
                strict=True,
            )
        )
    return result.coefficients, burdens


def run_empirical_bayes_calibration(
    *,
    n_simulations: int = 10,
    random_state: int = 0,
    max_iter: int = 150,
    coverage_simulations: int = 0,
) -> EmpiricalBayesCalibrationResult:
    """Compare L1, Gaussian EB, and Laplace EB in two known-truth regimes."""
    if n_simulations < 1:
        raise ValueError("n_simulations must be positive.")
    if coverage_simulations < 0:
        raise ValueError("coverage_simulations cannot be negative.")
    rng = np.random.default_rng(random_state)
    records: list[dict[str, float | int | str]] = []
    for simulation_index in range(n_simulations):
        for regime in EffectRegime:
            simulation = make_known_truth_simulation(
                seed=int(rng.integers(0, np.iinfo(np.int32).max)),
                regime=regime,
            )
            for method in ("l1", "gaussian", "laplace", "spike_slab"):
                fitted, burdens = _fit_method(
                    simulation=simulation, method=method, max_iter=max_iter
                )
                for family, truth in simulation.coefficients.items():
                    estimate = fitted[family]
                    signal = truth != 0.0
                    true_burden = float(
                        np.mean(
                            np.square(simulation.designs[family] @ truth)
                        )
                    )
                    localized: list[bool] = []
                    for gene in np.flatnonzero(signal.any(axis=0)):
                        true_locations = set(np.flatnonzero(signal[:, gene]).tolist())
                        localized.append(
                            int(np.argmax(np.abs(estimate[:, gene]))) in true_locations
                        )
                    records.append(
                        {
                            "simulation": simulation_index,
                            "regime": regime.value,
                            "method": method,
                            "family": family,
                            "level": family.removeprefix("species_tax_"),
                            "rmse": float(np.sqrt(np.mean((estimate - truth) ** 2))),
                            "signal_rmse": float(
                                np.sqrt(np.mean((estimate[signal] - truth[signal]) ** 2))
                            ),
                            "magnitude_ratio": float(
                                np.mean(np.abs(estimate[signal]))
                                / np.mean(np.abs(truth[signal]))
                            ),
                            "localization_rate": float(np.mean(localized)),
                            "true_burden": true_burden,
                            "estimated_burden": burdens[family],
                            "burden_ratio": burdens[family] / true_burden,
                        }
                    )
    coverage_records: list[dict[str, float | int | str]] = []
    if coverage_simulations:
        for method in ("l1", "gaussian", "laplace", "spike_slab"):
            config = _method_config(
                method=method,
                pilot_genes=3,
                max_iter=max_iter,
            )
            coverage = run_donor_honest_calibration(
                n_simulations=coverage_simulations,
                n_donors_per_species=12,
                random_state=random_state + 10_000,
                selection=DonorSelectionConfig(
                    global_lambda=0.01,
                    max_iter=max_iter,
                    empirical_bayes=config,
                ),
            )
            coverage_records.append(
                {
                    "method": method,
                    "n_simulations": coverage_simulations,
                    "signal_coverage": coverage.signal_coverage,
                    "null_coverage": coverage.null_coverage,
                    "null_rejection_rate": coverage.null_rejection_rate,
                    "localization_rate": coverage.localization_rate,
                    "selection_rate": coverage.selection_rate,
                }
            )
    metric_frame = pd.DataFrame(records)
    coverage_frame = pd.DataFrame(coverage_records)
    comparison = (
        metric_frame.assign(
            magnitude_error=lambda frame: (frame["magnitude_ratio"] - 1.0).abs(),
            burden_error=lambda frame: np.abs(np.log(frame["burden_ratio"])),
            localization_error=lambda frame: 1.0 - frame["localization_rate"],
        )
        .groupby("method", as_index=False)[
            ["magnitude_error", "burden_error", "localization_error"]
        ]
        .mean()
    )
    comparison["reconstruction_score"] = comparison[
        ["magnitude_error", "burden_error", "localization_error"]
    ].sum(axis=1)
    if not coverage_frame.empty:
        comparison = comparison.merge(coverage_frame, on="method", how="left")
        comparison["coverage_eligible"] = (
            comparison["signal_coverage"].between(0.88, 0.99)
            & comparison["null_coverage"].between(0.88, 0.99)
            & (comparison["null_rejection_rate"] <= 0.10)
            & (comparison["localization_rate"] >= 0.60)
        )
    else:
        comparison["coverage_eligible"] = True
    eb_candidates = comparison[
        (comparison["method"] != "l1") & comparison["coverage_eligible"]
    ]
    if eb_candidates.empty:
        recommended_prior = "none"
    else:
        recommended_prior = str(
            eb_candidates.sort_values("reconstruction_score").iloc[0]["method"]
        )
    return EmpiricalBayesCalibrationResult(
        metrics=metric_frame,
        donor_honest_metrics=coverage_frame,
        method_comparison=comparison.sort_values("reconstruction_score").reset_index(
            drop=True
        ),
        recommended_prior=recommended_prior,
        n_simulations=n_simulations,
    )
