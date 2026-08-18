"""Known-truth calibration for donor-bootstrap evolutionary-burden ratios."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from enum import Enum

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .burden_inference import (
    BootstrapIntervalMethod,
    BurdenBootstrapConfig,
    BurdenFitConfig,
    bootstrap_evolutionary_burden,
)
from .eb_calibration import EffectRegime, _canonicalize_coefficients
from .model import _build_design_matrices
from .pseudobulk import build_pseudobulk
from .species_tree import build_species_tree_design
from .taxonomy_tree import build_taxonomy_tree_from_obs


class BurdenTruth(str, Enum):
    """Whether level burdens are equal or follow known unequal ratios."""

    EQUAL = "equal"
    UNEQUAL = "unequal"


class TaxonomyScenario(str, Enum):
    """Balanced or incomplete taxonomy/species support for calibration."""

    BALANCED = "balanced"
    UNBALANCED_MISSING = "unbalanced_missing"


@dataclass(frozen=True)
class KnownBurdenSimulation:
    """Synthetic counts with fixed level burdens in the orthogonal basis."""

    adata: ad.AnnData
    taxonomy_cols: tuple[str, ...]
    species_tree: str
    true_burdens: dict[str, float]


@dataclass(frozen=True)
class BurdenCalibrationResult:
    """Per-ratio coverage records and aggregate confirmation gates."""

    records: pd.DataFrame
    summary: pd.DataFrame
    n_simulations: int
    n_bootstrap: int

    @property
    def confirms_calibration(self) -> bool:
        """Whether all empirical coverage and false-claim gates pass."""
        return bool(self.summary["passes_gate"].all())


type CalibrationRecord = dict[str, float | int | bool | str]
type CalibrationTask = tuple[
    BurdenTruth,
    EffectRegime,
    TaxonomyScenario,
    int,
    int,
    int,
    int,
    int,
    BootstrapIntervalMethod,
]


def _paths(scenario: TaxonomyScenario) -> tuple[tuple[str, str, str], ...]:
    """Return terminal paths for one balanced or unbalanced taxonomy."""
    if scenario is TaxonomyScenario.BALANCED:
        return tuple(
            (cell_class, f"{cell_class}{subclass}", f"{cell_class}{subclass}{group}")
            for cell_class in ("A", "B")
            for subclass in ("1", "2")
            for group in ("a", "b")
        )
    return (
        ("A", "A1", "A1a"),
        ("A", "A1", "A1b"),
        ("A", "A2", "A2a"),
        ("B", "B1", "B1a"),
        ("B", "B2", "B2a"),
        ("B", "B2", "B2b"),
    )


def _plant_coefficients(
    *,
    designs: dict[str, np.ndarray],
    metadata: dict[str, pd.DataFrame],
    node_groups: dict[str, list[list[int]]],
    regime: EffectRegime,
    target_burdens: dict[str, float],
    n_genes: int,
) -> dict[str, np.ndarray]:
    """Plant signed sum-to-zero effects and scale each family to target burden."""
    coefficients: dict[str, np.ndarray] = {}
    levels = ("Class", "Subclass", "Group")
    for level_index, level in enumerate(levels):
        family = f"species_tax_{level}"
        beta = np.zeros((designs[family].shape[1], n_genes), dtype=np.float64)
        groups = node_groups[family]
        if regime is EffectRegime.DENSE_SMALL:
            active_groups = groups
            active_genes = tuple(range(n_genes))
        else:
            active_groups = groups[:1]
            active_genes = tuple(range(2 * level_index, 2 * level_index + 2))
        meta = metadata[family].set_index("col_index")
        for group_index, columns in enumerate(active_groups):
            species = meta.loc[columns, "species"].astype(str)
            human = [column for column in columns if species.loc[column] == "Human"]
            other = [column for column in columns if species.loc[column] != "Human"]
            if not human or not other:
                raise RuntimeError("Simulation requires Human and another species at each node.")
            for gene in active_genes:
                sign = (1.0 if gene % 2 == 0 else -1.0) * (
                    1.0 if group_index % 2 == 0 else -1.0
                )
                beta[human[0], gene] = sign
                beta[other, gene] = -sign / len(other)
        coefficients[family] = beta
    canonical = _canonicalize_coefficients(
        coefficients=coefficients, designs=designs, node_groups=node_groups
    )
    for family, beta in canonical.items():
        current = float(np.mean(np.square(designs[family] @ beta)))
        canonical[family] = beta * np.sqrt(target_burdens[family] / current)
    return canonical


def make_known_burden_simulation(
    *,
    seed: int,
    burden_truth: BurdenTruth,
    regime: EffectRegime,
    taxonomy_scenario: TaxonomyScenario,
    n_genes: int = 9,
    n_donors_per_species: int = 12,
    cells_per_pseudobulk: int = 10,
) -> KnownBurdenSimulation:
    """Simulate donor-correlated counts with exact known level burdens."""
    if n_genes < 6:
        raise ValueError("n_genes must be at least six.")
    if n_donors_per_species < 2:
        raise ValueError("n_donors_per_species must be at least two.")
    rng = np.random.default_rng(seed)
    taxonomy_cols = ("Class", "Subclass", "Group")
    species = ("Human", "Macaque", "Mouse")
    rows: list[dict[str, str]] = []
    for species_name in species:
        for donor_index in range(n_donors_per_species):
            for path in _paths(taxonomy_scenario):
                if (
                    taxonomy_scenario is TaxonomyScenario.UNBALANCED_MISSING
                    and species_name == "Mouse"
                    and path[0] == "B"
                ):
                    continue
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
    adata = ad.AnnData(
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
    target_burdens = (
        {
            "species_tax_Class": 0.012,
            "species_tax_Subclass": 0.012,
            "species_tax_Group": 0.012,
        }
        if burden_truth is BurdenTruth.EQUAL
        else {
            "species_tax_Class": 0.006,
            "species_tax_Subclass": 0.012,
            "species_tax_Group": 0.024,
        }
    )
    coefficients = _plant_coefficients(
        designs=designs,
        metadata=metadata,
        node_groups=node_groups,
        regime=regime,
        target_burdens=target_burdens,
        n_genes=n_genes,
    )
    eta_group = np.zeros((pb.n_groups, n_genes), dtype=np.float64)
    for family, beta in coefficients.items():
        eta_group += designs[family] @ beta
    group_mapping = pb.cell_to_group.tocoo()
    group_for_cell = np.full(len(obs), -1, dtype=np.int64)
    group_for_cell[group_mapping.row] = group_mapping.col
    donor_effects = {
        donor: rng.normal(0.0, 0.15, size=n_genes)
        for donor in obs["donor"].astype(str).unique()
    }
    counts = np.zeros((len(obs), n_genes), dtype=np.int64)
    for cell_index, row in obs.iterrows():
        eta = eta_group[group_for_cell[cell_index]] + donor_effects[str(row["donor"])]
        unnormalized = np.exp(np.clip(eta, -5.0, 5.0))
        mean = 1_500.0 * unnormalized / unnormalized.sum()
        rate = rng.gamma(shape=20.0, scale=mean / 20.0)
        counts[cell_index] = rng.poisson(rate)
    adata.X = sparse.csr_matrix(counts)
    true_burdens = {
        family.removeprefix("species_tax_"): float(
            np.mean(np.square(designs[family] @ beta))
        )
        for family, beta in coefficients.items()
    }
    return KnownBurdenSimulation(
        adata=adata,
        taxonomy_cols=taxonomy_cols,
        species_tree=species_tree,
        true_burdens=true_burdens,
    )


def _calibration_records_for_scenario(task: CalibrationTask) -> list[CalibrationRecord]:
    """Run all replicates for one truth, effect, and topology scenario."""
    (
        burden_truth,
        regime,
        taxonomy_scenario,
        n_simulations,
        n_bootstrap,
        max_iter,
        n_donors_per_species,
        random_state,
        interval_method,
    ) = task
    rng = np.random.default_rng(random_state)
    records: list[CalibrationRecord] = []
    for simulation_index in range(n_simulations):
        simulation = make_known_burden_simulation(
            seed=int(rng.integers(0, np.iinfo(np.int32).max)),
            burden_truth=burden_truth,
            regime=regime,
            taxonomy_scenario=taxonomy_scenario,
            n_donors_per_species=n_donors_per_species,
        )
        result = bootstrap_evolutionary_burden(
            simulation.adata,
            taxonomy_cols=simulation.taxonomy_cols,
            species_col="species",
            species_tree=simulation.species_tree,
            donor_col="donor",
            batch_col="batch",
            fit_config=BurdenFitConfig(
                min_cells_per_pseudobulk=10,
                max_iter=max_iter,
                refit_support=True,
            ),
            bootstrap_config=BurdenBootstrapConfig(
                n_bootstrap=n_bootstrap,
                random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
                interval_method=interval_method,
            ),
        )
        for ratio in result.ratios.itertuples(index=False):
            numerator = str(ratio.numerator_level)
            denominator = str(ratio.denominator_level)
            true_ratio = (
                simulation.true_burdens[numerator]
                / simulation.true_burdens[denominator]
            )
            covered = bool(ratio.ci_low <= true_ratio <= ratio.ci_high)
            null_ratio = bool(np.isclose(true_ratio, 1.0, rtol=1e-10))
            claims_difference = bool(ratio.ci_low > 1.0 or ratio.ci_high < 1.0)
            records.append(
                {
                    "simulation": simulation_index,
                    "burden_truth": burden_truth.value,
                    "regime": regime.value,
                    "taxonomy_scenario": taxonomy_scenario.value,
                    "numerator_level": numerator,
                    "denominator_level": denominator,
                    "true_ratio": true_ratio,
                    "estimated_ratio": float(ratio.burden_ratio),
                    "ci_low": float(ratio.ci_low),
                    "ci_high": float(ratio.ci_high),
                    "bootstrap_probability_numerator_greater": float(
                        ratio.probability_numerator_greater
                    ),
                    "covered": covered,
                    "null_ratio": null_ratio,
                    "claims_difference": claims_difference,
                    "n_successful_bootstraps": result.n_successful,
                }
            )
    return records


def run_burden_bootstrap_calibration(
    *,
    n_simulations: int = 10,
    n_bootstrap: int = 40,
    random_state: int = 3101,
    max_iter: int = 120,
    n_donors_per_species: int = 12,
    n_jobs: int = 1,
    interval_method: BootstrapIntervalMethod = BootstrapIntervalMethod.SIMULTANEOUS_PERCENTILE,
) -> BurdenCalibrationResult:
    """Calibrate burden-ratio intervals across truth, sparsity, and topology."""
    if n_simulations < 1:
        raise ValueError("n_simulations must be positive.")
    if n_donors_per_species < 4:
        raise ValueError("n_donors_per_species must be at least four.")
    if n_jobs < 1:
        raise ValueError("n_jobs must be positive.")
    task_rng = np.random.default_rng(random_state)
    tasks: list[CalibrationTask] = []
    for burden_truth in BurdenTruth:
        for regime in EffectRegime:
            for taxonomy_scenario in TaxonomyScenario:
                tasks.append(
                    (
                        burden_truth,
                        regime,
                        taxonomy_scenario,
                        n_simulations,
                        n_bootstrap,
                        max_iter,
                        n_donors_per_species,
                        int(task_rng.integers(0, np.iinfo(np.int32).max)),
                        interval_method,
                    )
                )
    if n_jobs == 1:
        nested_records = [_calibration_records_for_scenario(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            nested_records = list(executor.map(_calibration_records_for_scenario, tasks))
    records = [record for scenario_records in nested_records for record in scenario_records]
    frame = pd.DataFrame(records)
    group_columns = ["burden_truth", "regime", "taxonomy_scenario"]
    summary = frame.groupby(group_columns, as_index=False).agg(
        interval_coverage=("covered", "mean"),
        mean_bootstrap_success=("n_successful_bootstraps", "mean"),
        n_ratio_records=("covered", "size"),
    )
    null_summary = frame[frame["null_ratio"]].groupby(group_columns, as_index=False).agg(
        false_claim_rate=("claims_difference", "mean")
    )
    summary = summary.merge(null_summary, on=group_columns, how="left")
    # With 60 ratios per setting, 60/60 coverage has a 95% binomial lower
    # bound below 0.95. It is therefore compatible with nominal coverage and
    # must not be rejected as "too high" on this finite simulation alone.
    summary["passes_gate"] = (
        (summary["interval_coverage"] >= 0.85)
        & (
            summary["false_claim_rate"].isna()
            | (summary["false_claim_rate"] <= 0.10)
        )
    )
    return BurdenCalibrationResult(
        records=frame,
        summary=summary,
        n_simulations=n_simulations,
        n_bootstrap=n_bootstrap,
    )
