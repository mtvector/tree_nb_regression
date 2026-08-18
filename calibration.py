"""Simulation harness for donor-honest tree-contrast calibration."""

from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .honest_inference import DonorSelectionConfig, donor_honest_intervals


@dataclass(frozen=True)
class CalibrationSummary:
    """Empirical coverage, null rejection, and localization metrics."""

    n_simulations: int
    n_signal_intervals: int
    n_null_intervals: int
    signal_coverage: float
    null_coverage: float
    null_rejection_rate: float
    localization_rate: float
    selection_rate: float
    records: pd.DataFrame


def _population_log_cpm_mean(*, effect: float, random_effect_sd: float) -> float:
    """Approximate the population donor-level log CPM target by Monte Carlo."""
    rng = np.random.default_rng(92741)
    signal_random_effects = rng.normal(0.0, random_effect_sd, size=200_000)
    null_random_effects = rng.normal(0.0, random_effect_sd, size=200_000)
    signal = np.exp(np.log(20.0) + effect + signal_random_effects)
    null = np.exp(np.log(15.0) + null_random_effects)
    total = signal + null + 10.0
    return float(np.mean(np.log(signal / total * 1_000_000.0)))


def _make_calibration_adata(*, seed: int, n_donors_per_species: int) -> ad.AnnData:
    """Simulate donor-correlated counts with one localized species interaction."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, str]] = []
    counts: list[np.ndarray] = []
    species_order = ("Human", "Macaque", "Mouse")
    for species in species_order:
        for donor_index in range(n_donors_per_species):
            donor = f"{species}_d{donor_index}"
            donor_signal = rng.normal(0.0, 0.25)
            donor_null = rng.normal(0.0, 0.25)
            for cell_class in ("A", "B"):
                interaction = 0.9 if species == "Human" and cell_class == "A" else 0.0
                rates = np.exp(
                    np.array(
                        [
                            np.log(20.0) + interaction + donor_signal,
                            np.log(15.0) + donor_null,
                            np.log(10.0),
                        ]
                    )
                )
                for cluster_index in (1, 2):
                    for _ in range(15):
                        counts.append(rng.poisson(rates))
                        rows.append(
                            {
                                "Class": cell_class,
                                "cluster": f"{cell_class}{cluster_index}",
                                "species": species,
                                "donor": donor,
                                "batch": f"batch_{donor_index % 2}",
                            }
                        )
    adata = ad.AnnData(
        X=sparse.csr_matrix(np.asarray(counts, dtype=np.int64)), obs=pd.DataFrame(rows)
    )
    adata.var_names = ["signal", "null", "background"]
    return adata


def _target_row(*, intervals: pd.DataFrame, gene: str, node_id: str) -> pd.Series | None:
    """Return one selected Human/Class contrast for a simulation gene."""
    matches = intervals[
        (intervals["family"] == "species_tax_Class")
        & (intervals["node_id"] == node_id)
        & (intervals["species"] == "Human")
        & (intervals["gene"] == gene)
    ]
    if len(matches) != 1:
        return None
    return matches.iloc[0]


def run_donor_honest_calibration(
    *,
    n_simulations: int = 100,
    n_donors_per_species: int = 12,
    random_state: int = 0,
    selection: DonorSelectionConfig = DonorSelectionConfig(max_iter=120),
) -> CalibrationSummary:
    """Evaluate held-out interval calibration and tree localization end to end.

    The signal is localized to Human within Class A. The null gene has no
    species or class effect. The selected-tree model sees only training donors;
    every reported interval uses the complementary donors.
    """
    if n_simulations < 1:
        raise ValueError("n_simulations must be positive.")
    if n_donors_per_species < 4:
        raise ValueError("n_donors_per_species must be at least four.")
    signal_truth = _population_log_cpm_mean(effect=0.9, random_effect_sd=0.25)
    baseline_truth = _population_log_cpm_mean(effect=0.0, random_effect_sd=0.25)
    true_signal_contrast = signal_truth - baseline_truth
    records: list[dict[str, float | int | bool | str]] = []
    rng = np.random.default_rng(random_state)
    for simulation_index in range(n_simulations):
        adata = _make_calibration_adata(
            seed=int(rng.integers(0, np.iinfo(np.int32).max)),
            n_donors_per_species=n_donors_per_species,
        )
        result = donor_honest_intervals(
            adata,
            taxonomy_cols=("Class", "cluster"),
            species_col="species",
            species_tree="(Human,Macaque,Mouse);",
            donor_col="donor",
            batch_col="batch",
            selection=selection,
            random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
        )
        signal_row = _target_row(intervals=result.intervals, gene="signal", node_id="A")
        # Class B has no species effect on any gene, so it supplies a genuine
        # null contrast unaffected by Class-A composition changes.
        null_row = _target_row(intervals=result.intervals, gene="null", node_id="B")
        class_signal = result.intervals[
            (result.intervals["family"] == "species_tax_Class")
            & (result.intervals["species"] == "Human")
            & (result.intervals["gene"] == "signal")
        ]
        score_column = (
            "selection_contrast_score"
            if "selection_contrast_score" in class_signal
            else "selection_beta"
        )
        beta_a = class_signal.loc[class_signal["node_id"] == "A", score_column]
        beta_b = class_signal.loc[class_signal["node_id"] == "B", score_column]
        # Correct localization means that the true Class-A signal is retained
        # and is stronger than Class B if B was selected at all. Penalized
        # selection is allowed to omit the null Class-B coefficient.
        localized = bool(
            len(beta_a) == 1
            and (len(beta_b) == 0 or abs(float(beta_a.iloc[0])) > abs(float(beta_b.iloc[0])))
        )
        for label, truth, row in (
            ("signal", true_signal_contrast, signal_row),
            ("null_effect", 0.0, null_row),
        ):
            if row is None or row["status"] != "ok":
                records.append(
                    {
                        "simulation": simulation_index,
                        "kind": label,
                        "selected": False,
                        "estimable": False,
                        "localized": localized,
                    }
                )
                continue
            records.append(
                {
                    "simulation": simulation_index,
                    "kind": label,
                    "selected": True,
                    "estimable": True,
                    "localized": localized,
                    "truth": truth,
                    "estimate": float(row["estimate"]),
                    "ci_lo": float(row["ci_lo"]),
                    "ci_hi": float(row["ci_hi"]),
                    "p": float(row["p"]),
                    "covered": bool(float(row["ci_lo"]) <= truth <= float(row["ci_hi"])),
                }
            )
    frame = pd.DataFrame(records)
    signal = frame[(frame["kind"] == "signal") & frame["estimable"]]
    null = frame[(frame["kind"] == "null_effect") & frame["estimable"]]
    localization = frame[frame["kind"] == "signal"]
    return CalibrationSummary(
        n_simulations=n_simulations,
        n_signal_intervals=len(signal),
        n_null_intervals=len(null),
        signal_coverage=float(signal["covered"].mean()) if len(signal) else float("nan"),
        null_coverage=float(null["covered"].mean()) if len(null) else float("nan"),
        null_rejection_rate=float((null["p"] < 0.05).mean()) if len(null) else float("nan"),
        localization_rate=float(localization["localized"].mean())
        if len(localization)
        else float("nan"),
        selection_rate=float(frame["selected"].mean()) if len(frame) else float("nan"),
        records=frame,
    )
