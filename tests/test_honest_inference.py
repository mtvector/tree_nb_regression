"""Tests for donor-honest post-selection contrast inference."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tree_nb_regression.calibration import _make_calibration_adata
from tree_nb_regression.honest_inference import (
    DonorSelectionConfig,
    _welch_contrast,
    donor_honest_intervals,
)


def test_donor_honest_intervals_hold_out_complete_donors() -> None:
    """Tree selection and interval estimation use disjoint donor sets."""
    adata = _make_calibration_adata(seed=5, n_donors_per_species=8)
    result = donor_honest_intervals(
        adata,
        taxonomy_cols=("Class", "cluster"),
        species_col="species",
        species_tree="(Human,Macaque,Mouse);",
        donor_col="donor",
        batch_col="batch",
        selection=DonorSelectionConfig(global_lambda=0.01, max_iter=60),
        selection_threshold=1e-6,
        random_state=14,
    )

    observed_donors = set(adata.obs["donor"].astype(str))
    training = set(result.training_donors)
    inference = set(result.inference_donors)
    assert training.isdisjoint(inference)
    assert training | inference == observed_donors

    signal = result.intervals[
        (result.intervals["family"] == "species_tax_Class")
        & (result.intervals["node_id"] == "A")
        & (result.intervals["species"] == "Human")
        & (result.intervals["gene"] == "signal")
    ]
    assert len(signal) == 1
    row = signal.iloc[0]
    assert row["status"] == "ok"
    assert row["ci_lo"] < row["estimate"] < row["ci_hi"]
    assert row["estimate"] > 0.0
    assert row["n_target_donors"] == 4


def test_welch_contrast_has_nominal_null_calibration() -> None:
    """Independent donor-level null intervals attain the requested coverage."""
    rng = np.random.default_rng(819)
    covered: list[bool] = []
    rejected: list[bool] = []
    for _ in range(600):
        donor_values = pd.DataFrame(
            {
                "donor": [
                    *(f"Human_{index}" for index in range(10)),
                    *(f"Macaque_{index}" for index in range(10)),
                    *(f"Mouse_{index}" for index in range(10)),
                ],
                "species": ["Human"] * 10 + ["Macaque"] * 10 + ["Mouse"] * 10,
                "log_cpm": rng.normal(8.0, 0.4, size=30),
            }
        )
        contrast = _welch_contrast(
            donor_values=donor_values,
            donor_col="donor",
            species_col="species",
            target_species="Human",
            expected_species=("Human", "Macaque", "Mouse"),
            confidence_level=0.95,
        )
        assert contrast["status"] == "ok"
        covered.append(bool(contrast["ci_lo"] <= 0.0 <= contrast["ci_hi"]))
        rejected.append(bool(contrast["p"] < 0.05))

    coverage = float(np.mean(covered))
    null_rejection = float(np.mean(rejected))
    assert 0.90 <= coverage <= 0.975
    assert null_rejection <= 0.06

