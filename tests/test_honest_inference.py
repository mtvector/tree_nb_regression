"""Tests for donor-honest post-selection contrast inference."""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from tree_nb_regression.calibration import _make_calibration_adata
from tree_nb_regression.honest_inference import (
    DonorSelectionConfig,
    _candidate_contrasts,
    _donor_log_cpm,
    _welch_contrast,
    donor_honest_intervals,
)


def test_vectorized_candidate_contrasts_match_scalar_reference() -> None:
    """Batched donor-node aggregation preserves every scalar contrast result."""
    adata = _make_calibration_adata(seed=17, n_donors_per_species=4)
    candidates = pd.DataFrame(
        [
            {"level": "Class", "node_id": "A", "gene_index": 0, "species": species}
            for species in ("Human", "Macaque", "Mouse")
        ]
        + [
            {
                "level": "cluster",
                "node_id": "B/B2",
                "gene_index": 1,
                "species": "Macaque",
            },
            {
                "level": "Class",
                "node_id": "missing",
                "gene_index": 2,
                "species": "Human",
            },
        ]
    )
    species = ("Human", "Macaque", "Mouse")
    cell_mask = np.ones(adata.n_obs, dtype=bool)
    vectorized = _candidate_contrasts(
        adata=adata,
        candidates=candidates,
        taxonomy_cols=("Class", "cluster"),
        donor_col="donor",
        species_col="species",
        counts_layer=None,
        library_size_col=None,
        cell_mask=cell_mask,
        expected_species=species,
        confidence_level=0.95,
        pseudocount=0.5,
    )

    obs = adata.obs.reset_index(drop=True)
    ancestors = {
        "Class": obs[["Class"]].astype(str).agg("/".join, axis=1),
        "cluster": obs[["Class", "cluster"]].astype(str).agg("/".join, axis=1),
    }
    scalar_rows: list[dict[str, float | int | str]] = []
    for candidate in candidates.itertuples(index=False):
        node_mask = (ancestors[candidate.level] == candidate.node_id).to_numpy()
        donor_values = _donor_log_cpm(
            adata=adata,
            cell_mask=node_mask & cell_mask,
            donor_col="donor",
            species_col="species",
            gene_index=int(candidate.gene_index),
            counts_layer=None,
            library_size_col=None,
            pseudocount=0.5,
        )
        scalar_rows.append(
            _welch_contrast(
                donor_values=donor_values,
                donor_col="donor",
                species_col="species",
                target_species=str(candidate.species),
                expected_species=species,
                confidence_level=0.95,
            )
        )
    scalar = pd.DataFrame(scalar_rows)

    assert vectorized["status"].tolist() == scalar["status"].tolist()
    numeric_columns = [
        "estimate",
        "se",
        "df",
        "ci_lo",
        "ci_hi",
        "statistic",
        "p",
        "n_target_donors",
        "n_other_donors",
    ]
    ok = vectorized["status"] == "ok"
    np.testing.assert_allclose(
        vectorized.loc[ok, numeric_columns].to_numpy(dtype=float),
        scalar.loc[ok, numeric_columns].to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )


def test_donor_log_cpm_uses_external_full_library_size() -> None:
    """Gene-panel inference can retain the full-transcriptome CPM denominator."""
    adata = ad.AnnData(
        X=sparse.csr_matrix([[5], [5]], dtype=np.int64),
        obs=pd.DataFrame(
            {
                "donor": ["Human_1", "Human_1"],
                "species": ["Human", "Human"],
                "full_library_size": [100.0, 900.0],
            }
        ),
    )
    values = _donor_log_cpm(
        adata=adata,
        cell_mask=np.ones(2, dtype=bool),
        donor_col="donor",
        species_col="species",
        gene_index=0,
        counts_layer=None,
        library_size_col="full_library_size",
        pseudocount=0.5,
    )
    expected = np.log((10.0 + 0.5) / (1000.0 + 0.5) * 1_000_000.0)
    assert np.isclose(values.loc[0, "log_cpm"], expected)


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
    assert row["selection_contrast_score"] > 0.0
    assert row["selection_contrast_rank"] >= 1


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
