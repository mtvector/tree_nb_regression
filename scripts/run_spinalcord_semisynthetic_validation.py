"""Run repeated donor-split spike-ins on real spinal-cord UMI counts."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import cast

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from tree_nb_regression import DonorSelectionConfig, donor_honest_intervals
from tree_nb_regression.honest_inference import _donor_log_cpm, _welch_contrast

DATA_PATH = Path(
    "/data/lipari_workshop_attempt4/SpC_workshop_snRNA_session1_processed.h5ad"
)
OUTPUT_DIR = Path("/results/tree_nb_spinalcord_semisynthetic_validation")
SPECIES = ("Human", "Macaque_nemestrina", "Mouse")
TAXONOMY = ("Class_V2", "Subclass_V2", "Group_V2")
TARGET_SPECIES = "Human"
TARGET_CLASS = "GABA"
TARGET_GENE = "ACTB"
LOG_FOLD_CHANGE = float(os.environ.get("TREE_NB_SPIKE_LOG_FC", "0.9"))
VALIDATION_LABEL = os.environ.get("TREE_NB_VALIDATION_LABEL", "moderate_spike")


def _load_subset() -> ad.AnnData:
    """Load a six-donor-per-species real-count subset with aligned taxonomy."""
    backed = ad.read_h5ad(DATA_PATH, backed="r")
    obs = cast(pd.DataFrame, backed.obs)
    valid = (
        obs["species"].isin(SPECIES)
        & obs[list(TAXONOMY)].notna().all(axis=1)
        & ~obs[list(TAXONOMY)].astype(str).eq("nan").any(axis=1)
    )
    donors: list[str] = []
    for species in SPECIES:
        counts = obs.loc[valid & (obs["species"] == species), "donor_name"].astype(str)
        donors.extend(counts.value_counts().head(6).index.tolist())
    rows = np.flatnonzero((valid & obs["donor_name"].astype(str).isin(donors)).to_numpy())
    genes = [backed.var_names.get_loc(gene) for gene in ("ACTB", "GAPDH", "TUBB")]
    genes.extend(np.flatnonzero(backed.var["highly_variable"].to_numpy())[:6].tolist())
    subset = backed[np.sort(rows), np.asarray(genes)].to_memory()
    backed.file.close()
    subset.X = sparse.csr_matrix(np.rint(subset.layers["counts"]).astype(np.int64))
    return subset


def _inject_effect(*, adata: ad.AnnData, seed: int) -> ad.AnnData:
    """Inject a known Human GABA ACTB effect while preserving count noise."""
    result = adata.copy()
    mask = (
        (result.obs["species"].astype(str) == TARGET_SPECIES)
        & (result.obs["Class_V2"].astype(str) == TARGET_CLASS)
    ).to_numpy()
    counts = cast(sparse.csr_matrix, result.X).tolil(copy=True)
    baseline = counts[mask, 0].toarray().ravel()
    counts[mask, 0] = np.random.default_rng(seed).poisson(
        baseline * np.exp(LOG_FOLD_CHANGE)
    )[:, None]
    result.X = counts.tocsr()
    return result


def _finite_cohort_truth(*, adata: ad.AnnData) -> float:
    """Return the spiked cohort's donor-level Human GABA contrast."""
    mask = (adata.obs["Class_V2"].astype(str) == TARGET_CLASS).to_numpy()
    values = _donor_log_cpm(
        adata=adata,
        cell_mask=mask,
        donor_col="donor_name",
        species_col="species",
        gene_index=0,
        counts_layer=None,
        pseudocount=0.5,
    )
    contrast = _welch_contrast(
        donor_values=values,
        donor_col="donor_name",
        species_col="species",
        target_species=TARGET_SPECIES,
        expected_species=SPECIES,
        confidence_level=0.95,
    )
    return float(contrast["estimate"])


def main() -> None:
    """Run ten independent donor splits and save machine-readable evidence."""
    output_dir = OUTPUT_DIR / VALIDATION_LABEL
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = _load_subset()
    injected = _inject_effect(adata=baseline, seed=4401)
    truth = _finite_cohort_truth(adata=injected)
    records: list[dict[str, float | int | bool]] = []
    for split_seed in range(10):
        result = donor_honest_intervals(
            injected,
            taxonomy_cols=TAXONOMY,
            species_col="species",
            species_tree="(Mouse,(Macaque_nemestrina,Human));",
            donor_col="donor_name",
            selection=DonorSelectionConfig(max_iter=180),
            random_state=100 + split_seed,
        )
        rows = result.intervals[
            (result.intervals["family"] == "species_tax_Class_V2")
            & (result.intervals["species"] == TARGET_SPECIES)
            & (result.intervals["gene"] == TARGET_GENE)
            & (result.intervals["node_id"] == TARGET_CLASS)
        ]
        if len(rows) != 1:
            records.append({"split_seed": split_seed, "selected": False})
            continue
        row = rows.iloc[0]
        records.append(
            {
                "split_seed": split_seed,
                "selected": True,
                "rank_one": bool(row["selection_contrast_rank"] == 1),
                "covered": bool(row["ci_lo"] <= truth <= row["ci_hi"]),
                "discoverable": bool(row["status"] == "ok"),
                "q_below_0_05": bool(row["q"] < 0.05),
            }
        )
    frame = pd.DataFrame(records)
    frame.to_csv(output_dir / "split_records.csv", index=False)
    summary = {
        "dataset": str(DATA_PATH),
        "n_cells": int(injected.n_obs),
        "n_donors": int(injected.obs["donor_name"].nunique()),
        "log_fold_change": LOG_FOLD_CHANGE,
        "finite_cohort_log_cpm_contrast": truth,
        "selection_rate": float(frame["selected"].mean()),
        "rank_one_rate": float(frame["rank_one"].fillna(False).mean()),
        "interval_coverage": float(frame["covered"].fillna(False).mean()),
        "discovery_rate": float(frame["q_below_0_05"].fillna(False).mean()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
