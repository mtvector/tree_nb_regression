"""Test held-out false discoveries by randomising mouse donor pseudo-species."""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from tree_nb_regression import DonorSelectionConfig, donor_honest_intervals

DATA_PATH = Path(
    "/data/lipari_workshop_attempt4/SpC_workshop_snRNA_session1_processed.h5ad"
)
OUTPUT_DIR = Path("/results/tree_nb_spinalcord_pseudolabel_null_validation")
TAXONOMY = ("Class_V2", "Subclass_V2", "Group_V2")
PSEUDO_SPECIES = ("PseudoA", "PseudoB", "PseudoC")


def _load_mouse_subset() -> ad.AnnData:
    """Load real mouse UMI counts from the twelve largest eligible donors."""
    backed = ad.read_h5ad(DATA_PATH, backed="r")
    obs = cast(pd.DataFrame, backed.obs)
    valid = (
        (obs["species"] == "Mouse")
        & obs[list(TAXONOMY)].notna().all(axis=1)
        & ~obs[list(TAXONOMY)].astype(str).eq("nan").any(axis=1)
    )
    donors = (
        obs.loc[valid, "donor_name"].astype(str).value_counts().head(12).index.tolist()
    )
    rows = np.flatnonzero((valid & obs["donor_name"].astype(str).isin(donors)).to_numpy())
    genes = [backed.var_names.get_loc(gene) for gene in ("ACTB", "GAPDH", "TUBB")]
    genes.extend(np.flatnonzero(backed.var["highly_variable"].to_numpy())[:6].tolist())
    subset = backed[np.sort(rows), np.asarray(genes)].to_memory()
    backed.file.close()
    subset.X = sparse.csr_matrix(np.rint(subset.layers["counts"]).astype(np.int64))
    return subset


def _assign_pseudo_species(*, adata: ad.AnnData, seed: int) -> ad.AnnData:
    """Assign unrelated pseudo-species labels evenly and randomly by donor."""
    result = adata.copy()
    donors = np.sort(result.obs["donor_name"].astype(str).unique())
    assigned = np.repeat(np.asarray(PSEUDO_SPECIES), len(donors) // len(PSEUDO_SPECIES))
    assigned = np.random.default_rng(seed).permutation(assigned)
    donor_to_label = dict(zip(donors, assigned, strict=True))
    result.obs["pseudo_species"] = result.obs["donor_name"].astype(str).map(donor_to_label)
    return result


def main() -> None:
    """Run ten donor-label randomisations and record selected-test false positives."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = _load_mouse_subset()
    records: list[pd.DataFrame] = []
    for seed in range(10):
        pseudo = _assign_pseudo_species(adata=baseline, seed=900 + seed)
        result = donor_honest_intervals(
            pseudo,
            taxonomy_cols=TAXONOMY,
            species_col="pseudo_species",
            species_tree="(PseudoA,(PseudoB,PseudoC));",
            donor_col="donor_name",
            selection=DonorSelectionConfig(max_iter=180),
            random_state=1200 + seed,
        )
        frame = result.intervals.copy()
        frame["randomisation_seed"] = seed
        records.append(frame)
    all_intervals = pd.concat(records, ignore_index=True)
    all_intervals.to_csv(OUTPUT_DIR / "intervals.csv", index=False)
    tested = all_intervals[all_intervals["status"] == "ok"]
    summary = {
        "dataset": str(DATA_PATH),
        "design": "Mouse donors randomly assigned to three pseudo-species labels (four donors each)",
        "n_cells": int(baseline.n_obs),
        "n_donors": int(baseline.obs["donor_name"].nunique()),
        "n_randomisations": 10,
        "n_selected_intervals": int(len(all_intervals)),
        "n_tested_intervals": int(len(tested)),
        "false_discoveries_q_0_05": int((tested["q"] < 0.05).sum()),
        "false_discovery_proportion_q_0_05": float((tested["q"] < 0.05).mean())
        if len(tested)
        else 0.0,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
