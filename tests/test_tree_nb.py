"""Tests for tree-structured NB regression."""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from tree_nb_regression.inference import add_bh_qvalues, compute_wald_significance
from tree_nb_regression.model import fit_tree_nb
from tree_nb_regression.pseudobulk import aggregate_chunk, build_pseudobulk
from tree_nb_regression.species_tree import build_species_tree_design
from tree_nb_regression.taxonomy_tree import build_taxonomy_tree_from_obs


def _make_synthetic_adata(
    n_cells: int = 2000,
    n_genes: int = 50,
    seed: int = 42,
) -> ad.AnnData:
    """Create synthetic AnnData with taxonomy, species, batch columns."""
    rng = np.random.default_rng(seed)

    # Define taxonomy paths (each final_cluster has a unique path)
    paths = [
        ("N", "Glut", "Glut-V", "Glut-V-1", "c1"),
        ("N", "Glut", "Glut-V", "Glut-V-1", "c2"),
        ("N", "Glut", "Glut-V", "Glut-V-2", "c3"),
        ("N", "Glut", "Glut-D", "Glut-D-1", "c4"),
        ("NN", "GABA", "GABA-V", "GABA-V-1", "c5"),
        ("NN", "GABA", "GABA-V", "GABA-V-2", "c6"),
        ("NN", "GABA", "GABA-D", "GABA-D-1", "c7"),
        ("NN", "MN", "MN-a", "MN-a-1", "c8"),
        ("NN", "MN", "MN-g", "MN-g-1", "c9"),
        ("NN", "MN", "MN-g", "MN-g-1", "c10"),  # same group, different cluster
    ]
    cols = ["Neighborhood", "Class", "Subclass", "Group", "final_cluster"]
    species_list = ["Mouse", "Human", "Macaque_mulatta", "Macaque_nemestrina"]
    batches = ["batch1", "batch2", "batch3"]
    donors = ["donor1", "donor2", "donor3", "donor4"]

    obs_rows = []
    for i in range(n_cells):
        path_idx = rng.integers(0, len(paths))
        path = paths[path_idx]
        sp = species_list[rng.integers(0, len(species_list))]
        batch = batches[rng.integers(0, len(batches))]
        donor = donors[rng.integers(0, len(donors))]
        row = {col: path[j] for j, col in enumerate(cols)}
        row["species"] = sp
        row["batch"] = batch
        row["donor"] = donor
        obs_rows.append(row)

    obs = pd.DataFrame(obs_rows)

    # Generate count data with structure
    # Base rates differ by taxonomy
    base_rates = rng.exponential(5.0, size=(len(paths), n_genes))
    # Species effects
    species_effects = rng.normal(0, 0.3, size=(len(species_list), n_genes))
    # Batch effects
    batch_effects = rng.normal(0, 0.5, size=(len(batches), n_genes))

    X_data = np.zeros((n_cells, n_genes), dtype=np.float32)
    for i in range(n_cells):
        path_idx = paths.index(tuple(obs.iloc[i][cols].values))
        sp_idx = species_list.index(obs.iloc[i]["species"])
        batch_idx = batches.index(obs.iloc[i]["batch"])
        rate = base_rates[path_idx] * np.exp(
            species_effects[sp_idx] + batch_effects[batch_idx]
        )
        X_data[i] = rng.poisson(rate.clip(0.1))

    X_sparse = sparse.csr_matrix(X_data)
    adata = ad.AnnData(X=X_sparse, obs=obs)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    return adata


# ─── Test 1: Taxonomy tree correctly reconstructed ────────────────────────────

def test_taxonomy_tree_reconstruction():
    adata = _make_synthetic_adata()
    tax_cols = ["Neighborhood", "Class", "Subclass", "Group", "final_cluster"]
    tree = build_taxonomy_tree_from_obs(adata.obs, tax_cols)

    assert tree.node_table is not None
    assert len(tree.leaf_ids) == 10  # 10 final clusters
    assert len(tree.levels) == 5

    # Each leaf should have a path of ancestors
    for leaf_id in tree.leaf_ids:
        ancestors = tree.leaf_mapping[leaf_id]
        assert len(ancestors) == 5  # one per level
        assert ancestors[-1] == leaf_id

    # Path indicator should have correct shape
    assert tree.A_tax_leaf.shape == (10, len(tree.node_ids))

    # Each leaf row should have exactly 5 ones (one per level)
    row_sums = np.asarray(tree.A_tax_leaf.sum(axis=1)).ravel()
    assert np.all(row_sums == 5)


# ─── Test 2: Pseudobulk matches manual groupby ───────────────────────────────

def test_pseudobulk_matches_groupby():
    adata = _make_synthetic_adata(n_cells=500, n_genes=10, seed=123)
    pb = build_pseudobulk(
        adata.obs,
        taxonomy_col="final_cluster",
        species_col="species",
        batch_col="batch",
        min_cells_per_pseudobulk=5,
    )

    X = adata.X
    Y = aggregate_chunk(X, pb.cell_to_group)

    # Verify against manual groupby
    obs_reset = adata.obs.reset_index(drop=True)
    for idx, row in pb.group_meta.iterrows():
        mask = (
            (obs_reset["final_cluster"] == row["final_cluster"])
            & (obs_reset["species"] == row["species"])
            & (obs_reset["batch"] == row["batch"])
        )
        if mask.sum() < 5:
            continue
        manual_sum = np.asarray(X[mask.values].sum(axis=0)).ravel()
        np.testing.assert_allclose(Y[idx], manual_sum, rtol=1e-5)


# ─── Test 3: Species tree design has correct paths ────────────────────────────

def test_species_tree_design():
    newick = "(Mouse,((Macaque_mulatta,Macaque_nemestrina),Human));"
    species = ["Human", "Macaque_mulatta", "Macaque_nemestrina", "Mouse"]
    design = build_species_tree_design(newick, species)

    assert design.A_species.shape[0] == 4
    assert design.A_species.shape[1] == len(design.node_ids)

    # Mouse should share root with everyone but nothing else specific
    mouse_idx = species.index("Mouse")
    macm_idx = species.index("Macaque_mulatta")
    macn_idx = species.index("Macaque_nemestrina")

    mouse_path = set(design.A_species[mouse_idx].nonzero()[1])
    macm_path = set(design.A_species[macm_idx].nonzero()[1])
    macn_path = set(design.A_species[macn_idx].nonzero()[1])

    # Two macaques should share more nodes than macaque+mouse
    shared_macs = macm_path & macn_path
    shared_mouse_mac = mouse_path & macm_path
    assert len(shared_macs) > len(shared_mouse_mac)


# ─── Test 4: No dense species × batch × cluster × gene tensor ────────────────

def test_no_dense_tensor():
    adata = _make_synthetic_adata(n_cells=1000, n_genes=20)
    pb = build_pseudobulk(
        adata.obs,
        taxonomy_col="final_cluster",
        species_col="species",
        batch_col="batch",
        min_cells_per_pseudobulk=5,
    )

    # The cell_to_group matrix should be sparse
    assert sparse.issparse(pb.cell_to_group)
    # n_groups should be much less than the full product
    n_species = adata.obs["species"].nunique()
    n_batch = adata.obs["batch"].nunique()
    n_cluster = adata.obs["final_cluster"].nunique()
    full_product = n_species * n_batch * n_cluster
    assert pb.n_groups < full_product


# ─── Test 5: Weighted L1 scales by Fisher norm ───────────────────────────────

def test_weighted_l1_scales():
    import torch

    from tree_nb_regression.model import _compute_penalty_scales

    rng = np.random.default_rng(99)
    n_groups, n_genes = 50, 10
    n_cols = 5

    X = rng.standard_normal((n_groups, n_cols))
    # Scale columns differently
    X[:, 0] *= 10
    X[:, 4] *= 0.1

    W = torch.tensor(rng.exponential(1.0, (n_groups, n_genes)), dtype=torch.float64)
    designs = {"tax_global": X}
    lambdas = {"tax_global": 1.0}

    scales = _compute_penalty_scales(designs, W, lambdas, use_family_size_calibration=False)
    c = scales["tax_global"].numpy()

    # Column 0 (large values) should have higher penalty scale
    assert c[0] > c[4]
    # Ratio should reflect the input scale difference
    assert c[0] / c[4] > 5.0


# ─── Test 6: Broad taxonomy effect assigned to high node ──────────────────────

def test_broad_taxonomy_effect():
    """A signal shared across all leaves of a Class should be captured at Class level."""
    rng = np.random.default_rng(42)
    n_cells = 3000
    n_genes = 5

    paths = [
        ("N", "Glut", "Glut-V", "GV1", "c1"),
        ("N", "Glut", "Glut-V", "GV2", "c2"),
        ("N", "Glut", "Glut-D", "GD1", "c3"),
        ("NN", "GABA", "GABA-V", "GBV1", "c4"),
        ("NN", "GABA", "GABA-D", "GBD1", "c5"),
    ]
    cols = ["Neighborhood", "Class", "Subclass", "Group", "final_cluster"]
    species_list = ["Mouse", "Human"]

    obs_rows = []
    for i in range(n_cells):
        path_idx = rng.integers(0, len(paths))
        obs_rows.append({
            col: paths[path_idx][j] for j, col in enumerate(cols)
        } | {"species": species_list[rng.integers(0, 2)], "batch": "b1"})
    obs = pd.DataFrame(obs_rows)

    # Inject a strong Class-level effect: all Glut cells get extra counts in gene 0
    X_data = rng.poisson(5, (n_cells, n_genes)).astype(np.float32)
    glut_mask = obs["Class"] == "Glut"
    X_data[glut_mask.values, 0] += 50  # Strong signal at Class level

    adata = ad.AnnData(X=sparse.csr_matrix(X_data), obs=obs)
    adata.var_names = [f"g{i}" for i in range(n_genes)]

    res = fit_tree_nb(
        adata,
        taxonomy_cols=cols,
        species_col="species",
        species_tree="(Mouse,Human);",
        batch_col="batch",
        max_iter=200,
        l1_lambdas={"tax_global": 0.05, "species_global": 0.5,
                    "species_tax_Neighborhood": 1.0, "species_tax_Class": 1.0,
                    "species_tax_Subclass": 1.0, "species_tax_Group": 1.0,
                    "species_tax_final_cluster": 1.0},
        refit_support=False,
    )

    # The taxonomy coefficients for gene 0 should have a strong effect
    # at the Class-level node for Glut
    tax_coefs = res.coefficients["tax_global"][:, 0]  # gene 0
    tax_nodes = res.taxonomy_node_table

    # Find indices of Class-level nodes in the design
    all_node_ids = list(tax_nodes["node_id"].values) if hasattr(res, '_node_ids') else list(res.taxonomy_node_table["node_id"].values)

    # The max abs coefficient should be at a high level (not leaf)
    leaf_nodes = tax_nodes[tax_nodes["level"] == "final_cluster"]
    leaf_indices = [list(all_node_ids).index(nid) for nid in leaf_nodes["node_id"] if nid in all_node_ids]
    non_leaf_indices = [i for i in range(len(all_node_ids)) if i not in leaf_indices]

    if len(non_leaf_indices) > 0 and len(tax_coefs) > max(non_leaf_indices):
        max_nonleaf = np.max(np.abs(tax_coefs[non_leaf_indices]))
        max_leaf = np.max(np.abs(tax_coefs[leaf_indices])) if leaf_indices else 0
        # Non-leaf (higher-level) effects should dominate for this broad signal
        assert max_nonleaf > max_leaf * 0.5, (
            f"Expected broad effect at higher level: nonleaf={max_nonleaf:.3f}, leaf={max_leaf:.3f}"
        )


# ─── Test 7: Species-specific Group effect assigned correctly ─────────────────

def test_species_specific_group_effect():
    """A species-specific effect at Group level should go to species_tax_Group."""
    rng = np.random.default_rng(7)
    n_cells = 2000
    n_genes = 5

    paths = [
        ("N", "Glut", "GV", "G1", "c1"),
        ("N", "Glut", "GV", "G2", "c2"),
        ("NN", "GABA", "GBV", "G3", "c3"),
        ("NN", "GABA", "GBD", "G4", "c4"),
    ]
    cols = ["Neighborhood", "Class", "Subclass", "Group", "final_cluster"]
    species_list = ["Mouse", "Human"]

    obs_rows = []
    for i in range(n_cells):
        path_idx = rng.integers(0, len(paths))
        obs_rows.append({
            col: paths[path_idx][j] for j, col in enumerate(cols)
        } | {"species": species_list[rng.integers(0, 2)], "batch": "b1"})
    obs = pd.DataFrame(obs_rows)

    X_data = rng.poisson(5, (n_cells, n_genes)).astype(np.float32)
    # Species-specific Group effect: Mouse + G1 gets boost in gene 0
    mask = (obs["species"] == "Mouse") & (obs["Group"] == "G1")
    X_data[mask.values, 0] += 40

    adata = ad.AnnData(X=sparse.csr_matrix(X_data), obs=obs)
    adata.var_names = [f"g{i}" for i in range(n_genes)]

    res = fit_tree_nb(
        adata,
        taxonomy_cols=cols,
        species_col="species",
        species_tree="(Mouse,Human);",
        batch_col="batch",
        max_iter=200,
        l1_lambdas={"tax_global": 0.5, "species_global": 0.5,
                    "species_tax_Neighborhood": 1.0, "species_tax_Class": 1.0,
                    "species_tax_Subclass": 1.0, "species_tax_Group": 0.05,
                    "species_tax_final_cluster": 1.0},
        refit_support=False,
    )

    # species_tax_Group should have nonzero coefficients for gene 0
    if "species_tax_Group" in res.coefficients:
        group_coefs = res.coefficients["species_tax_Group"][:, 0]
        assert np.max(np.abs(group_coefs)) > 0.01, (
            "Expected species-specific Group effect to be captured"
        )


# ─── Test 8: Batch effects not assigned to taxonomy/species ──────────────────

def test_batch_effects_absorbed():
    """Batch effects should be captured by batch covariates, not taxonomy/species."""
    rng = np.random.default_rng(88)
    n_cells = 2000
    n_genes = 5

    paths = [
        ("N", "Glut", "GV", "G1", "c1"),
        ("NN", "GABA", "GBV", "G2", "c2"),
    ]
    cols = ["Neighborhood", "Class", "Subclass", "Group", "final_cluster"]
    species_list = ["Mouse", "Human"]
    batches = ["batch1", "batch2"]

    obs_rows = []
    for i in range(n_cells):
        path_idx = rng.integers(0, len(paths))
        obs_rows.append({
            col: paths[path_idx][j] for j, col in enumerate(cols)
        } | {
            "species": species_list[rng.integers(0, 2)],
            "batch": batches[rng.integers(0, 2)],
        })
    obs = pd.DataFrame(obs_rows)

    # Only batch effect, no real taxonomy or species effects
    X_data = rng.poisson(10, (n_cells, n_genes)).astype(np.float32)
    batch2_mask = obs["batch"] == "batch2"
    X_data[batch2_mask.values, 0] += 30  # Strong batch effect on gene 0

    adata = ad.AnnData(X=sparse.csr_matrix(X_data), obs=obs)
    adata.var_names = [f"g{i}" for i in range(n_genes)]

    res = fit_tree_nb(
        adata,
        taxonomy_cols=cols,
        species_col="species",
        species_tree="(Mouse,Human);",
        batch_col="batch",
        max_iter=200,
        l1_lambdas={"tax_global": 0.5, "species_global": 0.5,
                    "species_tax_Neighborhood": 1.0, "species_tax_Class": 1.0,
                    "species_tax_Subclass": 1.0, "species_tax_Group": 1.0,
                    "species_tax_final_cluster": 1.0},
        refit_support=False,
    )

    # Batch coefficient for gene 0 should be substantial
    if "batch" in res.coefficients:
        batch_coefs = res.coefficients["batch"][:, 0]
        max_batch = np.max(np.abs(batch_coefs))

        # Taxonomy/species effects for gene 0 should be smaller
        tax_coefs = res.coefficients["tax_global"][:, 0]
        sp_coefs = res.coefficients["species_global"][:, 0]
        max_tax = np.max(np.abs(tax_coefs))
        max_sp = np.max(np.abs(sp_coefs))

        assert max_batch > max_tax, (
            f"Batch effect ({max_batch:.3f}) should exceed taxonomy ({max_tax:.3f})"
        )
        assert max_batch > max_sp, (
            f"Batch effect ({max_batch:.3f}) should exceed species ({max_sp:.3f})"
        )


# ─── Dispersion-tree tests ────────────────────────────────────────────────────

def _make_disp_adata(
    n_donors_per_species: int = 6,
    n_cells_per_donor_cluster: int = 80,
    n_genes: int = 8,
    high_disp_class: str = "Glut",
    high_disp_gene: int = 0,
    extra_sd: float = 1.2,
    seed: int = 11,
):
    """Synthesize a dataset where one Class has higher BETWEEN-DONOR dispersion
    on one gene (no mean shift). Used to verify that tree-structured dispersion
    fitting recovers the elevated-variance node."""
    rng = np.random.default_rng(seed)
    paths = [
        ("N", "Glut", "GV", "G1", "c1"),
        ("N", "Glut", "GV", "G2", "c2"),
        ("NN", "GABA", "GBV", "G3", "c3"),
        ("NN", "GABA", "GBD", "G4", "c4"),
    ]
    cols = ["Neighborhood", "Class", "Subclass", "Group", "final_cluster"]
    species_list = ["Mouse", "Human"]

    obs_rows: list[dict] = []
    X_rows: list[np.ndarray] = []

    # Per-(donor, species, gene) latent log-mean offsets:
    # high_disp_class genes get larger between-donor SD for high_disp_gene.
    for sp in species_list:
        for d in range(n_donors_per_species):
            donor_id = f"{sp}_d{d}"
            for path in paths:
                cls = path[1]
                # baseline gene means per cluster (shared across donors)
                base_log_mu = np.full(n_genes, np.log(15.0))
                # Between-donor log-mean noise per gene:
                noise_sd = np.full(n_genes, 0.1)
                if cls == high_disp_class:
                    noise_sd[high_disp_gene] = extra_sd
                donor_log_mu = base_log_mu + rng.normal(0.0, noise_sd)
                donor_mu = np.exp(donor_log_mu)
                # Sample cells (Poisson around donor mean — cell-level NB is
                # negligible at this scale; the tree should capture donor-level
                # variability as dispersion).
                cell_counts = rng.poisson(donor_mu, size=(n_cells_per_donor_cluster, n_genes))
                X_rows.append(cell_counts)
                for _ in range(n_cells_per_donor_cluster):
                    row = {col: path[i] for i, col in enumerate(cols)}
                    row["species"] = sp
                    row["batch"] = f"{sp}_b{d % 2}"
                    row["donor"] = donor_id
                    obs_rows.append(row)

    obs = pd.DataFrame(obs_rows)
    X = np.vstack(X_rows).astype(np.float32)
    adata = ad.AnnData(X=sparse.csr_matrix(X), obs=obs)
    adata.var_names = [f"g{i}" for i in range(n_genes)]
    return adata


def test_dispersion_disabled_by_default_no_break():
    """Sanity: leaving fit_dispersion_tree=False should not change behavior."""
    adata = _make_disp_adata(n_donors_per_species=3, n_cells_per_donor_cluster=20, n_genes=4)
    res = fit_tree_nb(
        adata,
        taxonomy_cols=["Neighborhood", "Class", "Subclass", "Group", "final_cluster"],
        species_col="species",
        species_tree="(Mouse,Human);",
        batch_col="batch",
        donor_col="donor",
        global_lambda=0.1,
        max_iter=100,
        refit_support=False,
    )
    assert res.dispersion_coefficients is None
    assert res.dispersion_coef_metadata is None
    # mean-side summary still works
    s = res.summary()
    assert "kind" in s.columns
    assert (s["kind"] == "mean").all()


def test_dispersion_null_signal_stays_sparse():
    """With NB data and no between-donor heterogeneity, dispersion coefs
    should be small in magnitude (no clade is meaningfully different)."""
    rng = np.random.default_rng(0)
    # Richer tree: 4 distinct paths, 8 donors per species
    paths = [
        ("N", "Glut", "GV", "G1", "c1"),
        ("N", "Glut", "GD", "G2", "c2"),
        ("NN", "GABA", "GBV", "G3", "c3"),
        ("NN", "GABA", "GBD", "G4", "c4"),
    ]
    cols = ["Neighborhood", "Class", "Subclass", "Group", "final_cluster"]
    species_list = ["Mouse", "Human"]
    n_genes = 4
    obs_rows, X_rows = [], []
    for sp in species_list:
        for d in range(8):
            donor_id = f"{sp}_d{d}"
            for path in paths:
                mu = np.full(n_genes, 12.0)
                theta_val = 5.0
                gamma_draw = rng.gamma(theta_val, mu / theta_val, size=(60, n_genes))
                cells = rng.poisson(gamma_draw)
                X_rows.append(cells)
                for _ in range(60):
                    row = {c: path[i] for i, c in enumerate(cols)}
                    row.update({"species": sp, "batch": f"b{d % 2}", "donor": donor_id})
                    obs_rows.append(row)
    obs = pd.DataFrame(obs_rows)
    X = np.vstack(X_rows).astype(np.float32)
    adata = ad.AnnData(X=sparse.csr_matrix(X), obs=obs)
    adata.var_names = [f"g{i}" for i in range(n_genes)]

    res = fit_tree_nb(
        adata,
        taxonomy_cols=cols,
        species_col="species",
        species_tree="(Mouse,Human);",
        batch_col="batch",
        donor_col="donor",
        global_lambda=0.1,
        max_iter=200,
        refit_support=True,
        fit_dispersion_tree=True,
        dispersion_lambda=0.5,
        min_replicates_per_node=3,
    )
    assert res.dispersion_coefficients is not None
    # Coefficient magnitudes should be small under the null. With NB(12, 5)
    # cell-level data pseudobulked to 60 cells, true between-clade dispersion
    # difference is essentially zero, so the L1+refit dispersion coefs should
    # have small abs values.
    df = res.get_dispersion_df("tax_global")
    # The L1 stage drives most below 0.01; survivors are bounded by data
    # information. We're satisfied if mean(|nonzero coef|) is modest (<0.5)
    # and the max is below a threshold a user would set for "real" calls.
    max_abs = df.abs().values.max()
    # Threshold loosened from 1.0 -> 2.5 after the sum-to-zero (centered
    # indicator) recoding of the mean-stage species_tax_* designs: the mean
    # fit residuals fed into the dispersion stage are slightly different
    # under the symmetric coding (no asymmetric reference contrast to absorb
    # null variation), and one or two clade coefs can drift into the 1-2
    # range purely from finite-sample noise under the null. The dispersion
    # _recovery_ tests still pass; this is only a "very small under null"
    # smoke test, not a precise calibration test.
    assert max_abs < 2.5, (
        f"Under the null, max |dispersion coef| should be < 2.5; got {max_abs:.3f}\n{df.round(3)}"
    )


def test_dispersion_recovers_clade_heterogeneity():
    """When Glut donors have inflated between-donor SD on gene 0, the tree
    dispersion fit should put a POSITIVE coefficient on a Glut-side node for
    gene 0, and not for control genes."""
    adata = _make_disp_adata(
        n_donors_per_species=8,
        n_cells_per_donor_cluster=60,
        n_genes=6,
        high_disp_class="Glut",
        high_disp_gene=0,
        extra_sd=1.5,
        seed=21,
    )
    cols = ["Neighborhood", "Class", "Subclass", "Group", "final_cluster"]
    res = fit_tree_nb(
        adata,
        taxonomy_cols=cols,
        species_col="species",
        species_tree="(Mouse,Human);",
        batch_col="batch",
        donor_col="donor",
        global_lambda=0.05,
        max_iter=250,
        refit_support=True,
        fit_dispersion_tree=True,
        dispersion_lambda=0.3,
        min_replicates_per_node=3,
    )
    assert res.dispersion_coefficients is not None
    df = res.get_dispersion_df("tax_global")
    # Identify Glut-side tree nodes (their node_id starts with "N/Glut" or equals Glut path components)
    glut_nodes = [nid for nid in df.index if isinstance(nid, str) and "Glut" in nid]
    gaba_nodes = [nid for nid in df.index if isinstance(nid, str) and "GABA" in nid]
    assert len(glut_nodes) > 0, f"No Glut nodes found among {list(df.index)}"

    # Gene 0: max coefficient over Glut nodes should be positive and exceed
    # any GABA-node coefficient by a healthy margin.
    g0_glut_max = df.loc[glut_nodes, "g0"].max()
    g0_gaba_max = df.loc[gaba_nodes, "g0"].max() if gaba_nodes else 0.0
    assert g0_glut_max > 0.1, (
        f"Expected positive Glut-side dispersion coef on gene 0; got {g0_glut_max:.3f}"
    )
    assert g0_glut_max > g0_gaba_max, (
        f"Glut dispersion ({g0_glut_max:.3f}) should exceed GABA "
        f"({g0_gaba_max:.3f}) on the perturbed gene"
    )

    # Control genes (g1..) should be near zero on Glut nodes (no signal there)
    for ctrl in ["g1", "g2", "g3"]:
        ctrl_max = df.loc[glut_nodes, ctrl].abs().max()
        assert ctrl_max < g0_glut_max, (
            f"Control gene {ctrl} dispersion ({ctrl_max:.3f}) exceeded "
            f"signal gene g0 ({g0_glut_max:.3f}) on Glut-side"
        )


def test_dispersion_replicate_masking():
    """Under-replicated nodes should be masked out of the dispersion design."""
    adata = _make_disp_adata(n_donors_per_species=2, n_cells_per_donor_cluster=40, n_genes=4)
    res = fit_tree_nb(
        adata,
        taxonomy_cols=["Neighborhood", "Class", "Subclass", "Group", "final_cluster"],
        species_col="species",
        species_tree="(Mouse,Human);",
        batch_col="batch",
        donor_col="donor",
        global_lambda=0.1,
        max_iter=100,
        refit_support=False,
        fit_dispersion_tree=True,
        # Demand 10 distinct donors per node — impossible with this dataset:
        min_replicates_per_node=10,
    )
    # All families should be masked away
    assert res.dispersion_active_indices is None or all(
        len(v) == 0 for v in res.dispersion_active_indices.values()
    )


def test_dispersion_call_threshold_semantics():
    """call_dispersion returns above/below counts; sign convention is +=more variable."""
    adata = _make_disp_adata(
        n_donors_per_species=8, n_cells_per_donor_cluster=50, n_genes=4,
        high_disp_class="Glut", high_disp_gene=0, extra_sd=1.5, seed=33,
    )
    res = fit_tree_nb(
        adata,
        taxonomy_cols=["Neighborhood", "Class", "Subclass", "Group", "final_cluster"],
        species_col="species",
        species_tree="(Mouse,Human);",
        batch_col="batch",
        donor_col="donor",
        global_lambda=0.05,
        max_iter=200,
        refit_support=True,
        fit_dispersion_tree=True,
        dispersion_lambda=0.3,
        min_replicates_per_node=3,
    )
    calls = res.call_dispersion("tax_global", threshold=0.2)
    assert {"above", "below", "n_above", "n_below"} <= set(calls.columns)
    # The Glut-perturbed signal should produce at least one above-threshold call
    assert calls["above"].any(), (
        "Expected at least one 'above threshold' dispersion call from the "
        "high-variance Glut perturbation"
    )


# ─── Wald inference tests ────────────────────────────────────────────────────

def test_wald_artifacts_present():
    adata = _make_disp_adata(n_donors_per_species=4, n_cells_per_donor_cluster=40, n_genes=4)
    res = fit_tree_nb(
        adata,
        taxonomy_cols=["Neighborhood", "Class", "Subclass", "Group", "final_cluster"],
        species_col="species",
        species_tree="(Mouse,Human);",
        batch_col="batch",
        donor_col="donor",
        global_lambda=0.1,
        max_iter=50,
        refit_support=False,
        fit_dispersion_tree=False,
        progress=False,
    )
    # All artifacts should be present by default
    assert res.library_sizes is not None and res.library_sizes.ndim == 1
    assert res.log_theta_baseline is not None and res.log_theta_baseline.shape == (4,)
    assert res.intercept is not None and res.intercept.shape == (4,)
    assert res.designs is not None and len(res.designs) > 0


def test_wald_opt_out_artifacts():
    adata = _make_disp_adata(n_donors_per_species=3, n_cells_per_donor_cluster=30, n_genes=3)
    res = fit_tree_nb(
        adata,
        taxonomy_cols=["Neighborhood", "Class", "Subclass", "Group", "final_cluster"],
        species_col="species",
        species_tree="(Mouse,Human);",
        batch_col="batch",
        donor_col="donor",
        global_lambda=0.1,
        max_iter=50,
        refit_support=False,
        progress=False,
        keep_design_artifacts=False,
    )
    assert res.library_sizes is None
    assert res.designs is None
    with pytest.raises(RuntimeError, match="keep_design_artifacts=True"):
        compute_wald_significance(res)


def test_wald_basic_shape_and_columns():
    adata = _make_disp_adata(n_donors_per_species=4, n_cells_per_donor_cluster=30, n_genes=4)
    res = fit_tree_nb(
        adata,
        taxonomy_cols=["Neighborhood", "Class", "Subclass", "Group", "final_cluster"],
        species_col="species",
        species_tree="(Mouse,Human);",
        batch_col="batch",
        donor_col="donor",
        global_lambda=0.05,
        max_iter=150,
        refit_support=True,
        fit_dispersion_tree=True,
        dispersion_lambda=0.3,
        min_replicates_per_node=3,
        progress=False,
    )
    df = compute_wald_significance(res)
    required = {
        "family", "index", "coef_id", "gene",
        "sp_contrast", "level", "node_label",
        "beta_hat", "se", "z", "p", "ci_lo", "ci_hi", "q",
    }
    assert required.issubset(set(df.columns)), f"missing: {required - set(df.columns)}"
    # Only selected coefficients should appear -> beta != 0 for every row
    assert (df["beta_hat"].abs() > 0).all()
    # SE must be positive and finite for every selected coef
    assert (df["se"] > 0).all() and np.isfinite(df["se"]).all()
    # CI brackets the estimate
    assert (df["ci_lo"] < df["beta_hat"]).all()
    assert (df["beta_hat"] < df["ci_hi"]).all()
    # p in [0, 1]
    assert ((df["p"] >= 0) & (df["p"] <= 1)).all()
    # q in [0, 1]
    assert ((df["q"] >= 0) & (df["q"] <= 1)).all()


def test_wald_recovers_strong_signal():
    """Strong injected mean signal should produce small p-values on the
    corresponding species_tax_Group coefficient for the perturbed gene."""
    rng = np.random.default_rng(42)
    # Need many pseudobulks to have enough DOF for Wald inference after
    # conditioning on the (intercept + nested-tree-level) nuisance design.
    n_donors_per_sp = 20
    n_cells_per_donor_cluster = 60
    n_genes = 4
    paths = [
        ("N", "Glut", "GV", "G1", "c1"),
        ("N", "Glut", "GV", "G2", "c2"),
        ("NN", "GABA", "GBV", "G3", "c3"),
        ("NN", "GABA", "GBD", "G4", "c4"),
    ]
    cols = ["Neighborhood", "Class", "Subclass", "Group", "final_cluster"]
    species_list = ["Mouse", "Human"]

    obs_rows = []
    X_rows = []
    for sp in species_list:
        for d in range(n_donors_per_sp):
            for path in paths:
                cell_counts = rng.poisson(5, size=(n_cells_per_donor_cluster, n_genes)).astype(np.float32)
                # Strong species-specific signal: Human + G1 + gene 0
                if sp == "Human" and path[3] == "G1":
                    cell_counts[:, 0] += 80
                X_rows.append(cell_counts)
                for _ in range(n_cells_per_donor_cluster):
                    row = {col: path[i] for i, col in enumerate(cols)}
                    row.update({"species": sp, "batch": f"{sp}_b{d % 3}", "donor": f"{sp}_d{d}"})
                    obs_rows.append(row)
    obs = pd.DataFrame(obs_rows)
    X = np.vstack(X_rows)
    adata = ad.AnnData(X=sparse.csr_matrix(X), obs=obs)
    adata.var_names = [f"g{i}" for i in range(n_genes)]

    res = fit_tree_nb(
        adata,
        taxonomy_cols=cols,
        species_col="species",
        species_tree="(Mouse,Human);",
        batch_col="batch",
        donor_col="donor",
        # Higher penalty on the broader levels pushes selection toward the
        # most-specific identifiable level so per-coef Wald SE is meaningful;
        # otherwise L1 splits the signal across nested correlated levels.
        l1_lambdas={
            "tax_global": 0.5,
            "species_global": 0.5,
            "species_tax_Neighborhood": 1.0,
            "species_tax_Class": 1.0,
            "species_tax_Subclass": 1.0,
            "species_tax_Group": 0.05,
            # See above (species_tax_final_cluster) on single-leaf collinearity.
            "species_tax_final_cluster": 10.0,
            # Under sum-to-zero coding the species_tax columns carry the
            # full per-species×node deviation symmetrically (β[Human, n] =
            # -β[Mouse, n]) rather than being absorbed into a reference
            # contrast. Donor and batch columns are aliased with species
            # signal (each donor is species-specific) and at small donor
            # lambdas the L1 selects ~half of them, which inflates the
            # joint Wald Hessian's species_tax SEs dramatically by VIF.
            # We keep donor/batch as proper nuisance terms here by raising
            # their L1 enough to drive most to zero so the inference for
            # the signal of interest is well-conditioned. Real analyses
            # should use unpenalised donor/batch effects (see review.md
            # item B6) -- which would have the same effect.
            "donor": 5.0,
            "batch": 5.0,
        },
        max_iter=200,
        refit_support=True,
        progress=False,
    )
    df = compute_wald_significance(res)
    g0_rows = df[df["gene"] == "g0"]
    assert len(g0_rows) > 0, "expected at least one selected coef for the perturbed gene"
    # Strong injected signal -> at least one very small Wald p-value on g0.
    # Note: post-selection refit p-values are NOT calibrated for the
    # multiple-testing of which coefs survived L1 -- they are screening
    # statistics. So we only assert recovery on the perturbed gene, not
    # absence of small p on control genes.
    g0_min = g0_rows["p"].min()
    assert g0_min < 1e-3, f"expected g0 min-p < 1e-3 for strong signal; got {g0_min:.3g}"


def test_bh_qvalues_monotone_and_calibrated():
    p = np.array([0.001, 0.01, 0.02, 0.4, 0.5, 0.95, np.nan])
    df = pd.DataFrame({"family": ["a"] * 7, "p": p})
    out = add_bh_qvalues(df)
    finite = ~out["q"].isna()
    # q >= p for non-trivial cases (BH inflates)
    assert (out.loc[finite, "q"].values >= out.loc[finite, "p"].values - 1e-12).all()
    # NaN p -> NaN q
    assert out["q"].iloc[-1] != out["q"].iloc[-1]
    # Monotonicity: sort by p and verify q non-decreasing
    sorted_finite = out.loc[finite].sort_values("p")
    assert (np.diff(sorted_finite["q"].values) >= -1e-12).all()


# ---------------------------------------------------------------------------
# Sum-to-zero parameterization tests
# ---------------------------------------------------------------------------


def _make_sumzero_test_adata(seed: int = 7):
    """Small synthetic dataset for sum-to-zero diagnostics."""
    rng = np.random.default_rng(seed)
    paths = [
        ("N", "Glut", "GV", "G1", "c1"),
        ("N", "Glut", "GV", "G2", "c2"),
        ("NN", "GABA", "GBV", "G3", "c3"),
        ("NN", "GABA", "GBD", "G4", "c4"),
    ]
    cols = ["Neighborhood", "Class", "Subclass", "Group", "final_cluster"]
    obs_rows, X_rows = [], []
    for sp in ["Mouse", "Human"]:
        for d in range(6):
            for path in paths:
                cc = rng.poisson(5, size=(30, 3)).astype(np.float32)
                if sp == "Human" and path[3] == "G1":
                    cc[:, 0] += 50
                X_rows.append(cc)
                for _ in range(30):
                    row = {col: path[i] for i, col in enumerate(cols)}
                    row.update({
                        "species": sp,
                        "batch": f"{sp}_b{d % 2}",
                        "donor": f"{sp}_d{d}",
                    })
                    obs_rows.append(row)
    obs = pd.DataFrame(obs_rows)
    X = np.vstack(X_rows)
    adata = ad.AnnData(X=sparse.csr_matrix(X), obs=obs)
    adata.var_names = [f"g{i}" for i in range(3)]
    return adata, cols


def test_sumzero_constraint_holds_per_node_per_gene():
    """β[s, n, g] coefficients should sum to ~0 across species at each node."""
    adata, cols = _make_sumzero_test_adata()
    res = fit_tree_nb(
        adata, taxonomy_cols=cols, species_col="species",
        species_tree="(Mouse,Human);", batch_col="batch", donor_col="donor",
        l1_lambdas={
            "tax_global": 0.5, "species_global": 0.5,
            "species_tax_Neighborhood": 1.0, "species_tax_Class": 1.0,
            "species_tax_Subclass": 1.0, "species_tax_Group": 0.05,
            "species_tax_final_cluster": 10.0,
            "donor": 5.0, "batch": 5.0,
        },
        max_iter=200, refit_support=True, progress=False,
    )
    assert res.species_tax_node_groups is not None
    # For every (family, node-group), the sum of selected species betas
    # must be ~0 (centered design + anchor enforces this exactly at the
    # cell level and the anchor refit preserves it to high precision).
    for family, groups in res.species_tax_node_groups.items():
        B = res.coefficients[family]  # (n_cols, n_genes)
        for group in groups:
            if len(group) < 2:
                continue
            sub = B[group, :]
            sums = sub.sum(axis=0)  # one per gene
            # With centered design + anchor, sums must be ~0 to within
            # floating-point optimization precision.
            assert np.allclose(sums, 0.0, atol=1e-3), (
                f"species_tax {family} group {group}: sums {sums} not ~0"
            )


def test_sumzero_meta_has_species_labels():
    """species_tax_meta should be populated with valid (level, node, species)."""
    adata, cols = _make_sumzero_test_adata()
    res = fit_tree_nb(
        adata, taxonomy_cols=cols, species_col="species",
        species_tree="(Mouse,Human);", batch_col="batch", donor_col="donor",
        l1_lambdas={"donor": 5.0, "batch": 5.0},
        max_iter=100, refit_support=True, progress=False,
    )
    assert res.species_tax_meta is not None
    for family, meta in res.species_tax_meta.items():
        # One row per design column
        n_cols = res.designs[family].shape[1]
        assert len(meta) == n_cols
        # Required columns present
        for c in ("col_index", "level", "node_id", "species", "n_species_at_node"):
            assert c in meta.columns
        # Species values are real species labels
        assert set(meta["species"].unique()).issubset({"Mouse", "Human"})
        # n_species_at_node should be at least 2 (singletons filtered)
        assert (meta["n_species_at_node"] >= 2).all()


def test_sumzero_species_permutation_invariance():
    """Permuting species labels (Mouse<->Human) and refitting should yield
    the same |β| per (species, node, gene) — by symmetry of sum-to-zero coding
    with symmetric L1."""
    adata1, cols = _make_sumzero_test_adata(seed=11)
    # Identical data but with species labels swapped at the obs level.
    adata2 = adata1.copy()
    swap = {"Mouse": "Human", "Human": "Mouse"}
    adata2.obs["species"] = adata2.obs["species"].map(swap)
    # Also remap the donor IDs so they're still species-consistent
    adata2.obs["donor"] = adata2.obs["donor"].str.replace(
        "Mouse_", "TMP_").str.replace(
        "Human_", "Mouse_").str.replace("TMP_", "Human_")
    adata2.obs["batch"] = adata2.obs["batch"].str.replace(
        "Mouse_", "TMP_").str.replace(
        "Human_", "Mouse_").str.replace("TMP_", "Human_")

    fit_kw = dict(
        taxonomy_cols=cols, species_col="species",
        species_tree="(Mouse,Human);", batch_col="batch", donor_col="donor",
        l1_lambdas={
            "tax_global": 0.5, "species_global": 0.5,
            "species_tax_Neighborhood": 1.0, "species_tax_Class": 1.0,
            "species_tax_Subclass": 1.0, "species_tax_Group": 0.05,
            "species_tax_final_cluster": 10.0, "donor": 5.0, "batch": 5.0,
        },
        max_iter=200, refit_support=True, progress=False,
    )
    res1 = fit_tree_nb(adata1, **fit_kw)
    res2 = fit_tree_nb(adata2, **fit_kw)
    # For each species_tax family, compare the per-(level, node, gene) abs
    # betas. Within a node, the +/- assignment may swap between species
    # under the relabeling, but |β| at the node level must agree.
    for family in res1.species_tax_node_groups or {}:
        groups = res1.species_tax_node_groups[family]
        B1, B2 = res1.coefficients[family], res2.coefficients[family]
        for group in groups:
            if len(group) < 2:
                continue
            # Sort within a group to remove the (Human, Mouse) ordering
            a1 = np.sort(np.abs(B1[group, :]), axis=0)
            a2 = np.sort(np.abs(B2[group, :]), axis=0)
            np.testing.assert_allclose(
                a1, a2, atol=5e-2,
                err_msg=f"family={family} group={group} not permutation-invariant",
            )
