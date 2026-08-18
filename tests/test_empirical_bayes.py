"""Tests for level-specific empirical-Bayes shrinkage and calibration."""
from __future__ import annotations

import numpy as np
import pytest

from tree_nb_regression.eb_calibration import (
    EffectRegime,
    make_known_truth_simulation,
    run_empirical_bayes_calibration,
)
from tree_nb_regression.model import (
    _build_design_matrices,
    fit_tree_nb,
)
from tree_nb_regression.pseudobulk import build_pseudobulk
from tree_nb_regression.shrinkage import (
    EmpiricalBayesConfig,
    ShrinkagePrior,
    evolutionary_burden,
)
from tree_nb_regression.species_tree import build_species_tree_design
from tree_nb_regression.taxonomy_tree import build_taxonomy_tree_from_obs


def test_empirical_bayes_config_rejects_invalid_scales() -> None:
    """Reject scale bounds that cannot contain the initialization."""
    with pytest.raises(ValueError, match="min_scale"):
        EmpiricalBayesConfig(
            prior=ShrinkagePrior.GAUSSIAN,
            initial_scale=0.01,
            min_scale=0.02,
        )


def test_interaction_levels_are_orthogonal_and_norm_preserving() -> None:
    """Orthogonal interaction coding removes cross-level overlap."""
    simulation = make_known_truth_simulation(
        seed=4,
        regime=EffectRegime.SPARSE_LARGE,
        n_genes=6,
        n_donors_per_species=2,
        cells_per_pseudobulk=2,
    )
    obs = simulation.adata.obs
    pb = build_pseudobulk(
        obs,
        taxonomy_col="Group",
        species_col="species",
        batch_col="batch",
        donor_col="donor",
        min_cells_per_pseudobulk=2,
    )
    tree = build_taxonomy_tree_from_obs(obs, list(simulation.taxonomy_cols))
    species = build_species_tree_design(
        simulation.species_tree, sorted(obs["species"].astype(str).unique())
    )
    raw, _, _ = _build_design_matrices(
        pb,
        tree,
        species,
        list(simulation.taxonomy_cols),
        "species",
        "batch",
        "donor",
        orthogonal_tree=False,
    )
    orthogonal, _, _ = _build_design_matrices(
        pb,
        tree,
        species,
        list(simulation.taxonomy_cols),
        "species",
        "batch",
        "donor",
        orthogonal_tree=True,
    )
    families = ["species_tax_Class", "species_tax_Subclass", "species_tax_Group"]
    for earlier, later in zip(families[:-1], families[1:], strict=True):
        cross_product = orthogonal[earlier].T @ orthogonal[later]
        np.testing.assert_allclose(cross_product, 0.0, atol=1e-8)
    for family in families:
        np.testing.assert_allclose(
            np.linalg.norm(orthogonal[family], axis=0),
            np.linalg.norm(raw[family], axis=0),
            rtol=1e-8,
            atol=1e-8,
        )


@pytest.mark.parametrize("prior", list(ShrinkagePrior))
def test_empirical_bayes_fit_records_level_burdens(prior: ShrinkagePrior) -> None:
    """A fitted EB model exposes finite positive per-level burdens."""
    simulation = make_known_truth_simulation(
        seed=8,
        regime=EffectRegime.DENSE_SMALL,
        n_genes=6,
        n_donors_per_species=3,
        cells_per_pseudobulk=2,
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
        max_iter=40,
        progress=False,
        refit_support=False,
        empirical_bayes=EmpiricalBayesConfig(
            prior=prior,
            pilot_genes=6,
            pilot_max_iter=40,
            warmup_iterations=5,
            update_interval=5,
        ),
    )
    burden = evolutionary_burden(result)
    assert set(burden["level"]) == {"Class", "Subclass", "Group"}
    assert np.isfinite(burden["variance_burden"]).all()
    assert (burden["variance_burden"] > 0.0).all()
    assert result.orthogonal_tree


def test_empirical_bayes_calibration_emits_both_regimes_and_all_methods() -> None:
    """The benchmark covers every requested prior and effect regime."""
    calibration = run_empirical_bayes_calibration(
        n_simulations=1, random_state=11, max_iter=30
    )
    metrics = calibration.metrics
    assert set(metrics["regime"]) == {regime.value for regime in EffectRegime}
    assert set(metrics["method"]) == {
        "l1",
        "gaussian",
        "laplace",
        "spike_slab",
    }
    assert set(metrics["level"]) == {"Class", "Subclass", "Group"}
    assert np.isfinite(metrics["rmse"]).all()
    assert np.isfinite(metrics["burden_ratio"]).all()
