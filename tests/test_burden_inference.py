"""Tests for donor-bootstrap evolutionary-burden inference."""

from __future__ import annotations

import numpy as np
import pytest

from tree_nb_regression.burden_inference import (
    BootstrapIntervalMethod,
    BurdenBootstrapConfig,
    BurdenFitConfig,
    bootstrap_evolutionary_burden,
)
from tree_nb_regression.eb_calibration import (
    EffectRegime,
    make_known_truth_simulation,
)
from tree_nb_regression.model import fit_tree_nb
from tree_nb_regression.shrinkage import evolutionary_burden


def test_default_bootstrap_uses_conservative_simultaneous_bca_bands() -> None:
    """Keep the calibrated donor-bootstrap interval contract explicit."""
    config = BurdenBootstrapConfig()
    assert config.interval_method is BootstrapIntervalMethod.SIMULTANEOUS_BCA
    assert config.log_interval_inflation == 1.2


def test_fixed_l1_burden_requires_orthogonal_basis() -> None:
    """Reject non-comparable legacy designs for level burden reporting."""
    simulation = make_known_truth_simulation(
        seed=41,
        regime=EffectRegime.DENSE_SMALL,
        n_genes=6,
        n_donors_per_species=2,
        cells_per_pseudobulk=2,
    )
    result = fit_tree_nb(
        simulation.adata,
        taxonomy_cols=list(simulation.taxonomy_cols),
        species_col="species",
        species_tree=simulation.species_tree,
        donor_col="donor",
        min_cells_per_pseudobulk=2,
        max_iter=30,
        progress=False,
        refit_support=False,
        orthogonal_tree=False,
    )
    with pytest.raises(ValueError, match="orthogonal_tree=True"):
        evolutionary_burden(result)


@pytest.mark.slow
def test_donor_bootstrap_refits_fixed_l1_and_reports_all_ratios() -> None:
    """Return finite donor-bootstrap burden intervals for every level pair."""
    simulation = make_known_truth_simulation(
        seed=42,
        regime=EffectRegime.SPARSE_LARGE,
        n_genes=6,
        n_donors_per_species=3,
        cells_per_pseudobulk=2,
    )
    result = bootstrap_evolutionary_burden(
        simulation.adata,
        taxonomy_cols=simulation.taxonomy_cols,
        species_col="species",
        species_tree=simulation.species_tree,
        donor_col="donor",
        fit_config=BurdenFitConfig(
            min_cells_per_pseudobulk=2,
            max_iter=50,
            refit_support=False,
        ),
        bootstrap_config=BurdenBootstrapConfig(n_bootstrap=3, random_state=42),
    )
    assert result.n_successful == 3
    assert set(result.point_burdens["prior"]) == {"fixed_l1"}
    assert len(result.ratios) == 3
    assert (result.ratios["status"] == "ok").all()
    assert np.isfinite(result.ratios[["ci_low", "ci_high"]].to_numpy()).all()
    assert (result.ratios["ci_low"] <= result.ratios["ci_high"]).all()
