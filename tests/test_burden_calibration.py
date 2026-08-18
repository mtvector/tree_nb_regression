"""Known-truth tests for donor-bootstrap burden-ratio calibration."""
from __future__ import annotations

import numpy as np
import pytest

from tree_nb_regression.burden_calibration import (
    BurdenTruth,
    TaxonomyScenario,
    make_known_burden_simulation,
    run_burden_bootstrap_calibration,
)
from tree_nb_regression.eb_calibration import EffectRegime


@pytest.mark.parametrize("scenario", list(TaxonomyScenario))
def test_known_burden_simulation_has_requested_ratios(
    scenario: TaxonomyScenario,
) -> None:
    """Plant equal and unequal burden targets in both topology variants."""
    equal = make_known_burden_simulation(
        seed=61,
        burden_truth=BurdenTruth.EQUAL,
        regime=EffectRegime.DENSE_SMALL,
        taxonomy_scenario=scenario,
    )
    unequal = make_known_burden_simulation(
        seed=62,
        burden_truth=BurdenTruth.UNEQUAL,
        regime=EffectRegime.SPARSE_LARGE,
        taxonomy_scenario=scenario,
    )
    assert np.allclose(list(equal.true_burdens.values()), 0.012)
    assert np.isclose(
        unequal.true_burdens["Group"] / unequal.true_burdens["Class"], 4.0
    )


@pytest.mark.slow
def test_burden_bootstrap_calibration_smoke_covers_all_scenarios() -> None:
    """Exercise equal/unequal, sparse/dense, and incomplete-tree calibration."""
    result = run_burden_bootstrap_calibration(
        n_simulations=1,
        n_bootstrap=3,
        random_state=3101,
        max_iter=50,
    )
    assert len(result.summary) == 8
    assert result.records["covered"].notna().all()
    assert set(result.summary["taxonomy_scenario"]) == {
        "balanced",
        "unbalanced_missing",
    }
