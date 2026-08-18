"""Tests for leakage-free fixed-L1 tuning comparisons."""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from tree_nb_regression.honest_inference import DonorSelectionConfig
from tree_nb_regression.l1_tuning import run_l1_tuning_comparison
from tree_nb_regression.model import DEFAULT_GLOBAL_LAMBDA, fit_tree_nb


def test_normal_fit_defaults_are_calibrated_orthogonal_l1() -> None:
    """Keep the public defaults on the selected production configuration."""
    assert DEFAULT_GLOBAL_LAMBDA == 0.05
    assert inspect.signature(fit_tree_nb).parameters["orthogonal_tree"].default is True
    assert DonorSelectionConfig().global_lambda == 0.05


def test_l1_tuning_validates_inputs() -> None:
    """Reject empty simulations and invalid penalty grids."""
    with pytest.raises(ValueError, match="positive"):
        run_l1_tuning_comparison(n_simulations=0)
    with pytest.raises(ValueError, match="penalties"):
        run_l1_tuning_comparison(n_simulations=1, penalties=(0.0,))


@pytest.mark.slow
def test_l1_tuning_comparison_smoke() -> None:
    """Tune both bases without truth leakage and return finite metrics."""
    result = run_l1_tuning_comparison(
        n_simulations=1,
        penalties=(0.02, 0.09),
        random_state=2601,
        max_iter=80,
    )
    assert set(result.selected_lambdas) == {"legacy", "orthogonal"}
    assert set(result.selected_lambdas.values()) <= {0.02, 0.09}
    assert np.isfinite(result.tuning_curve["validation_nll"]).all()
    assert np.isfinite(result.metrics["normalized_effect_rmse"]).all()
    assert set(result.method_comparison["method"]) == {
        "legacy_l1_historical",
        "legacy_l1_tuned",
        "orthogonal_l1_fixed",
        "orthogonal_l1_tuned",
        "gaussian_eb",
        "laplace_eb",
        "spike_slab_eb",
    }
