"""End-to-end calibration gate for level-specific empirical-Bayes shrinkage."""
from __future__ import annotations

import numpy as np
import pytest

from tree_nb_regression.eb_calibration import run_empirical_bayes_calibration


@pytest.mark.slow
def test_empirical_bayes_calibration_gate() -> None:
    """Require reconstruction, localization, and donor-honest calibration."""
    result = run_empirical_bayes_calibration(
        n_simulations=10,
        coverage_simulations=30,
        random_state=1701,
        max_iter=200,
    )
    metrics = result.metrics
    assert np.isfinite(metrics["burden_ratio"]).all()
    dense = metrics[metrics["regime"] == "dense_small"]
    assert dense.groupby("method")["localization_rate"].mean().min() >= 0.95
    assert (
        dense.groupby("method")["magnitude_ratio"].mean().between(0.85, 1.15).all()
    )
    sparse_spike = metrics[
        (metrics["regime"] == "sparse_large")
        & (metrics["method"] == "spike_slab")
    ]
    assert sparse_spike["localization_rate"].mean() >= 0.90
    assert sparse_spike["burden_ratio"].mean() >= 0.70
    coverage = result.donor_honest_metrics.set_index("method")
    assert (coverage["null_rejection_rate"] <= 0.10).all()
    assert coverage["null_coverage"].between(0.85, 1.0).all()
    assert coverage["signal_coverage"].between(0.85, 1.0).all()
    eligible = result.method_comparison[
        (result.method_comparison["method"] != "l1")
        & result.method_comparison["coverage_eligible"]
    ].sort_values("reconstruction_score")
    if eligible.empty:
        assert result.recommended_prior == "none"
    else:
        assert result.recommended_prior == eligible.iloc[0]["method"]
