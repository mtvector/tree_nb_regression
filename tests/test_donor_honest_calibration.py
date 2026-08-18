"""End-to-end calibration gate for donor-honest tree-contrast inference."""
from __future__ import annotations

import pytest

from tree_nb_regression.calibration import run_donor_honest_calibration


@pytest.mark.slow
def test_donor_honest_calibration_gate() -> None:
    """Require nominal coverage, controlled null rejection, and localization."""
    summary = run_donor_honest_calibration(n_simulations=300, random_state=202)

    assert summary.n_signal_intervals == 300
    assert summary.n_null_intervals >= 225
    assert 0.90 <= summary.signal_coverage <= 0.975
    assert 0.90 <= summary.null_coverage <= 0.975
    assert summary.null_rejection_rate <= 0.06
    assert summary.localization_rate >= 0.70
    assert summary.selection_rate >= 0.85

