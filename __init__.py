"""Tree-structured pseudobulk negative-binomial regression."""

from .calibration import CalibrationSummary, run_donor_honest_calibration
from .honest_inference import (
    DonorHonestContrastResult,
    DonorSelectionConfig,
    donor_honest_intervals,
)
from .inference import add_bh_qvalues, compute_wald_significance
from .model import DEFAULT_GLOBAL_LAMBDA, fit_tree_nb
from .pseudobulk import build_pseudobulk
from .results import TreeNBResult
from .species_tree import build_species_tree_design
from .taxonomy_tree import build_taxonomy_tree_from_obs

__all__ = [
    "build_taxonomy_tree_from_obs",
    "build_species_tree_design",
    "build_pseudobulk",
    "fit_tree_nb",
    "DEFAULT_GLOBAL_LAMBDA",
    "TreeNBResult",
    "compute_wald_significance",
    "add_bh_qvalues",
    "DonorHonestContrastResult",
    "DonorSelectionConfig",
    "donor_honest_intervals",
    "CalibrationSummary",
    "run_donor_honest_calibration",
]
