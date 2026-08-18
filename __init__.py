"""Tree-structured pseudobulk negative-binomial regression."""

from .burden_calibration import (
    BurdenCalibrationResult,
    BurdenTruth,
    TaxonomyScenario,
    make_known_burden_simulation,
    run_burden_bootstrap_calibration,
)
from .burden_inference import (
    BootstrapIntervalMethod,
    BurdenBootstrapConfig,
    BurdenBootstrapResult,
    BurdenFitConfig,
    bootstrap_evolutionary_burden,
)
from .calibration import CalibrationSummary, run_donor_honest_calibration
from .eb_calibration import (
    EffectRegime,
    EmpiricalBayesCalibrationResult,
    run_empirical_bayes_calibration,
)
from .honest_inference import (
    DonorHonestContrastResult,
    DonorSelectionConfig,
    donor_honest_intervals,
)
from .inference import add_bh_qvalues, compute_wald_significance
from .l1_tuning import L1TuningComparisonResult, run_l1_tuning_comparison
from .model import DEFAULT_GLOBAL_LAMBDA, fit_tree_nb
from .pseudobulk import build_pseudobulk
from .results import TreeNBResult
from .shrinkage import EmpiricalBayesConfig, ShrinkagePrior, evolutionary_burden
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
    "BurdenFitConfig",
    "BootstrapIntervalMethod",
    "BurdenBootstrapConfig",
    "BurdenBootstrapResult",
    "bootstrap_evolutionary_burden",
    "BurdenTruth",
    "TaxonomyScenario",
    "BurdenCalibrationResult",
    "make_known_burden_simulation",
    "run_burden_bootstrap_calibration",
    "EmpiricalBayesConfig",
    "ShrinkagePrior",
    "evolutionary_burden",
    "EffectRegime",
    "EmpiricalBayesCalibrationResult",
    "run_empirical_bayes_calibration",
    "L1TuningComparisonResult",
    "run_l1_tuning_comparison",
]
