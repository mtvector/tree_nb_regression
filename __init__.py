"""Tree-structured pseudobulk negative-binomial regression."""

from .taxonomy_tree import build_taxonomy_tree_from_obs
from .species_tree import build_species_tree_design
from .pseudobulk import build_pseudobulk
from .model import fit_tree_nb, DEFAULT_GLOBAL_LAMBDA
from .results import TreeNBResult
from .inference import compute_wald_significance, add_bh_qvalues

__all__ = [
    "build_taxonomy_tree_from_obs",
    "build_species_tree_design",
    "build_pseudobulk",
    "fit_tree_nb",
    "DEFAULT_GLOBAL_LAMBDA",
    "TreeNBResult",
    "compute_wald_significance",
    "add_bh_qvalues",
]
