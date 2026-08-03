"""Result container for tree NB regression."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TreeNBResult:
    coefficients: dict[str, np.ndarray]
    coef_metadata: pd.DataFrame
    selected_nonzero: dict[str, np.ndarray]
    group_meta: pd.DataFrame
    taxonomy_node_table: pd.DataFrame
    species_node_table: pd.DataFrame
    gene_names: list[str]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    gamma: np.ndarray | None = None  # per-group residual intercepts (n_groups, n_genes)
    intercept: np.ndarray | None = None  # per-gene intercept alpha (n_genes,)
    # Dispersion-tree fields (populated only when fit_dispersion_tree=True).
    # Each coefficient is in log_overdispersion space (= -log theta), so a
    # POSITIVE value means the node's subtree is MORE VARIABLE across replicates
    # than the gene baseline; NEGATIVE means more consistent.
    dispersion_coefficients: dict[str, np.ndarray] | None = None
    dispersion_selected_nonzero: dict[str, np.ndarray] | None = None
    dispersion_coef_metadata: pd.DataFrame | None = None
    dispersion_active_indices: dict[str, np.ndarray] | None = None
    # Artifacts needed for downstream Wald-style inference. None when not fit
    # or when the user opts out (keep_design_artifacts=False).
    library_sizes: np.ndarray | None = None
    log_theta_baseline: np.ndarray | None = None  # per-gene baseline log(theta)
    disp_offset: np.ndarray | None = None  # per-group additive offset on log_overdisp
    designs: dict[str, np.ndarray] | None = None  # full design matrices (unmasked)
    dispersion_designs: dict[str, np.ndarray] | None = None  # post-mask disp designs
    # species_tax_<level> per-column metadata (level, node_id, node_label,
    # species, n_species_at_node) — required by inference to label rows for
    # the sum-to-zero parameterization, where column index does NOT have the
    # old (sp_idx, node_idx) reference-coded layout.
    species_tax_meta: dict[str, pd.DataFrame] | None = None
    # For each species_tax_<level> family, a list of column-index groups (one
    # per node) over which Σ_s β[s,n,g] = 0 holds by construction. Used by
    # compute_wald_significance to augment the Hessian and by post-processing
    # to project coefficients onto the constraint subspace exactly.
    species_tax_node_groups: dict[str, list[list[int]]] | None = None

    @property
    def n_genes(self) -> int:
        return len(self.gene_names)

    def get_coefficients_df(self, family: str) -> pd.DataFrame:
        """Get coefficient matrix as DataFrame for a given family."""
        if family not in self.coefficients:
            raise KeyError(f"Family '{family}' not found. Available: {list(self.coefficients.keys())}")
        coefs = self.coefficients[family]
        meta = self.coef_metadata[self.coef_metadata["family"] == family]
        return pd.DataFrame(
            coefs,
            index=meta["coef_id"].values,
            columns=self.gene_names,
        )

    def get_dispersion_df(self, family: str) -> pd.DataFrame:
        """Get dispersion coefficient matrix for a family as a DataFrame.

        Rows are tree node ids (original positions in the tree, restricted to
        active/identifiable columns). Values are in log_overdispersion units
        (positive => more variable than gene baseline).
        """
        if self.dispersion_coefficients is None:
            raise RuntimeError(
                "No dispersion fit was performed. Re-run fit_tree_nb with "
                "fit_dispersion_tree=True."
            )
        if family not in self.dispersion_coefficients:
            raise KeyError(
                f"Dispersion family '{family}' not found. Available: "
                f"{list(self.dispersion_coefficients.keys())}"
            )
        coefs = self.dispersion_coefficients[family]
        meta = self.dispersion_coef_metadata[
            self.dispersion_coef_metadata["family"] == family
        ]
        index = (
            meta["node_id"].fillna(meta["coef_id"]).values
            if "node_id" in meta.columns
            else meta["coef_id"].values
        )
        return pd.DataFrame(coefs, index=index, columns=self.gene_names)

    def call_dispersion(
        self,
        family: str,
        threshold: float = 0.0,
    ) -> pd.DataFrame:
        """Boolean above/below threshold calls for dispersion coefficients.

        Parameters
        ----------
        family : str
            Dispersion design family.
        threshold : float, default 0.0
            log_overdispersion threshold. Coefficient > threshold => "above"
            (more variable than baseline by at least the threshold);
            coefficient < -threshold => "below".

        Returns
        -------
        DataFrame indexed by node_id with columns:
            'above'  : boolean (n_genes-wide aggregate: any gene above)
            'below'  : boolean (any gene below)
            'n_above': number of genes above threshold
            'n_below': number of genes below threshold
        """
        df = self.get_dispersion_df(family)
        n_above = (df > threshold).sum(axis=1)
        n_below = (df < -threshold).sum(axis=1)
        return pd.DataFrame({
            "above": n_above > 0,
            "below": n_below > 0,
            "n_above": n_above.astype(int),
            "n_below": n_below.astype(int),
        })

    def summary(self) -> pd.DataFrame:
        """Summary table of nonzero coefficients per family per gene."""
        rows = []
        for family, mask in self.selected_nonzero.items():
            n_nonzero = mask.sum(axis=0) if mask.ndim == 2 else mask.sum()
            rows.append({
                "family": family,
                "kind": "mean",
                "n_coefficients": mask.shape[0] if mask.ndim == 2 else len(mask),
                "mean_nonzero_per_gene": float(np.mean(n_nonzero)),
                "max_nonzero_per_gene": int(np.max(n_nonzero)),
            })
        if self.dispersion_selected_nonzero is not None:
            for family, mask in self.dispersion_selected_nonzero.items():
                n_nonzero = mask.sum(axis=0) if mask.ndim == 2 else mask.sum()
                rows.append({
                    "family": family,
                    "kind": "dispersion",
                    "n_coefficients": mask.shape[0] if mask.ndim == 2 else len(mask),
                    "mean_nonzero_per_gene": float(np.mean(n_nonzero)),
                    "max_nonzero_per_gene": int(np.max(n_nonzero)),
                })
        return pd.DataFrame(rows)

