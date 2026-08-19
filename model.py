"""Core NB regression model with tree-structured penalties."""
from __future__ import annotations

import math
import warnings
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from .pseudobulk import PseudobulkData, aggregate_chunk, build_pseudobulk
from .results import TreeNBResult
from .shrinkage import EmpiricalBayesConfig, ShrinkagePrior
from .species_tree import SpeciesTreeDesign, build_species_tree_design
from .taxonomy_tree import TaxonomyTree, build_taxonomy_tree_from_obs

DEFAULT_GLOBAL_LAMBDA = 0.05

# Dispersion side is noisier than the mean side (information per coefficient
# comes from replicate variance, not from cell-count). Default penalty is
# stronger to compensate. See _fit_gene_chunk for the calibration rationale.
DEFAULT_DISPERSION_LAMBDA = 0.2

# Strength of the sum-to-zero anchor for species_tax_<level> interaction
# families. Each node n with K = |S(n)| >= 2 observed species contributes a
# quadratic anchor 0.5 * SUMZERO_ANCHOR_MU * (sum_{s in S(n)} beta[s,n,g])^2
# per gene g, so the K species effects at that node are forced to sum to ~0
# during fitting. This makes the raw K-one-hot parameterization identifiable
# (rank K per node would otherwise be K-1) while keeping the L1 penalty fully
# symmetric across species. The anchor must also be applied during
# _refit_support and during Wald Hessian construction.
SUMZERO_ANCHOR_MU = 100.0

# Default set of design families that get tree-structured dispersion fitting.
# We intentionally omit species_tax_<level> interactions: they explode the
# parameter count and are usually under-replicated for dispersion identification.
DEFAULT_DISPERSION_FAMILIES = ("tax_global", "species_global")

# Legacy per-family defaults (deprecated, kept for backward compatibility)
DEFAULT_LAMBDAS = {
    "tax_global": 0.1,
    "species_global": 0.1,
    "species_tax_Neighborhood": 0.2,
    "species_tax_Class": 0.3,
    "species_tax_Subclass": 0.5,
    "species_tax_Group": 0.8,
    "species_tax_final_cluster": 1.0,
    "dispersion_tax": 1.5,
    "dispersion_species": 1.5,
    "batch": 0.05,
    "donor": 0.05,
}


def _build_design_matrices(
    pb: PseudobulkData,
    tax_tree: TaxonomyTree,
    sp_design: SpeciesTreeDesign,
    taxonomy_cols: list[str],
    species_col: str,
    batch_col: str | None,
    donor_col: str | None,
    orthogonal_tree: bool = False,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, pd.DataFrame],
    dict[str, list[list[int]]],
]:
    """Build all design sub-matrices for the pseudobulk groups.

    Returns
    -------
    designs : dict[family -> (n_groups, n_cols) ndarray]
        The numerical design matrices. For ``species_tax_<level>`` families
        these are **raw K-one-hot per-(species, node) indicators** restricted
        to species observed at each node (``S(n)``, |S(n)| >= 2). The
        sum-to-zero across species is enforced softly via an augmented
        quadratic anchor inside :func:`_fit_gene_chunk` and
        :func:`_refit_support` (see ``SUMZERO_ANCHOR_MU``).
    species_tax_meta : dict[family -> DataFrame]
        Per-column metadata for ``species_tax_*`` families. Columns:
        ``col_index, level, node_id, node_label, species, n_species_at_node``.
        Empty (no entry) for non-interaction families.
    species_tax_node_groups : dict[family -> list[list[int]]]
        For each ``species_tax_*`` family, a list of column-index groups —
        one inner list per (level, node) — used by the sum-to-zero anchor
        (penalize ``(sum over inner-list columns of beta)^2`` per node).
    """
    gm = pb.group_meta
    n = pb.n_groups

    leaf_col = taxonomy_cols[-1]
    leaf_to_idx = {lid: i for i, lid in enumerate(tax_tree.leaf_ids)}
    sp_to_idx = {sp: i for i, sp in enumerate(sp_design.species_order)}

    # Taxonomy path indicator for each group
    tax_indices = []
    for _, row in gm.iterrows():
        leaf_path = "/".join(
            str(row.get(col, row.get(leaf_col, "")))
            for col in taxonomy_cols
            if col in row.index
        )
        if leaf_path not in leaf_to_idx:
            full_leaf_label = str(row[leaf_col])
            found = False
            for lid in tax_tree.leaf_ids:
                if lid.endswith("/" + full_leaf_label) or lid == full_leaf_label:
                    leaf_path = lid
                    found = True
                    break
            if not found:
                leaf_path = tax_tree.leaf_ids[0]
                warnings.warn(f"Could not map group to taxonomy leaf: {row.to_dict()}")
        tax_indices.append(leaf_to_idx.get(leaf_path, 0))

    A_tax_full = tax_tree.A_tax_leaf[tax_indices].toarray()

    if orthogonal_tree:
        # ── Helmert / nested-deviation re-parameterization of tax_global ──
        # Replace each non-root path-indicator column with its residual
        # against its parent's path-indicator column over the GROUP axis:
        #   col_n_new[i] = col_n[i] - (|S(n)|/|S(p)|) * col_p[i]
        # where S(·) = set of groups in the subtree of that node.
        # This makes col_n orthogonal to col_p (and recursively to every
        # ancestor), removing the catastrophic rank deficiency between
        # tree levels that drives anti-conservative Wald SE at the
        # Class/Subclass nodes. Siblings are still partially aliased
        # (sum to zero over each parent's children); that residual
        # rank-1-per-parent issue is broken by the existing ridge floor.
        A_tax_full = _orthogonalize_path_indicators(A_tax_full, tax_tree)

    # Species path indicator for each group
    sp_indices = []
    for _, row in gm.iterrows():
        sp = str(row[species_col])
        sp_indices.append(sp_to_idx.get(sp, 0))
    A_species_full = sp_design.A_species[sp_indices].toarray()
    sp_indices_arr = np.asarray(sp_indices, dtype=np.int64)

    # ── Species × taxonomy interaction (sum-to-zero, node-restricted) ────
    # For each taxonomy level L and each node n at that level, let
    #   S(n) = species observed in at least one pseudobulk group with
    #          ancestor-at-L == n.
    # If |S(n)| < 2, no identifiable species contrast exists at this node →
    # no columns are generated for it. Otherwise we emit K = |S(n)| raw
    # one-hot columns (one per species ∈ S(n)). The sum-to-zero constraint
    # Σ_s β[s,n] = 0 is imposed softly during fitting via the quadratic
    # anchor (SUMZERO_ANCHOR_MU), keeping the L1 penalty fully symmetric
    # across species.
    species_tax_blocks: dict[str, np.ndarray] = {}
    species_tax_meta: dict[str, pd.DataFrame] = {}
    species_tax_node_groups: dict[str, list[list[int]]] = {}
    node_table = tax_tree.node_table

    # Precompute each group's ancestor-id at every level.
    leaf_path_parts = [tax_tree.leaf_ids[ti].split("/") for ti in tax_indices]

    for level in taxonomy_cols:
        level_nodes = node_table[node_table["level"] == level]
        if len(level_nodes) == 0:
            continue
        level_idx_in_cols = taxonomy_cols.index(level)
        level_node_ids = list(level_nodes["node_id"].values)
        level_node_set = set(level_node_ids)

        # Ancestor node id at this level for each group (None if shorter path).
        ancestor_at_level = np.empty(n, dtype=object)
        for grp_idx in range(n):
            parts = leaf_path_parts[grp_idx]
            if level_idx_in_cols < len(parts):
                anc = "/".join(parts[: level_idx_in_cols + 1])
                ancestor_at_level[grp_idx] = anc if anc in level_node_set else None
            else:
                ancestor_at_level[grp_idx] = None

        cols_list: list[np.ndarray] = []
        meta_rows: list[dict] = []
        node_groups: list[list[int]] = []

        for node_id in level_node_ids:
            node_mask = ancestor_at_level == node_id
            if not node_mask.any():
                continue
            species_at_node = np.unique(sp_indices_arr[node_mask])
            K = int(species_at_node.size)
            if K < 2:
                # No identifiable cross-species contrast at this node.
                continue
            group_cols: list[int] = []
            # Centered (sum-to-zero by row) indicators: for each row in the
            # node, the K values across species sum to 0 (each = 1 - 1/K for
            # the species present, -1/K for the others). This makes the
            # species_tax block ORTHOGONAL to ``tax_global[n]`` and to any
            # constant column, eliminating the inter-family collinearity
            # that would otherwise blow up joint Wald SEs. The within-block
            # rank-deficiency (K columns spanning a K-1-dim subspace per
            # node) is broken softly by the quadratic anchor mu * (Σβ)^2.
            inv_K = 1.0 / float(K)
            for s_idx in species_at_node:
                sp_at_node_mask = node_mask & (sp_indices_arr == int(s_idx))
                col = np.zeros(n, dtype=np.float64)
                # +1 - 1/K where (species, node) matches; -1/K for other
                # species rows within the same node; 0 outside the node.
                col[node_mask] = -inv_K
                col[sp_at_node_mask] = 1.0 - inv_K
                col_index = len(cols_list)
                cols_list.append(col)
                meta_rows.append({
                    "col_index": col_index,
                    "level": level,
                    "node_id": node_id,
                    "node_label": node_id.split("/")[-1]
                    if isinstance(node_id, str) else str(node_id),
                    "species": sp_design.species_order[int(s_idx)],
                    "n_species_at_node": K,
                })
                group_cols.append(col_index)
            node_groups.append(group_cols)

        if cols_list:
            family_name = f"species_tax_{level}"
            species_tax_blocks[family_name] = np.column_stack(cols_list)
            species_tax_meta[family_name] = pd.DataFrame(meta_rows)
            species_tax_node_groups[family_name] = node_groups

    if orthogonal_tree and species_tax_blocks:
        species_tax_blocks = _orthogonalize_interaction_blocks(
            blocks=species_tax_blocks,
            taxonomy_cols=taxonomy_cols,
            nuisance=np.column_stack([A_tax_full, A_species_full]),
        )

    # Batch design (one-hot, unpenalized)
    batch_design = None
    if batch_col is not None and batch_col in gm.columns:
        batches = sorted(gm[batch_col].unique())
        if len(batches) > 1:
            # Use reference coding: drop first level
            batch_to_idx = {b: i for i, b in enumerate(batches[1:])}
            batch_design = np.zeros((n, len(batches) - 1), dtype=np.float64)
            for grp_idx, (_, row) in enumerate(gm.iterrows()):
                b = str(row[batch_col])
                if b in batch_to_idx:
                    batch_design[grp_idx, batch_to_idx[b]] = 1.0

    # Donor design
    donor_design = None
    if donor_col is not None and donor_col in gm.columns:
        donors = sorted(gm[donor_col].unique())
        if len(donors) > 1:
            donor_to_idx = {d: i for i, d in enumerate(donors[1:])}
            donor_design = np.zeros((n, len(donors) - 1), dtype=np.float64)
            for grp_idx, (_, row) in enumerate(gm.iterrows()):
                d = str(row[donor_col])
                if d in donor_to_idx:
                    donor_design[grp_idx, donor_to_idx[d]] = 1.0

    designs = {
        "tax_global": A_tax_full,
        "species_global": A_species_full,
    }
    designs.update(species_tax_blocks)

    if batch_design is not None:
        designs["batch"] = batch_design
    if donor_design is not None:
        designs["donor"] = donor_design

    return designs, species_tax_meta, species_tax_node_groups


def _build_dispersion_designs(
    designs: dict[str, np.ndarray],
    group_meta: pd.DataFrame,
    dispersion_families: list[str],
    donor_col: str | None,
    batch_col: str | None,
    min_replicates_per_node: int,
    min_groups_per_node: int = 2,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Construct dispersion design matrices by subsetting + masking mean designs.

    A dispersion design column is identifiable only if its non-zero rows
    correspond to enough independent replicates. We mask a column when:

    1. It is constant (variance 0) across pseudobulk groups — would be
       absorbed by phi0, causing rank deficiency.
    2. Fewer than `min_groups_per_node` pseudobulk groups load on it.
    3. Fewer than `min_replicates_per_node` distinct donor labels (or batch
       labels if donor_col is None) appear among loading groups — without
       independent replicates dispersion is unidentified.
    4. It exactly duplicates a column already kept (rank deficiency that
       commonly arises in short-tree slices: e.g. when a Class has a single
       Subclass, the Class and Subclass indicator columns are identical and
       L1 would pick both, splitting the same effect spuriously).

    Returns
    -------
    disp_designs : dict[family -> (n_groups, n_active_cols) ndarray]
    active_indices : dict[family -> ndarray of int]
        Original column indices kept after masking; used to map dispersion
        coefficients back to tree node ids in the result.
    """
    rep_col = donor_col if donor_col is not None else batch_col
    if rep_col is None or rep_col not in group_meta.columns:
        # No explicit replicate label; fall back to pseudobulk group identity.
        # Each group becomes its own "replicate"; only the min_groups check applies.
        rep_labels = np.arange(len(group_meta), dtype=object)
    else:
        rep_labels = group_meta[rep_col].astype(str).values

    disp_designs: dict[str, np.ndarray] = {}
    active_indices: dict[str, np.ndarray] = {}
    for family in dispersion_families:
        # species_tax_* families now use a raw K-one-hot sum-to-zero
        # parameterization on the mean side that depends on the per-node
        # quadratic anchor for identifiability. Re-using those columns for
        # dispersion would (a) not inherit the anchor (so the K columns are
        # again rank-deficient on the dispersion side) and (b) make the
        # `col != 0` masking heuristic meaningless. Refuse them explicitly;
        # users who really want species-by-clade dispersion should build a
        # bespoke design.
        if family.startswith("species_tax_"):
            warnings.warn(
                f"Dispersion family '{family}' is a species_tax interaction "
                "and is not supported as a dispersion design (the mean-side "
                "sum-to-zero anchor does not transfer); skipping."
            )
            continue
        if family not in designs:
            warnings.warn(
                f"Dispersion family '{family}' not present in mean designs; skipping."
            )
            continue
        X = designs[family]
        n_cols = X.shape[1]
        keep: list[int] = []
        kept_columns: list[np.ndarray] = []  # for dedup
        for j in range(n_cols):
            col = X[:, j]
            nz_mask = col != 0
            n_loaded = int(nz_mask.sum())
            if n_loaded < min_groups_per_node:
                continue
            # Reject constant columns (variance 0 across groups).
            if np.allclose(col, col[0]):
                continue
            n_replicates = len(np.unique(rep_labels[nz_mask]))
            if n_replicates < min_replicates_per_node:
                continue
            # Reject exact duplicates of already-kept columns (rank deficiency).
            is_dup = any(np.array_equal(col, kc) for kc in kept_columns)
            if is_dup:
                continue
            keep.append(j)
            kept_columns.append(col)
        if not keep:
            warnings.warn(
                f"All columns in dispersion family '{family}' were masked out "
                f"(min_replicates_per_node={min_replicates_per_node}, "
                f"min_groups_per_node={min_groups_per_node}). "
                "Family will be dropped from dispersion fit."
            )
            continue
        disp_designs[family] = X[:, keep].astype(np.float64, copy=True)
        active_indices[family] = np.asarray(keep, dtype=np.int64)

    return disp_designs, active_indices


def _compute_dispersion_fisher_weights(
    Y: torch.Tensor,
    mu: torch.Tensor,
    log_overdisp: torch.Tensor,
) -> torch.Tensor:
    """Observed Fisher info diagonal of NLL w.r.t. log_overdisp (delta = -log theta).

    Computed by double backprop on the NB NLL evaluated element-wise. Returns
    a tensor of the same shape as `log_overdisp` (n_groups, n_genes) holding
    -d^2 logL / d delta^2 per (i,g) — these are the per-observation weights
    that make the L1 penalty on dispersion coefficients comparable in scale
    to the mean-side Fisher-weighted penalty.
    """
    delta = log_overdisp.detach().clone().requires_grad_(True)
    theta = torch.exp(-delta.clamp(min=-10.0, max=10.0))
    mu_d = mu.detach()
    # Element-wise NLL (no sum); use same form as _nb_nll but elementwise.
    eps = 1e-8
    nll_elem = (
        -torch.lgamma(Y + theta)
        + torch.lgamma(theta)
        + torch.lgamma(Y + 1)
        - theta * torch.log(theta / (theta + mu_d) + eps)
        - Y * torch.log(mu_d / (theta + mu_d) + eps)
    )
    grad1 = torch.autograd.grad(nll_elem.sum(), delta, create_graph=True)[0]
    # NLL is separable across (i,g) so grad1.sum().grad wrt delta gives diag(H).
    grad2 = torch.autograd.grad(grad1.sum(), delta)[0]
    # Numerical floor; weights must be positive for the sqrt below.
    return grad2.detach().clamp(min=1e-6)


def _compute_fisher_weights(
    mu: torch.Tensor, theta: torch.Tensor
) -> torch.Tensor:
    """NB Fisher information weights: W = mu / (1 + mu/theta)."""
    return mu / (1.0 + mu / theta)


def _compute_penalty_scales(
    designs: dict[str, np.ndarray],
    W: torch.Tensor,
    l1_lambdas: dict[str, float] | float,
    use_family_size_calibration: bool = True,
) -> dict[str, torch.Tensor]:
    """Compute per-column penalty scales c_j = sqrt(sum_i W_i * x_ij^2)."""
    penalty_scales = {}
    for family, X in designs.items():
        if isinstance(l1_lambdas, dict):
            lam = l1_lambdas.get(family, 0.1)
        else:
            lam = l1_lambdas
        X_t = torch.tensor(X, dtype=W.dtype, device=W.device)
        W_mean = W.mean(dim=1)  # (n_groups,)
        c_j = torch.sqrt((W_mean.unsqueeze(1) * X_t**2).sum(dim=0) + 1e-10)
        p_family = X_t.shape[1]
        if use_family_size_calibration and p_family > 1:
            lam_scaled = lam * math.sqrt(2 * math.log(p_family))
        else:
            lam_scaled = lam
        penalty_scales[family] = c_j * lam_scaled
    return penalty_scales


def _nb_nll(
    Y: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """Stable NB negative log-likelihood with log-link."""
    # Y ~ NB(mu, theta): p(y) = C(y,theta) * (theta/(theta+mu))^theta * (mu/(theta+mu))^y
    # NLL = -lgamma(y+theta) + lgamma(theta) + lgamma(y+1)
    #        - theta*log(theta/(theta+mu)) - y*log(mu/(theta+mu))
    eps = 1e-8
    nll = (
        -torch.lgamma(Y + theta)
        + torch.lgamma(theta)
        + torch.lgamma(Y + 1)
        - theta * torch.log(theta / (theta + mu) + eps)
        - Y * torch.log(mu / (theta + mu) + eps)
    )
    return nll.sum()


def _orthogonalize_path_indicators(
    A: np.ndarray, tax_tree: "TaxonomyTree"
) -> np.ndarray:
    """Residualize every taxonomy level against all of its ancestor levels.

    Input:  A[i, n] = 1 if group i's path passes through node n.
    Output: columns at level L are jointly residualized against the span of
            every earlier level. Each residual is rescaled to the original
            column L2 norm, retaining a common RMS log-effect interpretation
            while making comparisons across levels insensitive to nesting.
    """
    A_orth = np.zeros_like(A, dtype=np.float64)
    node_table = tax_tree.node_table
    node_id_to_col = {nid: i for i, nid in enumerate(tax_tree.node_ids)}
    prior = np.empty((A.shape[0], 0), dtype=np.float64)
    for level_idx in sorted(node_table["level_idx"].unique()):
        rows = node_table[node_table["level_idx"] == level_idx]
        indices = [node_id_to_col[node_id] for node_id in rows["node_id"]]
        raw = A[:, indices].astype(np.float64, copy=False)
        residual = _residualize_and_rescale(raw=raw, basis=prior)
        A_orth[:, indices] = residual
        prior = np.column_stack([prior, residual])
    return A_orth


def _orthonormal_basis(matrix: np.ndarray) -> np.ndarray:
    """Return a tolerance-truncated orthonormal basis for a column space."""
    if matrix.shape[1] == 0:
        return np.empty((matrix.shape[0], 0), dtype=np.float64)
    u, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    if singular_values.size == 0:
        return np.empty((matrix.shape[0], 0), dtype=np.float64)
    tolerance = np.finfo(np.float64).eps * max(matrix.shape) * singular_values[0]
    return cast(np.ndarray, u[:, singular_values > tolerance])


def _residualize_and_rescale(*, raw: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project columns off a basis and restore each original L2 norm."""
    q = _orthonormal_basis(basis)
    residual = raw - q @ (q.T @ raw) if q.shape[1] else raw.copy()
    raw_norm = np.linalg.norm(raw, axis=0)
    residual_norm = np.linalg.norm(residual, axis=0)
    usable = residual_norm > 1e-10
    residual[:, usable] *= raw_norm[usable] / residual_norm[usable]
    residual[:, ~usable] = 0.0
    return residual


def _orthogonalize_interaction_blocks(
    *,
    blocks: dict[str, np.ndarray],
    taxonomy_cols: list[str],
    nuisance: np.ndarray,
) -> dict[str, np.ndarray]:
    """Make species-taxonomy levels orthogonal to nuisance and ancestors."""
    result: dict[str, np.ndarray] = {}
    prior = nuisance.astype(np.float64, copy=False)
    for level in taxonomy_cols:
        family = f"species_tax_{level}"
        if family not in blocks:
            continue
        residual = _residualize_and_rescale(raw=blocks[family], basis=prior)
        result[family] = residual
        prior = np.column_stack([prior, residual])
    return result


def _smooth_l1(beta: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Smooth L1 approximation: sqrt(beta^2 + eps).

    eps=1e-6 corresponds to a soft-threshold transition at |β|≈1e-3, well
    below the post-fit selection threshold (typically 1e-2 or larger).
    With the legacy eps=1e-3 the gradient at |β|=0.03 was only 0.95·sign(β),
    causing systematic attenuation of small true effects (~30-50% bias on
    planted effects with magnitude <0.5 in the sim eval).
    """
    return torch.sqrt(beta**2 + eps)


def _empirical_bayes_penalty(
    *,
    beta: torch.Tensor,
    prior: ShrinkagePrior,
    scale: float,
    responsibilities: torch.Tensor | None = None,
    spike_scale: float = 0.02,
) -> torch.Tensor:
    """Return a fixed-scale Gaussian or Laplace negative log-prior."""
    if prior is ShrinkagePrior.GAUSSIAN:
        return 0.5 * (beta / scale).square().sum()
    if prior is ShrinkagePrior.LAPLACE:
        return (_smooth_l1(beta) / scale).sum()
    if responsibilities is None:
        raise ValueError("Spike-and-slab shrinkage requires responsibilities.")
    precision = responsibilities / scale**2 + (1.0 - responsibilities) / spike_scale**2
    return cast(torch.Tensor, 0.5 * (precision * beta.square()).sum())


def _normal_expected_absolute(
    *, mean: torch.Tensor, variance: torch.Tensor
) -> torch.Tensor:
    """Return E|X| for elementwise normal means and variances."""
    sd = torch.sqrt(variance.clamp(min=1e-12))
    standardized = mean.abs() / sd
    return (
        sd * math.sqrt(2.0 / math.pi) * torch.exp(-0.5 * standardized.square())
        + mean.abs() * torch.erf(standardized / math.sqrt(2.0))
    )


def _update_empirical_bayes_scales(
    *,
    betas: dict[str, torch.Tensor],
    designs: dict[str, torch.Tensor],
    W: torch.Tensor,
    scales: dict[str, float],
    config: EmpiricalBayesConfig,
) -> dict[str, float]:
    """Perform one damped approximate-EM update of level prior scales."""
    updated = dict(scales)
    for family, old_scale in scales.items():
        beta = betas[family].detach()
        X = designs[family]
        information = X.square().T @ W
        usable = information > 1e-8
        if not bool(usable.any()):
            continue
        if config.prior is ShrinkagePrior.GAUSSIAN:
            posterior_variance = 1.0 / (information + old_scale**-2)
            second_moment = beta.square() + posterior_variance
            target = float(torch.sqrt(second_moment[usable].mean()).cpu())
        else:
            # A local Gaussian approximation supplies uncertainty for the
            # Laplace M-step b = mean(E|beta|). The stability precision avoids
            # infinite moments for nearly unsupported columns.
            posterior_variance = 1.0 / (information + old_scale**-2)
            expected_absolute = _normal_expected_absolute(
                mean=beta, variance=posterior_variance
            )
            target = float(expected_absolute[usable].mean().cpu())
        target = min(max(target, config.min_scale), config.max_scale)
        updated[family] = (
            (1.0 - config.damping) * old_scale + config.damping * target
        )
    return updated


def _update_spike_slab_state(
    *,
    betas: dict[str, torch.Tensor],
    designs: dict[str, torch.Tensor],
    W: torch.Tensor,
    scales: dict[str, float],
    inclusion: dict[str, float],
    responsibilities: dict[str, torch.Tensor],
    config: EmpiricalBayesConfig,
    update_global: bool,
) -> tuple[dict[str, float], dict[str, float], dict[str, torch.Tensor]]:
    """Update local mixture assignments and optional global spike-slab state."""
    updated_scales = dict(scales)
    updated_inclusion = dict(inclusion)
    updated_responsibilities: dict[str, torch.Tensor] = {}
    for family, slab_scale in scales.items():
        beta = betas[family].detach()
        information = designs[family].square().T @ W
        usable = information > 1e-8
        z = torch.zeros_like(beta)
        z[usable] = beta[usable] * (
            information[usable] + slab_scale**-2
        ) / information[usable]
        sampling_variance = torch.full_like(information, 1e8)
        sampling_variance[usable] = 1.0 / information[usable]
        pi = min(max(inclusion[family], 1e-4), 1.0 - 1e-4)
        slab_variance = sampling_variance + slab_scale**2
        spike_variance = sampling_variance + config.spike_scale**2
        log_slab = (
            math.log(pi)
            - 0.5 * torch.log(slab_variance)
            - 0.5 * z.square() / slab_variance
        )
        log_spike = (
            math.log1p(-pi)
            - 0.5 * torch.log(spike_variance)
            - 0.5 * z.square() / spike_variance
        )
        new_r = torch.sigmoid((log_slab - log_spike).clamp(-30.0, 30.0))
        new_r = torch.where(usable, new_r, torch.zeros_like(new_r))
        updated_responsibilities[family] = new_r
        if not update_global:
            continue
        target_pi = float(new_r[usable].mean().cpu())
        target_pi = min(max(target_pi, 0.001), 0.999)
        updated_inclusion[family] = (
            (1.0 - config.damping) * pi + config.damping * target_pi
        )
        posterior_variance = 1.0 / (information + slab_scale**-2)
        posterior_mean = posterior_variance * information * z
        second_moment = posterior_mean.square() + posterior_variance
        total_weight = new_r[usable].sum()
        if float(total_weight.cpu()) > 1e-8:
            target_scale = float(
                torch.sqrt(
                    (new_r[usable] * second_moment[usable]).sum() / total_weight
                ).cpu()
            )
            target_scale = min(
                max(target_scale, config.min_scale), config.max_scale
            )
            updated_scales[family] = (
                (1.0 - config.damping) * slab_scale
                + config.damping * target_scale
            )
    return updated_scales, updated_inclusion, updated_responsibilities


def _sumzero_anchor_penalty(
    betas: dict[str, torch.Tensor],
    species_tax_node_groups: dict[str, list[list[int]]] | None,
    mu: float,
) -> torch.Tensor:
    """Quadratic sum-to-zero anchor for ``species_tax_*`` interaction families.

    For each family in ``species_tax_node_groups`` and each node (a group of
    column indices), adds ``0.5 * mu * (sum_{c in group} beta[c, g])^2`` summed
    over genes ``g``. This enforces ``sum_{s in S(n)} beta[s, n, g] = 0``
    softly during optimization while keeping the L1 penalty (applied
    elsewhere) symmetric across species.

    Returns a scalar 0-d tensor (uses ``betas`` value's device/dtype).
    """
    if not species_tax_node_groups:
        # Return a zero tensor that shares device with one of the betas to keep
        # the loss graph consistent. Pick any beta if available; else fall back
        # to a plain scalar (caller adds it to a tensor loss anyway).
        any_beta = next(iter(betas.values()), None)
        if any_beta is None:
            return torch.tensor(0.0)
        return torch.zeros((), dtype=any_beta.dtype, device=any_beta.device)
    total = None
    for family, groups in species_tax_node_groups.items():
        if family not in betas:
            continue
        beta = betas[family]  # (n_cols, n_genes)
        for group in groups:
            if not group:
                continue
            idx = torch.as_tensor(group, dtype=torch.long, device=beta.device)
            node_sum = beta.index_select(0, idx).sum(dim=0)  # (n_genes,)
            term = 0.5 * mu * (node_sum ** 2).sum()
            total = term if total is None else total + term
    if total is None:
        any_beta = next(iter(betas.values()))
        return torch.zeros((), dtype=any_beta.dtype, device=any_beta.device)
    return cast(torch.Tensor, total)


def _group_lasso_penalty(
    betas: dict[str, torch.Tensor],
    species_tax_node_groups: dict[str, list[list[int]]] | None,
    designs: dict[str, torch.Tensor],
    W_mean: torch.Tensor,
    lam: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Group-lasso penalty for ``species_tax_*`` interaction families.

    For each taxonomy node within a species_tax family, penalizes the
    Fisher-weighted group-L2 norm of the K coefficient vectors:

        λ · √(2·log(G_L)) · √K · Σ_g √(Σ_k c_k² · β[k,g]²)

    where:
        G_L = number of node-groups in the family (multiplicity correction),
        K   = number of species columns at the node,
        c_k = √(Σ_i W_i · X[i,k]²) is the per-column Fisher-weighted norm
              (same as the c_j used in the standard L1 penalty for other
              families).

    The three scaling factors provide automatic, principled calibration:

    1. **Fisher weights c_k**: scale-equivariance (same as L1 families).
       Detection threshold in effect-size units is independent of node
       information content.

    2. **√(2·log(G_L))**: multiplicity correction across G_L nodes within
       a level (Bonferroni-style universal threshold).

    3. **√K**: chi-distribution calibration. Under the null, the weighted
       group norm of K independent noise terms is chi_K with mean ~√K.
       Without this factor, null nodes exceed the threshold ~√K times
       more often than individual columns exceed the L1 threshold.

    This replaces per-column L1 + support promotion for species_tax families,
    providing:
      - Node-level selection (all-or-nothing per node per gene)
      - Automatic balance across taxonomy levels (no manual lambda tuning)
      - Correct multiple-testing calibration via √(2·log(G_L)) · √K

    Returns a scalar 0-d tensor.
    """
    if not species_tax_node_groups:
        any_beta = next(iter(betas.values()), None)
        if any_beta is None:
            return torch.tensor(0.0)
        return torch.zeros((), dtype=any_beta.dtype, device=any_beta.device)

    total = None
    for family, groups in species_tax_node_groups.items():
        if family not in betas:
            continue
        beta = betas[family]  # (n_cols, n_genes)
        X_t = designs[family]  # (n_groups, n_cols)
        G_L = len(groups)  # number of nodes at this level
        if G_L < 1:
            continue
        # Multiplicity correction: sqrt(2*log(G_L)) accounts for testing
        # G_L nodes. This is less aggressive than the per-column L1's
        # sqrt(2*log(p_family)) since we're making G_L group tests rather
        # than p_family = sum(K_g) individual tests.
        level_scale = math.sqrt(2.0 * math.log(max(G_L, 2)))

        for group in groups:
            if not group or len(group) < 2:
                continue
            K = len(group)
            idx = torch.as_tensor(group, dtype=torch.long, device=beta.device)

            # Fisher-weighted group norm per gene:
            # sqrt(sum_k c_k^2 * beta_k^2)
            # Uses the same per-column Fisher weights as the L1 penalty,
            # ensuring scale-equivariance across taxonomy levels.
            col_block = X_t[:, group]  # (n_groups, K)
            c_k_sq = (W_mean.unsqueeze(1) * col_block ** 2).sum(dim=0)  # (K,)

            # Group norm: per-gene weighted L2 norm
            beta_node = beta.index_select(0, idx)  # (K, n_genes)
            weighted_sq = c_k_sq.unsqueeze(1) * (beta_node ** 2)  # (K, n_genes)
            group_norm_per_gene = torch.sqrt(
                weighted_sq.sum(dim=0) + eps
            )  # (n_genes,)

            # sqrt(K) calibration (Yuan & Lin 2006): under the null, the
            # Fisher-weighted group norm of K columns with independent noise
            # follows a chi_K distribution with mean ~sqrt(K). Without this
            # factor, null nodes exceed the detection threshold more often
            # than individual columns exceed the L1 threshold. The sum-to-zero
            # anchor reduces effective df to K-1, but the anchor is soft and
            # does not perfectly eliminate the Kth direction — using sqrt(K)
            # gives cleaner empirical calibration.
            group_scale = math.sqrt(K)

            term = lam * level_scale * group_scale * group_norm_per_gene.sum()
            total = term if total is None else total + term

    if total is None:
        any_beta = next(iter(betas.values()))
        return torch.zeros((), dtype=any_beta.dtype, device=any_beta.device)
    return total


def _project_sumzero(
    betas: dict[str, np.ndarray],
    species_tax_node_groups: dict[str, list[list[int]]] | None,
) -> dict[str, np.ndarray]:
    """Hard-project species_tax_* per-node coefficients onto sum-to-zero.

    The augmented quadratic anchor enforces ``sum ~ 0`` to numerical
    precision (typically O(1/mu)). This helper performs a final exact
    projection so the stored coefficients literally satisfy the constraint.
    The linear predictor at the L1-selected support is preserved exactly only
    when the anchor was active during fitting (the projection direction lies
    in the unidentified null space of the linear predictor at convergence).
    """
    if not species_tax_node_groups:
        return betas
    out = dict(betas)
    for family, groups in species_tax_node_groups.items():
        if family not in out:
            continue
        b = out[family].copy()
        for group in groups:
            if len(group) < 2:
                continue
            sub = b[group, :]
            sub -= sub.mean(axis=0, keepdims=True)
            b[group, :] = sub
        out[family] = b
    return out


def _fit_gene_chunk(
    Y_chunk: np.ndarray,
    designs: dict[str, np.ndarray],
    library_sizes: np.ndarray,
    l1_lambdas: dict[str, float] | float,
    max_iter: int,
    device: str,
    refit_support: bool = True,
    residual_lambda: float | None = None,
    disp_designs: dict[str, np.ndarray] | None = None,
    dispersion_lambda: float | None = None,
    disp_offset: np.ndarray | None = None,
    species_tax_node_groups: dict[str, list[list[int]]] | None = None,
    sumzero_anchor_mu: float = SUMZERO_ANCHOR_MU,
    empirical_bayes: EmpiricalBayesConfig | None = None,
    empirical_bayes_scales: dict[str, float] | None = None,
    empirical_bayes_inclusion: dict[str, float] | None = None,
    learn_empirical_bayes_scales: bool = False,
    refit_all_species_tax: bool = False,
) -> dict[str, Any]:
    """Fit NB model for a chunk of genes.

    Parameters
    ----------
    residual_lambda : float, optional
        If provided, adds a penalized per-group intercept (gamma_ig) that
        captures variance the tree design cannot represent. Penalized with
        L1 at this strength to encourage sparsity — most groups should be
        well-explained by the tree, with residuals only where needed.
    disp_designs : dict[str, ndarray], optional
        If provided, fits tree-structured dispersion in a second stage
        (after mean post-selection refit), with one design matrix per
        dispersion family. Coefficients parameterize log_overdispersion
        (= -log theta); positive coefficient ⇒ more variable than baseline.
    dispersion_lambda : float, optional
        L1 strength for dispersion families. Required when disp_designs is given.
    disp_offset : ndarray, optional
        Known per-group offset added to log_overdisp (e.g., -log(n_cells) to
        account for pseudobulk sums of iid cell-level NB observations).
    species_tax_node_groups : dict[str, list[list[int]]], optional
        For each species_tax_* family, a list of column-index groups (one per
        node). Used to apply the quadratic sum-to-zero anchor that makes the
        raw K-one-hot species_tax parameterization identifiable.
    sumzero_anchor_mu : float
        Strength of the per-node sum-to-zero quadratic anchor. See
        ``SUMZERO_ANCHOR_MU``.
    """
    dev = torch.device(device)
    n_groups, n_genes = Y_chunk.shape

    Y = torch.tensor(Y_chunk, dtype=torch.float64, device=dev)
    offset = torch.tensor(np.log(library_sizes + 1e-8), dtype=torch.float64, device=dev)

    # Initialize parameters
    alpha = torch.zeros(n_genes, dtype=torch.float64, device=dev, requires_grad=True)
    log_theta = torch.full((n_genes,), 2.0, dtype=torch.float64, device=dev, requires_grad=True)

    betas: dict[str, torch.Tensor] = {}
    for family, X in designs.items():
        n_cols = X.shape[1]
        betas[family] = torch.zeros(
            n_cols, n_genes, dtype=torch.float64, device=dev, requires_grad=True
        )

    # Per-group residual intercepts (only if requested)
    gamma: torch.Tensor | None = None
    if residual_lambda is not None:
        gamma = torch.zeros(
            n_groups, n_genes, dtype=torch.float64, device=dev, requires_grad=True
        )

    design_tensors = {
        k: torch.tensor(v, dtype=torch.float64, device=dev)
        for k, v in designs.items()
    }
    eb_scales = (
        dict(empirical_bayes_scales)
        if empirical_bayes_scales is not None
        else {
            family: empirical_bayes.initial_scale
            for family in designs
            if empirical_bayes is not None and family.startswith("species_tax_")
        }
    )
    eb_inclusion = (
        dict(empirical_bayes_inclusion)
        if empirical_bayes_inclusion is not None
        else {family: empirical_bayes.initial_inclusion for family in eb_scales}
        if empirical_bayes is not None
        else {}
    )
    eb_responsibilities = {
        family: torch.full_like(
            betas[family],
            1.0
            if empirical_bayes is not None
            and empirical_bayes.prior is ShrinkagePrior.SPIKE_SLAB
            else eb_inclusion[family],
        )
        for family in eb_scales
    }

    all_params = [alpha, log_theta] + list(betas.values())
    if gamma is not None:
        all_params.append(gamma)
    optimizer = torch.optim.Adam(all_params, lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5
    )

    best_loss = float("inf")
    patience_counter = 0

    for iteration in range(max_iter):
        optimizer.zero_grad()

        # Compute linear predictor
        eta = offset.unsqueeze(1) + alpha.unsqueeze(0)
        for family, X_t in design_tensors.items():
            eta = eta + X_t @ betas[family]
        if gamma is not None:
            eta = eta + gamma

        mu = torch.exp(eta.clamp(max=20.0))
        theta = torch.exp(log_theta.clamp(min=-5.0, max=10.0)).unsqueeze(0)

        loss = _nb_nll(Y, mu, theta)

        # Nuisance and non-interaction families retain Fisher-weighted L1.
        # Species-taxonomy families optionally use learned Gaussian/Laplace
        # priors in common biological log-effect units.
        # Per-column L1 provides strong per-coefficient sparsity. Cross-level
        # balance is achieved by removing support promotion in _refit_support
        # (the original cause of the cascade).
        W = _compute_fisher_weights(mu.detach(), theta.detach())
        W_mean = W.mean(dim=1)
        for family, beta in betas.items():
            if empirical_bayes is not None and family in eb_scales:
                loss = loss + _empirical_bayes_penalty(
                    beta=beta,
                    prior=empirical_bayes.prior,
                    scale=eb_scales[family],
                    responsibilities=eb_responsibilities.get(family),
                    spike_scale=empirical_bayes.spike_scale,
                )
                continue
            if isinstance(l1_lambdas, dict):
                lam = l1_lambdas.get(family, 0.1)
            else:
                lam = l1_lambdas
            X_t = design_tensors[family]
            c_j = torch.sqrt((W_mean.unsqueeze(1) * X_t**2).sum(dim=0) + 1e-10)
            p_family = X_t.shape[1]
            if p_family > 1:
                lam_scaled = lam * math.sqrt(2 * math.log(p_family))
            else:
                lam_scaled = lam
            penalty = lam_scaled * (c_j.unsqueeze(1) * _smooth_l1(beta)).sum()
            loss = loss + penalty

        # Sum-to-zero anchor: still needed to keep the K-one-hot
        # parameterization identifiable (group-lasso selects nodes but
        # doesn't constrain the within-node null direction).
        if species_tax_node_groups:
            loss = loss + _sumzero_anchor_penalty(
                betas, species_tax_node_groups, sumzero_anchor_mu,
            )

        # Penalty on residual intercepts: simple L1 scaled by sqrt(2*log(n_groups))
        if gamma is not None:
            assert residual_lambda is not None
            residual_scale = residual_lambda * math.sqrt(2 * math.log(n_groups))
            loss = loss + residual_scale * _smooth_l1(gamma).sum()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 5.0)
        optimizer.step()
        if (
            empirical_bayes is not None
            and iteration >= empirical_bayes.warmup_iterations
            and (iteration + 1) % empirical_bayes.update_interval == 0
        ):
            if (
                empirical_bayes.prior is not ShrinkagePrior.SPIKE_SLAB
                and learn_empirical_bayes_scales
            ):
                eb_scales = _update_empirical_bayes_scales(
                    betas=betas,
                    designs=design_tensors,
                    W=W,
                    scales=eb_scales,
                    config=empirical_bayes,
                )
        scheduler.step(loss.item())

        current_loss = loss.item()
        if current_loss < best_loss - 1e-4:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > 50:
                break

    if empirical_bayes is not None and empirical_bayes.prior is ShrinkagePrior.SPIKE_SLAB:
        # Fit under the slab first, then perform mixture EB on the resulting
        # approximately normal coefficient estimates. This avoids the local
        # mode where a strong spike prevents a real coefficient escaping zero.
        n_hyper_updates = 10 if learn_empirical_bayes_scales else 1
        for _ in range(n_hyper_updates):
            eb_scales, eb_inclusion, eb_responsibilities = _update_spike_slab_state(
                betas=betas,
                designs=design_tensors,
                W=W,
                scales=eb_scales,
                inclusion=eb_inclusion,
                responsibilities=eb_responsibilities,
                config=empirical_bayes,
                update_global=learn_empirical_bayes_scales,
            )
        if not learn_empirical_bayes_scales:
            with torch.no_grad():
                for family, slab_scale in eb_scales.items():
                    beta = betas[family]
                    information = design_tensors[family].square().T @ W
                    usable = information > 1e-8
                    z = torch.zeros_like(beta)
                    z[usable] = beta[usable] * (
                        information[usable] + slab_scale**-2
                    ) / information[usable]
                    slab_mean = information * z / (
                        information + slab_scale**-2
                    )
                    spike_mean = information * z / (
                        information + empirical_bayes.spike_scale**-2
                    )
                    responsibility = eb_responsibilities[family]
                    beta.copy_(
                        responsibility * slab_mean
                        + (1.0 - responsibility) * spike_mean
                    )

    # Extract results
    result_betas = {}
    result_nonzero = {}
    threshold = 0.01

    for family, beta in betas.items():
        b = beta.detach().cpu().numpy()
        result_betas[family] = b
        # Per-column selection for ALL families (no node-level promotion).
        result_nonzero[family] = np.abs(b) > threshold

    # Post-selection refit if requested
    if refit_support:
        gamma_np = gamma.detach().cpu().numpy() if gamma is not None else None
        selection_nonzero = {family: mask.copy() for family, mask in result_nonzero.items()}
        refit_nonzero = {family: mask.copy() for family, mask in result_nonzero.items()}
        if refit_all_species_tax:
            for family, mask in refit_nonzero.items():
                if family.startswith("species_tax_"):
                    mask[...] = True
        result_betas, refit_nonzero, gamma_np, refit_alpha, refit_log_theta = _refit_support(
            Y_chunk, designs, library_sizes, result_betas, refit_nonzero,
            max_iter // 2, device, gamma_init=gamma_np,
            species_tax_node_groups=species_tax_node_groups,
            sumzero_anchor_mu=sumzero_anchor_mu,
        )
        result_nonzero = selection_nonzero
        alpha_np = refit_alpha
        log_theta_np = refit_log_theta
    else:
        gamma_np = gamma.detach().cpu().numpy() if gamma is not None else None
        alpha_np = alpha.detach().cpu().numpy()
        log_theta_np = log_theta.detach().cpu().numpy()

    # Note: we intentionally do NOT hard-project species_tax coefficients
    # onto Σ_s β[s,n] = 0 here. Doing so would change the linear predictor
    # because Σ_s X[(s,n)][i] equals the tax_global indicator at node n
    # (not zero), so subtracting the per-node mean from β[s,n] is equivalent
    # to subtracting it from tax_global[n] — and that compensation is not
    # carried out. The quadratic anchor inside the fit/refit drives
    # |Σ_s β[s,n,g]| to O(1/μ) ≈ 1e-2 by construction, which is sufficient
    # for interpretation. Downstream consumers wanting exact sum-to-zero can
    # post-process and apply the matching shift to tax_global themselves.

    # Stage 2: tree-structured dispersion fit (optional).
    disp_result: dict[str, Any] | None = None
    if disp_designs is not None and dispersion_lambda is not None and len(disp_designs) > 0:
        disp_result = _fit_dispersion_stage(
            Y_chunk=Y_chunk,
            designs=designs,
            mean_betas=result_betas,
            alpha=alpha_np,
            log_theta_baseline=log_theta_np,
            library_sizes=library_sizes,
            disp_designs=disp_designs,
            dispersion_lambda=dispersion_lambda,
            disp_offset=disp_offset,
            gamma=gamma_np,
            max_iter=max_iter,
            device=device,
            refit_support=refit_support,
        )
        log_theta_np = disp_result["log_theta_baseline"]

    result = {
        "coefficients": result_betas,
        "nonzero": result_nonzero,
        "alpha": alpha_np,
        "log_theta": log_theta_np,
        "loss": best_loss,
        "empirical_bayes_scales": eb_scales,
        "empirical_bayes_inclusion": eb_inclusion,
    }
    if gamma_np is not None:
        result["gamma"] = gamma_np
    if disp_result is not None:
        result["disp_coefficients"] = disp_result["coefficients"]
        result["disp_nonzero"] = disp_result["nonzero"]
    return result


def _refit_support(
    Y_chunk: np.ndarray,
    designs: dict[str, np.ndarray],
    library_sizes: np.ndarray,
    init_betas: dict[str, np.ndarray],
    nonzero_masks: dict[str, np.ndarray],
    max_iter: int,
    device: str,
    gamma_init: np.ndarray | None = None,
    species_tax_node_groups: dict[str, list[list[int]]] | None = None,
    sumzero_anchor_mu: float = SUMZERO_ANCHOR_MU,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray | None, np.ndarray, np.ndarray]:
    """Refit on selected support — species_tax families frozen at L1 estimates.

    Non-interaction families (tax_global, species_global, batch, donor) are
    refitted without penalty (standard post-selection debiasing). Species_tax
    families remain at their penalized L1 estimates to prevent false-positive
    inflation that occurs with unpenalized refit after removing support
    promotion.

    Returns
    -------
    result_betas, result_nonzero, gamma_np, alpha_np, log_theta_np
    """
    dev = torch.device(device)
    n_groups, n_genes = Y_chunk.shape
    Y = torch.tensor(Y_chunk, dtype=torch.float64, device=dev)
    offset = torch.tensor(np.log(library_sizes + 1e-8), dtype=torch.float64, device=dev)

    alpha = torch.zeros(n_genes, dtype=torch.float64, device=dev, requires_grad=True)
    log_theta = torch.full((n_genes,), 2.0, dtype=torch.float64, device=dev, requires_grad=True)

    # species_tax_* families are FROZEN at their L1 estimates during refit.
    # The refit only optimizes non-interaction families (tax_global,
    # species_global, batch, donor) plus alpha/log_theta. This prevents the
    # unpenalized refit from inflating falsely-selected interaction
    # coefficients. The L1 estimates are conservative (biased toward zero)
    # but correctly ranked, which is appropriate for selection/screening.
    stx_families = set(species_tax_node_groups.keys()) if species_tax_node_groups else set()
    betas: dict[str, torch.Tensor] = {}
    frozen_betas: dict[str, torch.Tensor] = {}
    masks_t: dict[str, torch.Tensor] = {}
    for family, b in init_betas.items():
        mask = nonzero_masks[family]
        masks_t[family] = torch.tensor(mask, dtype=torch.bool, device=dev)
        init_val = torch.tensor(b, dtype=torch.float64, device=dev)
        init_val[~masks_t[family]] = 0.0
        if family in stx_families:
            frozen_betas[family] = init_val.clone()
        else:
            betas[family] = init_val.clone().requires_grad_(True)

    # Refit gamma on its support (nonzero entries only)
    gamma: torch.Tensor | None = None
    gamma_mask: torch.Tensor | None = None
    if gamma_init is not None:
        gamma_mask = torch.tensor(
            np.abs(gamma_init) > 0.01, dtype=torch.bool, device=dev
        )
        g_init = torch.tensor(gamma_init, dtype=torch.float64, device=dev)
        g_init[~gamma_mask] = 0.0
        gamma = g_init.clone().requires_grad_(True)

    design_tensors = {
        k: torch.tensor(v, dtype=torch.float64, device=dev) for k, v in designs.items()
    }

    # Pre-compute frozen contribution to eta (species_tax families)
    frozen_eta = torch.zeros(n_groups, n_genes, dtype=torch.float64, device=dev)
    for family, fb in frozen_betas.items():
        frozen_eta = frozen_eta + design_tensors[family] @ fb

    all_params = [alpha, log_theta] + list(betas.values())
    if gamma is not None:
        all_params.append(gamma)
    optimizer = torch.optim.Adam(all_params, lr=0.005)

    for _ in range(max_iter):
        optimizer.zero_grad()

        eta = offset.unsqueeze(1) + alpha.unsqueeze(0) + frozen_eta
        for family, X_t in design_tensors.items():
            if family in stx_families:
                continue  # already in frozen_eta
            eta = eta + X_t @ betas[family]
        if gamma is not None:
            eta = eta + gamma

        mu = torch.exp(eta.clamp(max=20.0))
        theta = torch.exp(log_theta.clamp(min=-5.0, max=10.0)).unsqueeze(0)
        loss = _nb_nll(Y, mu, theta)

        loss.backward()
        optimizer.step()

        # Zero out non-selected coefficients
        with torch.no_grad():
            for family, beta in betas.items():
                beta[~masks_t[family]] = 0.0
            if gamma is not None and gamma_mask is not None:
                gamma[~gamma_mask] = 0.0

    result_betas = {}
    result_nonzero = {}
    # Include frozen species_tax families (unchanged L1 estimates)
    for family, fb in frozen_betas.items():
        b = fb.cpu().numpy()
        result_betas[family] = b
        result_nonzero[family] = np.abs(b) > 0.001
    for family, beta in betas.items():
        b = beta.detach().cpu().numpy()
        result_betas[family] = b
        result_nonzero[family] = np.abs(b) > 0.001

    gamma_np = gamma.detach().cpu().numpy() if gamma is not None else None
    return (
        result_betas,
        result_nonzero,
        gamma_np,
        alpha.detach().cpu().numpy(),
        log_theta.detach().cpu().numpy(),
    )


def _fit_dispersion_stage(
    Y_chunk: np.ndarray,
    designs: dict[str, np.ndarray],
    mean_betas: dict[str, np.ndarray],
    alpha: np.ndarray,
    log_theta_baseline: np.ndarray,
    library_sizes: np.ndarray,
    disp_designs: dict[str, np.ndarray],
    dispersion_lambda: float,
    disp_offset: np.ndarray | None,
    gamma: np.ndarray | None,
    max_iter: int,
    device: str,
    refit_support: bool,
) -> dict[str, Any]:
    """Two-stage dispersion fit: freeze mean, fit tree-structured log_overdisp.

    Parameterization (per pseudobulk group i, gene g):

        log_overdisp[i,g] = phi0[g] + disp_offset[i]
                          + sum_family A_disp[family][i,:] @ phi[family][:,g]
        theta[i,g]        = exp(-log_overdisp[i,g])

    Positive phi coefficient => more variable than baseline. phi0 absorbs the
    gene-wise global dispersion (initialized to the frozen baseline).
    """
    dev = torch.device(device)
    n_groups, n_genes = Y_chunk.shape

    Y = torch.tensor(Y_chunk, dtype=torch.float64, device=dev)
    offset = torch.tensor(np.log(library_sizes + 1e-8), dtype=torch.float64, device=dev)

    # Frozen mean linear predictor (no grad)
    with torch.no_grad():
        alpha_t = torch.tensor(alpha, dtype=torch.float64, device=dev)
        eta_mean = offset.unsqueeze(1) + alpha_t.unsqueeze(0)
        for family, X in designs.items():
            X_t = torch.tensor(X, dtype=torch.float64, device=dev)
            b_t = torch.tensor(mean_betas[family], dtype=torch.float64, device=dev)
            eta_mean = eta_mean + X_t @ b_t
        if gamma is not None:
            eta_mean = eta_mean + torch.tensor(gamma, dtype=torch.float64, device=dev)
        mu_frozen = torch.exp(eta_mean.clamp(max=20.0))

    disp_offset_t = (
        torch.tensor(disp_offset, dtype=torch.float64, device=dev)
        if disp_offset is not None
        else torch.zeros(n_groups, dtype=torch.float64, device=dev)
    )

    # phi0: gene-wise dispersion baseline, init from frozen log_theta_baseline.
    # log_overdisp = -log_theta, so phi0_init = -log_theta_baseline.
    phi0 = torch.tensor(
        -log_theta_baseline, dtype=torch.float64, device=dev
    ).clone().requires_grad_(True)

    disp_betas: dict[str, torch.Tensor] = {}
    for family, X in disp_designs.items():
        disp_betas[family] = torch.zeros(
            X.shape[1], n_genes, dtype=torch.float64, device=dev, requires_grad=True
        )

    disp_tensors = {
        k: torch.tensor(v, dtype=torch.float64, device=dev)
        for k, v in disp_designs.items()
    }

    all_params = [phi0] + list(disp_betas.values())
    optimizer = torch.optim.Adam(all_params, lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5
    )

    # L1 column scaling for dispersion: pure column L2 norm sqrt(sum_i x_ij^2)
    # rather than Fisher-weighted. The dispersion Fisher weight collapses
    # toward zero in the Poisson limit (theta -> inf), making the penalty
    # degenerate exactly where regularization matters most. For 0/1 path
    # indicators, the L2 norm equals sqrt(n_groups_under_node), which is the
    # natural and stable scale.

    best_loss = float("inf")
    patience_counter = 0

    for _ in range(max_iter):
        optimizer.zero_grad()

        log_overdisp = phi0.unsqueeze(0).expand(n_groups, n_genes).clone()
        log_overdisp = log_overdisp + disp_offset_t.unsqueeze(1)
        for family, X_t in disp_tensors.items():
            log_overdisp = log_overdisp + X_t @ disp_betas[family]

        log_overdisp = log_overdisp.clamp(min=-10.0, max=10.0)
        theta = torch.exp(-log_overdisp)
        loss = _nb_nll(Y, mu_frozen, theta)

        # L1 with unweighted column L2 norm (see comment above).
        for family, beta in disp_betas.items():
            X_t = disp_tensors[family]
            c_j = torch.sqrt((X_t**2).sum(dim=0) + 1e-10)
            p_family = X_t.shape[1]
            if p_family > 1:
                lam_scaled = dispersion_lambda * math.sqrt(2 * math.log(p_family))
            else:
                lam_scaled = dispersion_lambda
            penalty = lam_scaled * (c_j.unsqueeze(1) * _smooth_l1(beta)).sum()
            loss = loss + penalty

        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 5.0)
        optimizer.step()
        scheduler.step(loss.item())

        current_loss = loss.item()
        if current_loss < best_loss - 1e-4:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > 50:
                break

    # Extract penalized estimates
    threshold = 0.01
    disp_coef = {f: b.detach().cpu().numpy() for f, b in disp_betas.items()}
    disp_nonzero = {f: np.abs(b) > threshold for f, b in disp_coef.items()}

    # Optional weak-ridge post-selection refit
    if refit_support:
        disp_coef, disp_nonzero, phi0_np = _refit_dispersion_support(
            Y_chunk=Y_chunk,
            mu_frozen=mu_frozen.detach().cpu().numpy(),
            disp_designs=disp_designs,
            init_disp=disp_coef,
            nonzero_masks=disp_nonzero,
            phi0_init=phi0.detach().cpu().numpy(),
            disp_offset=disp_offset,
            max_iter=max(50, max_iter // 4),
            device=device,
            ridge=1e-3,
        )
    else:
        phi0_np = phi0.detach().cpu().numpy()

    return {
        "coefficients": disp_coef,
        "nonzero": disp_nonzero,
        "log_theta_baseline": -phi0_np,  # store as log_theta for backward compat
        "loss": best_loss,
    }


def _refit_dispersion_support(
    Y_chunk: np.ndarray,
    mu_frozen: np.ndarray,
    disp_designs: dict[str, np.ndarray],
    init_disp: dict[str, np.ndarray],
    nonzero_masks: dict[str, np.ndarray],
    phi0_init: np.ndarray,
    disp_offset: np.ndarray | None,
    max_iter: int,
    device: str,
    ridge: float = 1e-3,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Unpenalized + weak ridge refit of dispersion params on selected support."""
    dev = torch.device(device)
    n_groups, n_genes = Y_chunk.shape
    Y = torch.tensor(Y_chunk, dtype=torch.float64, device=dev)
    mu_t = torch.tensor(mu_frozen, dtype=torch.float64, device=dev)
    disp_offset_t = (
        torch.tensor(disp_offset, dtype=torch.float64, device=dev)
        if disp_offset is not None
        else torch.zeros(n_groups, dtype=torch.float64, device=dev)
    )

    phi0 = torch.tensor(phi0_init, dtype=torch.float64, device=dev).clone().requires_grad_(True)

    disp_betas: dict[str, torch.Tensor] = {}
    masks_t: dict[str, torch.Tensor] = {}
    for family, b in init_disp.items():
        mask = nonzero_masks[family]
        masks_t[family] = torch.tensor(mask, dtype=torch.bool, device=dev)
        init_val = torch.tensor(b, dtype=torch.float64, device=dev)
        init_val[~masks_t[family]] = 0.0
        disp_betas[family] = init_val.clone().requires_grad_(True)

    disp_tensors = {
        k: torch.tensor(v, dtype=torch.float64, device=dev)
        for k, v in disp_designs.items()
    }

    all_params = [phi0] + list(disp_betas.values())
    optimizer = torch.optim.Adam(all_params, lr=0.005)

    for _ in range(max_iter):
        optimizer.zero_grad()
        log_overdisp = phi0.unsqueeze(0).expand(n_groups, n_genes).clone()
        log_overdisp = log_overdisp + disp_offset_t.unsqueeze(1)
        for family, X_t in disp_tensors.items():
            log_overdisp = log_overdisp + X_t @ disp_betas[family]
        log_overdisp = log_overdisp.clamp(min=-10.0, max=10.0)
        theta = torch.exp(-log_overdisp)
        loss = _nb_nll(Y, mu_t, theta)
        # Weak ridge stabilizer on active dispersion coefficients
        for family, beta in disp_betas.items():
            loss = loss + ridge * (beta**2).sum()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for family, beta in disp_betas.items():
                beta[~masks_t[family]] = 0.0

    out_coef = {f: b.detach().cpu().numpy() for f, b in disp_betas.items()}
    out_nonzero = {f: np.abs(b) > 0.001 for f, b in out_coef.items()}
    return out_coef, out_nonzero, phi0.detach().cpu().numpy()


def fit_tree_nb(
    adata: Any,
    taxonomy_cols: list[str],
    species_col: str,
    species_tree: str | tuple,
    batch_col: str | None = None,
    donor_col: str | None = None,
    counts_layer: str | None = None,
    library_size_col: str | None = None,
    gene_chunk_size: int = 512,
    min_cells_per_pseudobulk: int = 10,
    global_lambda: float | None = None,
    l1_lambdas: dict[str, float] | None = None,
    residual_lambda: float | None = None,
    max_iter: int = 500,
    device: str = "cpu",
    refit_support: bool = True,
    fit_dispersion_tree: bool = False,
    dispersion_families: list[str] | None = None,
    dispersion_lambda: float | None = None,
    min_replicates_per_node: int = 3,
    min_groups_per_disp_node: int = 2,
    dispersion_cell_offset: bool = True,
    progress: bool | str = True,
    keep_design_artifacts: bool = True,
    orthogonal_tree: bool = True,
    empirical_bayes: EmpiricalBayesConfig | None = None,
    refit_all_species_tax: bool = False,
) -> TreeNBResult:
    """Fit tree-structured NB regression on pseudobulk counts.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with raw counts in X or specified layer.
    taxonomy_cols : list[str]
        Ordered list of obs columns defining the taxonomy hierarchy (root to leaf).
    species_col : str
        Obs column with species labels.
    species_tree : str or tuple
        Newick string or nested tuple defining the species phylogeny.
    batch_col : str, optional
        Obs column for batch covariate.
    donor_col : str, optional
        Obs column for donor/sample covariate.
    counts_layer : str, optional
        Layer name for counts. Default uses adata.X.
    library_size_col : str, optional
        Obs column containing each cell's full-transcriptome library size.
        Use this when fitting a gene-subset AnnData so the NB offset is not
        recomputed from the restricted panel.
    gene_chunk_size : int
        Number of genes to process per chunk.
    min_cells_per_pseudobulk : int
        Minimum cells required per pseudobulk group.
    global_lambda : float, optional
        Single global L1 penalty strength applied uniformly to all design
        families. The Fisher-weighted column scaling and sqrt(2*log(p))
        family-size correction provide adaptive regularization, so a uniform
        lambda naturally prefers parsimonious (shallower) explanations.
        If provided, overrides l1_lambdas. Defaults to the calibrated
        ``DEFAULT_GLOBAL_LAMBDA = 0.05``.
    l1_lambdas : dict, optional
        Per-family penalty strengths (legacy interface). Ignored if
        global_lambda is provided.
    residual_lambda : float, optional
        Penalty strength for per-group residual intercepts. When provided,
        the model adds a gamma_ig term for each pseudobulk group i and gene g,
        penalized with L1 to stay sparse. This allows the model to fit
        variance beyond the tree design's rank while keeping the tree
        coefficients interpretable. Groups that need large residuals indicate
        the tree structure doesn't fully capture their expression pattern.
    max_iter : int
        Maximum optimization iterations.
    device : str
        PyTorch device.
    refit_support : bool
        Whether to refit on selected support after L1 (unpenalized MLE).
    fit_dispersion_tree : bool, default False
        Enable a second-stage, tree-structured dispersion fit. Each pseudobulk
        group's log-overdispersion (= -log theta) gets a baseline plus tree
        deviations on the specified families. Mean coefficients (after
        post-selection refit) are frozen during this stage. Positive
        dispersion coefficient => more variable across replicates than the
        gene baseline; negative => more consistent.
    dispersion_families : list[str], optional
        Design families to use for the dispersion tree. Defaults to
        ("tax_global", "species_global"). species_tax_<level> interactions
        are intentionally excluded by default — they explode the parameter
        count and are usually under-replicated for dispersion estimation.
    dispersion_lambda : float, optional
        L1 strength for the dispersion stage. Independent of global_lambda
        because dispersion information per coefficient is fundamentally
        smaller than mean information. Defaults to DEFAULT_DISPERSION_LAMBDA.
    min_replicates_per_node : int, default 3
        Mask out any dispersion-design column whose loading groups span fewer
        than this many distinct donor labels (or batch labels if donor_col is
        None). Prevents L1 from "selecting" coefficients that are not
        identifiable from the available replicate structure.
    min_groups_per_disp_node : int, default 2
        Additional mask: drop dispersion columns supported by fewer than this
        many pseudobulk groups.
    dispersion_cell_offset : bool, default True
        Add `-log(n_cells)` as a known offset on log_overdisp. Justification:
        a pseudobulk sum of n iid NB(mu, theta_cell) is NB(n*mu, n*theta_cell),
        so the pseudobulk dispersion scales trivially with cell count. The
        offset removes this size confound from the tree coefficients.
    progress : bool or str, default True
        If truthy, show a tqdm progress bar over the gene chunks (one tick per
        gene). Pass ``"notebook"`` to force the Jupyter widget variant;
        otherwise ``tqdm.auto`` is used. Set to ``False`` to disable.
    keep_design_artifacts : bool, default True
        If True, attach ``library_sizes``, ``log_theta_baseline``,
        ``disp_offset``, ``designs`` and ``dispersion_designs`` to the
        returned :class:`TreeNBResult`. These are required by
        :func:`tree_nb_regression.inference.compute_wald_significance` to
        compute dispersion-aware standard errors / p-values on the L1-selected
        coefficient support. Set False to save memory if you only need the
        coefficients themselves.
    orthogonal_tree : bool, default True
        Residualize each taxonomy and species-interaction level against all
        earlier levels and restore column norms. This calibrated basis makes
        fitted level contributions comparable. Pass False only to reproduce
        legacy non-orthogonal fits.
    empirical_bayes : EmpiricalBayesConfig, optional
        Learn one Gaussian or Laplace prior scale per species-taxonomy level
        from an evenly spaced pilot-gene panel, then hold those scales fixed
        for the complete fit. Enabling this also enables orthogonal taxonomy
        and interaction blocks so learned level burdens are comparable.
    refit_all_species_tax : bool, default False
        After ordinary L1 support selection, refit every orthogonal
        ``species_tax_*`` coefficient without an L1 penalty. This relaxed
        estimator is intended for donor-bootstrap evolutionary-burden
        inference; selected-support metadata remains the original L1 support.

    Returns
    -------
    TreeNBResult
    """
    if fit_dispersion_tree and residual_lambda is not None:
        warnings.warn(
            "fit_dispersion_tree=True together with residual_lambda is "
            "discouraged: per-group residual intercepts and tree dispersion "
            "both explain unmodelled variance and can compete during fitting. "
            "Consider disabling one."
        )
    if empirical_bayes is not None and not orthogonal_tree:
        warnings.warn(
            "empirical_bayes requires comparable taxonomy increments; "
            "enabling orthogonal_tree=True."
        )
        orthogonal_tree = True
    # Determine penalty: global_lambda takes precedence
    penalty: float | dict[str, float]
    if global_lambda is not None:
        penalty = global_lambda
    elif l1_lambdas is not None:
        penalty = dict(DEFAULT_LAMBDAS)
        penalty.update(l1_lambdas)
    else:
        penalty = DEFAULT_GLOBAL_LAMBDA

    # Build taxonomy tree
    tax_tree = build_taxonomy_tree_from_obs(adata.obs, taxonomy_cols)

    # Build species design
    observed_species = sorted(adata.obs[species_col].unique().tolist())
    sp_design = build_species_tree_design(species_tree, observed_species)

    # Build pseudobulk
    leaf_col = taxonomy_cols[-1]
    pb = build_pseudobulk(
        adata.obs,
        taxonomy_col=leaf_col,
        species_col=species_col,
        batch_col=batch_col,
        donor_col=donor_col,
        min_cells_per_pseudobulk=min_cells_per_pseudobulk,
    )

    # Build design matrices (also returns species_tax sum-to-zero metadata
    # and per-node column groups used by the quadratic anchor).
    designs, species_tax_meta, species_tax_node_groups = _build_design_matrices(
        pb, tax_tree, sp_design, taxonomy_cols, species_col, batch_col, donor_col,
        orthogonal_tree=orthogonal_tree,
    )

    # Resolve penalty to pass to chunk fitting
    if isinstance(penalty, dict):
        active_lambdas: dict[str, float] | float = {
            family: penalty.get(family, 0.1) for family in designs
        }
    else:
        active_lambdas = penalty

    if library_size_col is not None and library_size_col not in adata.obs:
        raise KeyError(f"library-size column '{library_size_col}' was not found in obs.")

    # Compute library sizes from the fitted matrix unless an external
    # full-transcriptome exposure was supplied for a gene-subset fit.
    n_genes = adata.shape[1]
    gene_names = list(adata.var_names)

    # Get count matrix
    if counts_layer is not None:
        X_full = adata.layers[counts_layer]
    else:
        X_full = adata.X

    # Compute library sizes per pseudobulk group
    P = pb.cell_to_group
    if library_size_col is not None:
        cell_totals = np.asarray(adata.obs[library_size_col], dtype=np.float64)
        if not np.isfinite(cell_totals).all() or (cell_totals < 0).any():
            raise ValueError("library_size_col must contain finite nonnegative values.")
    elif sparse.issparse(X_full):
        cell_totals = np.asarray(X_full.sum(axis=1)).ravel()
    else:
        cell_totals = X_full.sum(axis=1)
    library_sizes = np.asarray((P.T @ cell_totals)).ravel()
    pb.library_sizes = library_sizes

    learned_eb_scales: dict[str, float] | None = None
    learned_eb_inclusion: dict[str, float] | None = None
    if empirical_bayes is not None:
        n_pilot = min(empirical_bayes.pilot_genes, n_genes)
        pilot_indices = np.unique(
            np.linspace(0, n_genes - 1, num=n_pilot, dtype=np.int64)
        )
        if sparse.issparse(X_full):
            X_pilot = X_full[:, pilot_indices]
        else:
            X_pilot = np.asarray(X_full)[:, pilot_indices]
        Y_pilot = aggregate_chunk(X_pilot, pb.cell_to_group)
        pilot_result = _fit_gene_chunk(
            Y_pilot,
            designs,
            library_sizes,
            active_lambdas,
            empirical_bayes.pilot_max_iter,
            device,
            refit_support=False,
            residual_lambda=residual_lambda,
            species_tax_node_groups=species_tax_node_groups,
            empirical_bayes=empirical_bayes,
            learn_empirical_bayes_scales=True,
            refit_all_species_tax=refit_all_species_tax,
        )
        learned_eb_scales = cast(
            dict[str, float], pilot_result["empirical_bayes_scales"]
        )
        learned_eb_inclusion = cast(
            dict[str, float], pilot_result["empirical_bayes_inclusion"]
        )

    # Build dispersion designs (subset of mean designs, with masking).
    disp_designs: dict[str, np.ndarray] = {}
    disp_active_indices: dict[str, np.ndarray] = {}
    disp_offset: np.ndarray | None = None
    if fit_dispersion_tree:
        fam_list = (
            list(dispersion_families)
            if dispersion_families is not None
            else list(DEFAULT_DISPERSION_FAMILIES)
        )
        disp_designs, disp_active_indices = _build_dispersion_designs(
            designs,
            pb.group_meta,
            fam_list,
            donor_col,
            batch_col,
            min_replicates_per_node=min_replicates_per_node,
            min_groups_per_node=min_groups_per_disp_node,
        )
        if dispersion_cell_offset and "n_cells" in pb.group_meta.columns:
            disp_offset = -np.log(pb.group_meta["n_cells"].astype(float).values + 1e-8)
        disp_lambda = (
            dispersion_lambda
            if dispersion_lambda is not None
            else DEFAULT_DISPERSION_LAMBDA
        )
    else:
        disp_lambda = None

    # Process genes in chunks
    all_coefficients: dict[str, list[np.ndarray]] = {f: [] for f in designs}
    all_nonzero: dict[str, list[np.ndarray]] = {f: [] for f in designs}
    all_disp_coef: dict[str, list[np.ndarray]] = {f: [] for f in disp_designs}
    all_disp_nonzero: dict[str, list[np.ndarray]] = {f: [] for f in disp_designs}
    all_alphas = []
    all_log_thetas = []
    all_gammas = []
    total_loss = 0.0

    n_chunks = (n_genes + gene_chunk_size - 1) // gene_chunk_size

    chunk_iter = range(n_chunks)
    pbar = None
    if progress:
        try:
            if progress == "notebook":
                from tqdm.notebook import tqdm as _tqdm
            else:
                from tqdm.auto import tqdm as _tqdm
            pbar = _tqdm(
                total=n_genes,
                desc=f"fit_tree_nb (genes, {n_chunks} chunks of {gene_chunk_size})",
                unit="gene",
                smoothing=0.1,
                dynamic_ncols=True,
            )
        except ImportError:
            warnings.warn("tqdm not installed; progress bar disabled.")
            pbar = None

    for chunk_idx in chunk_iter:
        start = chunk_idx * gene_chunk_size
        end = min(start + gene_chunk_size, n_genes)

        # Read chunk
        if sparse.issparse(X_full):
            X_chunk = X_full[:, start:end]
            if not sparse.issparse(X_chunk):
                X_chunk = sparse.csc_matrix(X_chunk)
        else:
            X_chunk = X_full[:, start:end]

        # Aggregate
        Y_chunk = aggregate_chunk(X_chunk, pb.cell_to_group)

        # Fit
        chunk_result = _fit_gene_chunk(
            Y_chunk,
            designs,
            library_sizes,
            active_lambdas,
            max_iter,
            device,
            refit_support=refit_support,
            residual_lambda=residual_lambda,
            disp_designs=disp_designs if fit_dispersion_tree else None,
            dispersion_lambda=disp_lambda,
            disp_offset=disp_offset,
            species_tax_node_groups=species_tax_node_groups,
            empirical_bayes=empirical_bayes,
            empirical_bayes_scales=learned_eb_scales,
            empirical_bayes_inclusion=learned_eb_inclusion,
            refit_all_species_tax=refit_all_species_tax,
        )

        for family in designs:
            all_coefficients[family].append(chunk_result["coefficients"][family])
            all_nonzero[family].append(chunk_result["nonzero"][family])
        for family in disp_designs:
            all_disp_coef[family].append(chunk_result["disp_coefficients"][family])
            all_disp_nonzero[family].append(chunk_result["disp_nonzero"][family])
        all_alphas.append(chunk_result["alpha"])
        all_log_thetas.append(chunk_result["log_theta"])
        if "gamma" in chunk_result:
            all_gammas.append(chunk_result["gamma"])
        total_loss += chunk_result["loss"]

        if pbar is not None:
            pbar.update(end - start)
            postfix = {"loss/gene": f"{total_loss / end:.3g}"}
            pbar.set_postfix(postfix, refresh=False)

    if pbar is not None:
        pbar.close()

    # Concatenate results across chunks
    final_coefficients = {f: np.concatenate(v, axis=1) for f, v in all_coefficients.items()}
    final_nonzero = {f: np.concatenate(v, axis=1) for f, v in all_nonzero.items()}
    final_gamma = np.concatenate(all_gammas, axis=1) if all_gammas else None
    final_log_theta = np.concatenate(all_log_thetas, axis=0) if all_log_thetas else None
    final_alpha = np.concatenate(all_alphas, axis=0) if all_alphas else None

    final_disp_coef: dict[str, np.ndarray] | None = None
    final_disp_nonzero: dict[str, np.ndarray] | None = None
    if disp_designs:
        final_disp_coef = {f: np.concatenate(v, axis=1) for f, v in all_disp_coef.items()}
        final_disp_nonzero = {f: np.concatenate(v, axis=1) for f, v in all_disp_nonzero.items()}

    # Build coefficient metadata. For species_tax_<level> families, enrich
    # each row with (level, node_id, node_label, species) from the per-column
    # metadata built during _build_design_matrices so downstream consumers
    # (inference, reporting) never have to back-compute it.
    coef_meta_rows = []
    for family, X in designs.items():
        n_cols = X.shape[1]
        st_meta = species_tax_meta.get(family)
        st_lookup = (
            {int(r["col_index"]): r for r in st_meta.to_dict("records")}
            if st_meta is not None else {}
        )
        for j in range(n_cols):
            row: dict[str, Any] = {
                "coef_id": f"{family}_{j}",
                "family": family,
                "index": j,
                "n_nonzero_genes": int(final_nonzero[family][j].sum()),
            }
            if j in st_lookup:
                meta = st_lookup[j]
                row.update({
                    "level": meta["level"],
                    "node_id": meta["node_id"],
                    "node_label": meta["node_label"],
                    "species": meta["species"],
                    "n_species_at_node": meta["n_species_at_node"],
                })
            coef_meta_rows.append(row)
    coef_metadata = pd.DataFrame(coef_meta_rows)

    # Dispersion coefficient metadata — map back to original tree node ids
    disp_coef_metadata: pd.DataFrame | None = None
    if final_disp_coef is not None:
        assert final_disp_nonzero is not None
        disp_rows = []
        for family, coefs in final_disp_coef.items():
            active = disp_active_indices[family]
            if family == "tax_global":
                node_ids = list(tax_tree.node_ids)
            elif family == "species_global":
                node_ids = list(sp_design.node_ids)
            else:
                node_ids = [f"{family}_{j}" for j in range(designs[family].shape[1])]
            for k, orig_j in enumerate(active):
                disp_rows.append({
                    "coef_id": f"disp_{family}_{orig_j}",
                    "family": family,
                    "index": int(orig_j),
                    "node_id": node_ids[orig_j] if orig_j < len(node_ids) else None,
                    "n_nonzero_genes": int(final_disp_nonzero[family][k].sum()),
                })
        disp_coef_metadata = pd.DataFrame(disp_rows)

    # Diagnostics
    diagnostics = {
        "total_nll": total_loss,
        "n_groups": pb.n_groups,
        "n_genes": n_genes,
        "n_chunks": n_chunks,
        "nonzero_per_family": {
            f: int(m.sum()) for f, m in final_nonzero.items()
        },
    }
    if final_gamma is not None:
        n_nonzero_gamma = int((np.abs(final_gamma) > 0.01).sum())
        n_total_gamma = final_gamma.size
        diagnostics["gamma_nonzero"] = n_nonzero_gamma
        diagnostics["gamma_total"] = n_total_gamma
        diagnostics["gamma_sparsity_pct"] = 100 * (1 - n_nonzero_gamma / n_total_gamma)
    if final_disp_coef is not None:
        assert final_disp_nonzero is not None
        diagnostics["dispersion_nonzero_per_family"] = {
            f: int(m.sum()) for f, m in final_disp_nonzero.items()
        }
        diagnostics["dispersion_active_cols_per_family"] = {
            f: int(len(idx)) for f, idx in disp_active_indices.items()
        }
    if empirical_bayes is not None:
        diagnostics["shrinkage_prior"] = empirical_bayes.prior.value
        diagnostics["shrinkage_scales"] = learned_eb_scales
        diagnostics["empirical_bayes_pilot_genes"] = min(
            empirical_bayes.pilot_genes, n_genes
        )
        if empirical_bayes.prior is ShrinkagePrior.SPIKE_SLAB:
            diagnostics["shrinkage_inclusion"] = learned_eb_inclusion
            diagnostics["spike_scale"] = empirical_bayes.spike_scale
    diagnostics["refit_all_species_tax"] = refit_all_species_tax

    return TreeNBResult(
        coefficients=final_coefficients,
        coef_metadata=coef_metadata,
        selected_nonzero=final_nonzero,
        group_meta=pb.group_meta,
        taxonomy_node_table=tax_tree.node_table,
        species_node_table=sp_design.node_table,
        gene_names=gene_names,
        diagnostics=diagnostics,
        gamma=final_gamma,
        intercept=final_alpha if keep_design_artifacts else None,
        dispersion_coefficients=final_disp_coef,
        dispersion_selected_nonzero=final_disp_nonzero,
        dispersion_coef_metadata=disp_coef_metadata,
        dispersion_active_indices=disp_active_indices or None,
        library_sizes=library_sizes if keep_design_artifacts else None,
        log_theta_baseline=final_log_theta if keep_design_artifacts else None,
        disp_offset=disp_offset if keep_design_artifacts else None,
        designs=(designs if keep_design_artifacts else None),
        dispersion_designs=(disp_designs if (keep_design_artifacts and disp_designs) else None),
        species_tax_meta=(species_tax_meta or None),
        species_tax_node_groups=(species_tax_node_groups or None),
        shrinkage_prior=(
            empirical_bayes.prior.value if empirical_bayes is not None else None
        ),
        shrinkage_scales=learned_eb_scales,
        shrinkage_inclusion=(
            learned_eb_inclusion
            if empirical_bayes is not None
            and empirical_bayes.prior is ShrinkagePrior.SPIKE_SLAB
            else None
        ),
        orthogonal_tree=orthogonal_tree,
    )
