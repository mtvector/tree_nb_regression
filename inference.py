"""Dispersion-informed Wald inference for tree-NB regression coefficients.

Given a fitted :class:`TreeNBResult` that retains its design artifacts
(``library_sizes``, ``intercept``, ``log_theta_baseline``, ``disp_offset``,
``designs``, ``dispersion_designs`` -- all True by default with
``keep_design_artifacts=True``), :func:`compute_wald_significance` returns a
long DataFrame of per-coefficient standard errors, z-scores, two-sided
p-values and 95% confidence intervals on the L1-selected support.

Statistical caveats
-------------------
These are **post-selection refit Wald** statistics. The selection event
(L1 picking the support S) is ignored. In practice:

* Coefficients deep inside the support (much larger than the L1 threshold)
  are well calibrated.
* Coefficients that *just barely* survived L1 have SE under-estimated and p
  over-stated. Treat small p-values near the selection boundary as
  *screening* rather than confirmatory.
* No multiple-testing correction is applied to the coefficient axis other
  than the optional Benjamini-Hochberg q-value per (family, gene-axis) and
  per (family, contrast-axis) -- see ``add_bh_qvalues``.

What the dispersion fit buys you
--------------------------------
The NB Fisher weight at pseudobulk group i, gene j is

    W_ij = mu_ij / (1 + mu_ij / theta_ij)

so the effective sample size at a clade depends critically on theta. With
the tree-structured dispersion fit, clades with elevated donor-level
variability (high log_overdisp) get a smaller W, larger SE, and therefore
more honest p-values. Without it we would use a single per-gene theta and
would understate uncertainty in high-variability clades and overstate it
in low-variability clades.
"""

from __future__ import annotations

import warnings
from typing import Any, Iterable, cast

import numpy as np
import pandas as pd
from scipy.special import ndtr  # standard-normal CDF

from .results import Array, TreeNBResult

__all__ = ["compute_wald_significance", "add_bh_qvalues"]


def _ensure_artifacts(res: TreeNBResult) -> None:
    missing = [
        name
        for name in (
            "library_sizes",
            "log_theta_baseline",
            "designs",
            "intercept",
        )
        if getattr(res, name) is None
    ]
    if missing:
        raise RuntimeError(
            "TreeNBResult is missing fit artifacts required for Wald "
            f"inference: {missing}. Re-run fit_tree_nb with "
            "keep_design_artifacts=True."
        )


def _expand_dispersion_coefs(
    res: TreeNBResult,
    family: str,
    n_groups: int,
) -> Array | None:
    """Return the contribution X_disp_f @ disp_beta_f to log_overdisp,
    expanded to shape (n_groups, n_genes) for the active columns of family.
    Returns None if there is no dispersion fit for ``family``.
    """
    if res.dispersion_coefficients is None or family not in res.dispersion_coefficients:
        return None
    if res.dispersion_designs is None or family not in res.dispersion_designs:
        return None
    X = res.dispersion_designs[family]              # (n_groups, n_active)
    B = res.dispersion_coefficients[family]         # (n_active, n_genes)
    if X.shape[1] == 0 or B.shape[0] == 0:
        return None
    return X @ B


def _compute_mu(res: TreeNBResult) -> Array:
    """Compute the fitted per-(group, gene) NB mean mu = exp(eta).

    eta_ig = log(L_i) + alpha_g + sum_f (X_f @ beta_f)[i, g] + gamma_ig
    """
    L = res.library_sizes
    intercept = res.intercept
    designs = res.designs
    assert L is not None
    assert intercept is not None
    assert designs is not None
    n_groups = L.shape[0]
    n_genes = res.n_genes
    eta = np.log(L + 1e-8)[:, None] + np.broadcast_to(
        intercept[None, :], (n_groups, n_genes)
    ).copy()
    for family, X in designs.items():
        B = res.coefficients[family]                # (n_cols, n_genes)
        if X.shape[1] != B.shape[0]:
            raise RuntimeError(
                f"Design / coefficient shape mismatch for family '{family}': "
                f"X has {X.shape[1]} cols but B has {B.shape[0]} rows."
            )
        eta += X @ B
    if res.gamma is not None:
        eta += res.gamma
    np.clip(eta, a_min=None, a_max=20.0, out=eta)
    return cast(Array, np.exp(eta))


def _compute_theta(res: TreeNBResult) -> Array:
    """Compute the fitted per-(group, gene) NB dispersion theta.

    log_overdisp_ig = phi0_g + disp_offset_i + sum_f (X_disp_f @ disp_beta_f)[i, g]
    theta_ig        = exp(-log_overdisp_ig)

    phi0_g = -log_theta_baseline[g] (per the model parameterization).
    """
    library_sizes = res.library_sizes
    log_theta_baseline = res.log_theta_baseline
    assert library_sizes is not None
    assert log_theta_baseline is not None
    n_groups = library_sizes.shape[0]
    n_genes = res.n_genes
    phi0 = -np.asarray(log_theta_baseline)      # (n_genes,)
    log_od = np.broadcast_to(phi0[None, :], (n_groups, n_genes)).copy()
    if res.disp_offset is not None:
        log_od += np.asarray(res.disp_offset)[:, None]
    if res.dispersion_coefficients is not None and res.dispersion_designs is not None:
        for family in res.dispersion_coefficients:
            contrib = _expand_dispersion_coefs(res, family, n_groups)
            if contrib is not None:
                log_od += contrib
    np.clip(log_od, a_min=-10.0, a_max=10.0, out=log_od)
    return cast(Array, np.exp(-log_od))


def _family_node_label(
    res: TreeNBResult, family: str, index: int
) -> tuple[str | None, str | None, str | None]:
    """Map a (family, index) coef back to (sp_contrast, level, node_label).

    Returns (sp_contrast, level, node_label) with None where not applicable.
    For ``species_tax_*`` families the mapping is read from
    ``res.species_tax_meta`` populated at fit time.
    """
    if family == "tax_global":
        tax_nt = res.taxonomy_node_table
        if index < len(tax_nt):
            row = tax_nt.iloc[index]
            return None, row.get("level"), row.get("label", row.get("node_id"))
        return None, None, None
    if family == "species_global":
        sp_nt = res.species_node_table
        if index < len(sp_nt):
            row = sp_nt.iloc[index]
            return row.get("label", row.get("node_id")), None, None
        return None, None, None
    if family.startswith("species_tax_"):
        meta = (
            res.species_tax_meta.get(family)
            if res.species_tax_meta is not None else None
        )
        if meta is not None:
            hit = meta[meta["col_index"] == index]
            if len(hit) > 0:
                r = hit.iloc[0]
                return (
                    r.get("species"),
                    r.get("level"),
                    r.get("node_label"),
                )
        # Fall-through: metadata missing → return level from family name only.
        return None, family[len("species_tax_") :], None
    return None, None, None


def compute_wald_significance(
    res: TreeNBResult,
    families: Iterable[str] | None = None,
    selection_threshold: float = 1e-8,
    ridge: float = 1e-4,
    add_q: bool = True,
    q_within: tuple[str, ...] = ("gene", "family"),
    drop_root_columns: bool = True,
    empirical_null_calibration: bool = True,
) -> pd.DataFrame:
    """Compute dispersion-aware Wald SE / p-values for L1-selected coefs.

    Per-family Hessian, full-family columns, Cholesky-based covariance,
    empirical-null SE calibration. See README §"Wald inference" for the
    statistical model.

    Parameters
    ----------
    res : TreeNBResult
        Must have been fitted with ``keep_design_artifacts=True``.
    families : iterable of str, optional
        Restrict the *returned rows* to these families. The Hessian and
        covariance are still computed over the full family columns;
        ``families`` is just a row filter on the output.
    selection_threshold : float, default 1e-8
        Magnitude floor below which a coefficient is treated as L1-zero.
        Defaults below the model's own selection threshold so the support
        matches whatever was returned in ``res.coefficients``.
    ridge : float, default 1e-4
        Absolute ridge added to the per-gene Hessian for numerical
        stability. Necessary for low-information species_tax columns
        where ``trace(H)/k`` itself can be < 1e-4.
    add_q : bool, default True
        Append Benjamini-Hochberg q-values.
    q_within : tuple of str, default ``("gene", "family")``
        Columns within which to compute BH q-values. The default
        per-(gene, family) cascade is the standard genomics convention:
        each gene's coefs are FDR-adjusted as a unit, isolating null
        genes from polluting the multiple-testing burden of true-effect
        genes. Pass ``("family",)`` for the legacy family-wide BH
        (treats every (gene, coef) pair as one test). Pass ``()`` for
        one global BH adjustment. NOTE: these are post-selection
        screening q-values, NOT confirmatory FDR over the full
        pre-selection coef x gene universe.
    drop_root_columns : bool, default True
        Exclude all-constant columns (the root path-indicator of
        ``tax_global`` / ``species_global``) from the Hessian. These are
        exactly aliased to the per-gene intercept ``alpha``; including
        them inflates every SE.
    empirical_null_calibration : bool, default True
        Per-family Efron-style empirical-null calibration: inflate SE by
        ``IQR(z)/IQR(N(0,1))`` estimated from the central 50% of |z| so
        the empirical null variance is unit. The per-family
        ``se_emp_inflation`` is recorded in the output. Floored at 1.0
        (never deflate).

    Returns
    -------
    DataFrame with columns:
        family, index, coef_id, gene, sp_contrast, level, node_label,
        beta_hat, se, z, p, ci_lo, ci_hi, se_emp_inflation, q

    Rows whose per-gene Hessian fails Cholesky factorization (i.e. the
    SPD assumption breaks even after ridge regularization) are silently
    dropped — these correspond to genes with degenerate fits, and their
    NA p-values aren't useful.

    Statistical caveats
    -------------------
    These are post-selection refit Wald statistics — the selection event
    is ignored. The per-family Hessian is the *profile* information
    matrix (other families' coefficients held at refit values), so SEs
    are anti-conservatively tight when families overlap heavily; the
    empirical-null calibration corrects this on average but not in the
    tails (z-stat heavy tails from gene-to-gene heterogeneity). Trust
    p-values from well-identified families (``tax_global``,
    ``species_global``, ``species_tax_<leaf level>``,
    ``species_tax_Group``) as the primary inferential signal; treat
    inner-tree-level families (``species_tax_Class``,
    ``species_tax_Subclass``) and the donor/batch families as
    *screening* with possibly inflated Type-I unless used with a
    stricter q < 0.01 threshold.
    """
    _ensure_artifacts(res)

    designs = res.designs
    assert designs is not None
    all_mean_families = list(designs.keys())
    if families is not None:
        report_families = set(families)
        for f in report_families:
            if f not in designs:
                raise KeyError(f"Family '{f}' not in fitted designs.")
    else:
        report_families = set(all_mean_families)

    n_genes = res.n_genes

    mu = _compute_mu(res)
    theta = _compute_theta(res)
    W = mu / (1.0 + mu / np.clip(theta, 1e-8, None))

    # ── Identify columns to drop (root / constant columns aliased to alpha) ──
    dropped_cols: dict[str, set[int]] = {f: set() for f in all_mean_families}
    if drop_root_columns:
        for f in ("tax_global", "species_global"):
            if f not in designs:
                continue
            Xf = designs[f]
            for j in range(Xf.shape[1]):
                if np.ptp(Xf[:, j]) < 1e-12:
                    dropped_cols[f].add(j)

    from scipy.linalg import cho_factor, cho_solve

    from .model import SUMZERO_ANCHOR_MU  # local import to avoid cycle

    rows: list[dict[str, Any]] = []
    n_chol_fail = 0

    for family in all_mean_families:
        X = designs[family]                          # (n_groups, n_cols)
        B = res.coefficients[family]                     # (n_cols, n_genes)
        n_cols = X.shape[1]
        if n_cols == 0:
            continue
        family_node_groups: list[list[int]] = []
        if (
            res.species_tax_node_groups
            and family in res.species_tax_node_groups
        ):
            family_node_groups = list(res.species_tax_node_groups[family])

        # Full-family Hessian column set (root cols dropped once per family)
        hess_cols = np.asarray(
            [c for c in range(n_cols) if c not in dropped_cols[family]],
            dtype=np.int64,
        )
        if hess_cols.size == 0:
            continue
        Xh = X[:, hess_cols]
        hess_pos = {int(c): k for k, c in enumerate(hess_cols)}
        # Pre-compute the design-only static contribution to H:
        #   sum-to-zero anchors + ridge·I (added once, then per-gene W contribution).
        k_h = hess_cols.size
        H_static = ridge * np.eye(k_h)
        for group in family_node_groups:
            locals_in = [hess_pos[c] for c in group if c in hess_pos]
            if len(locals_in) >= 2:
                idx = np.asarray(locals_in, dtype=np.int64)
                H_static[np.ix_(idx, idx)] += SUMZERO_ANCHOR_MU

        support_mat = np.abs(B) > selection_threshold

        for g in range(n_genes):
            sup_local = np.flatnonzero(support_mat[:, g])
            sup_local = np.asarray(
                [c for c in sup_local if c in hess_pos], dtype=np.int64
            )
            if sup_local.size == 0:
                continue

            w = W[:, g]
            H_reg = (Xh.T * w) @ Xh + H_static

            try:
                cho = cho_factor(H_reg, lower=False, check_finite=False)
            except np.linalg.LinAlgError:
                n_chol_fail += 1
                continue
            # Inverse only the diagonal we need: solve H_reg X = e_k columns
            # of the selected support, take the k-th entry.
            sel_in_hess = np.asarray(
                [hess_pos[int(c)] for c in sup_local], dtype=np.int64
            )
            E = np.zeros((k_h, sel_in_hess.size))
            E[sel_in_hess, np.arange(sel_in_hess.size)] = 1.0
            Cov_cols = cho_solve(cho, E, check_finite=False)
            diag_C = np.clip(
                Cov_cols[sel_in_hess, np.arange(sel_in_hess.size)],
                a_min=0.0, a_max=None,
            )
            se = np.sqrt(diag_C)
            beta_hat = B[sup_local, g]
            with np.errstate(divide="ignore", invalid="ignore"):
                z = np.where(se > 0, beta_hat / se, np.nan)
            p = 2.0 * ndtr(-np.abs(z))
            ci_lo = beta_hat - 1.959963984540054 * se
            ci_hi = beta_hat + 1.959963984540054 * se
            gene_name = res.gene_names[g]

            if family not in report_families:
                continue
            for k, j_local in enumerate(sup_local):
                sp_c, lvl, node_lbl = _family_node_label(
                    res, family, int(j_local)
                )
                rows.append({
                    "family": family,
                    "index": int(j_local),
                    "coef_id": f"{family}_{int(j_local)}",
                    "gene": gene_name,
                    "sp_contrast": sp_c,
                    "level": lvl,
                    "node_label": node_lbl,
                    "beta_hat": float(beta_hat[k]),
                    "se": float(se[k]),
                    "z": float(z[k]),
                    "p": float(p[k]),
                    "ci_lo": float(ci_lo[k]),
                    "ci_hi": float(ci_hi[k]),
                })

    if n_chol_fail > 0:
        warnings.warn(
            f"{n_chol_fail} gene-family Hessian(s) failed Cholesky "
            "factorization and were dropped from the output."
        )

    if not rows:
        warnings.warn("No L1-selected coefficients found across requested families.")
        return pd.DataFrame(
            columns=[
                "family", "index", "coef_id", "gene", "sp_contrast", "level",
                "node_label", "beta_hat", "se", "z", "p", "ci_lo", "ci_hi",
                "se_emp_inflation", "q",
            ]
        )

    df = pd.DataFrame(rows)

    # ── Empirical null calibration (Efron-style IQR-based SE inflation) ───
    # Per-family conditional Hessian gives the SE *as if* other families'
    # coefs were known; with nested tree designs they're estimated, so
    # the SE is anti-conservatively tight. We rescale by IQR(z)/IQR(N(0,1))
    # on the central 50% of |z| so the empirical null has unit variance.
    if empirical_null_calibration and len(df):
        df["se_emp_inflation"] = 1.0
        for fam, sub in df.groupby("family"):
            z = sub["z"].values
            z = z[np.isfinite(z)]
            if z.size < 30:
                continue
            cutoff = np.quantile(np.abs(z), 0.50)
            central = z[np.abs(z) <= cutoff]
            if central.size < 10:
                continue
            target = 0.6744897501960817  # qnorm(0.75) = IQR/2 of N(0,1)
            empirical_scale = max(1.0, float(cutoff) / target)
            mask = df["family"] == fam
            df.loc[mask, "se"] = df.loc[mask, "se"] * empirical_scale
            df.loc[mask, "se_emp_inflation"] = empirical_scale
            df.loc[mask, "z"] = df.loc[mask, "beta_hat"] / df.loc[mask, "se"]
            df.loc[mask, "ci_lo"] = (
                df.loc[mask, "beta_hat"] - 1.959963984540054 * df.loc[mask, "se"]
            )
            df.loc[mask, "ci_hi"] = (
                df.loc[mask, "beta_hat"] + 1.959963984540054 * df.loc[mask, "se"]
            )
            df.loc[mask, "p"] = 2.0 * ndtr(-np.abs(df.loc[mask, "z"]))

    if add_q:
        df = add_bh_qvalues(df, within=q_within)
    return df



def add_bh_qvalues(
    df: pd.DataFrame,
    p_col: str = "p",
    q_col: str = "q",
    within: tuple[str, ...] = ("family",),
) -> pd.DataFrame:
    """Append Benjamini-Hochberg q-values, computed within ``within`` groups."""
    out = df.copy()
    out[q_col] = np.nan
    if len(within) == 0:
        out[q_col] = _bh(out[p_col].values)
        return out
    for _, sub in out.groupby(list(within), dropna=False):
        out.loc[sub.index, q_col] = _bh(sub[p_col].values)
    return out


def _bh(p: Array) -> Array:
    """Benjamini-Hochberg step-up."""
    p = np.asarray(p, dtype=float)
    n = p.size
    if n == 0:
        return p
    finite = np.isfinite(p)
    out = np.full(n, np.nan)
    if not finite.any():
        return out
    p_finite = p[finite]
    order = np.argsort(p_finite, kind="stable")
    ranked = p_finite[order]
    m = ranked.size
    q = ranked * m / (np.arange(m) + 1)
    # Enforce monotonic non-increasing q in reverse
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, a_min=0.0, a_max=1.0)
    q_full = np.empty(m)
    q_full[order] = q
    out[finite] = q_full
    return out
