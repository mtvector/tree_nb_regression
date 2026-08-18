# Statistical Review: `tree_nb_regression`

> **Historical working document.** This file preserves the original review and
> subsequent simulation iteration log, so some early findings and remediation
> rows describe superseded implementations. The current user-facing behavior,
> defaults, and limitations are summarized in `README.md`. Re-run the simulation
> notebook before treating any recorded calibration number as current evidence.

Scope: review of `model.py`, `inference.py`, `pseudobulk.py`, `taxonomy_tree.py`,
`species_tree.py`, and `README.md` in
`/code/HMBA_Genomics/SpinalCord/xspecies/analysis/tree_nb_regression/`.

The approach is conceptually attractive (tree-path indicators + Fisher-calibrated
weighted L1, optional second-stage dispersion fit, post-selection Wald with
disclosed caveats), but it has several real statistical problems and a few
outright bugs / code-vs-documentation inconsistencies. Below is the review with
a remediation plan.

---

## A. Bugs / code-vs-documentation inconsistencies

### A1. README says "sum-to-zero" but the code uses reference coding for species×taxonomy  ✅ **RESOLVED**

**Status (post-fix):** Implemented true sum-to-zero coding using centered
indicator columns. Each `species_tax_<L>` block now has `K = |S(n)|`
columns per node `n` (one per species observed at node `n`), with values
`+(K-1)/K` for the matching species, `-1/K` for the other species at the
same node, and 0 outside the node. A quadratic anchor
`0.5 * SUMZERO_ANCHOR_MU * (Σ_s β[s,n,g])²` is added to the optimization
loss (and to the Wald Hessian) so the K-column block is identifiable and
fitted coefficients exactly satisfy `Σ_s β[s,n,g] = 0`.

Key behavioral implications:

- L1 selection is now permutation-symmetric across species (verified by
  `test_sumzero_species_permutation_invariance`).
- Per-species coefficients are interpreted as *deviations from the
  cross-species mean at that node*, not as contrasts to a reference.
- For K=2, the per-species |β| is half the magnitude of the corresponding
  reference-coded contrast (the effect is split symmetrically between the
  two species).
- `TreeNBResult` now exposes `species_tax_meta` (DataFrame of
  per-column (level, node, species, K)) and `species_tax_node_groups`
  (per-family list of column-index groups for sum-to-zero) for downstream
  inference and reporting.

See `model.py:_build_design_matrices`, `_sumzero_anchor_penalty`,
`_fit_gene_chunk`, `_refit_support`, and `inference.py:compute_wald_significance`.
README §"Sum-to-zero coding" documents the new design.

**Original review (kept for context):**

`model.py` lines ~104–130 build the `species_tax_<level>` block with
`(n_species − 1) * n_level_nodes` columns and *the reference (alphabetically
last) species contributes zero*. README §"Model" (line 96) calls it
"sum-to-zero: deviation = indicator − mean across species for that node."
These are not equivalent.

Consequences:

- L1 with reference coding is **not equivariant** to the choice of reference;
  reorder/rename the species list and you get different "selected" interactions.
- The Fisher-calibrated column norm `c_j` becomes asymmetric across species:
  the reference species contributes nothing to any deviation column, so its
  species×clade effects get absorbed into `species_global` and `tax_global`
  instead of being penalized symmetrically with the others.
- Downstream effect-size comparisons across species are not directly comparable.

**Fix:** Either (i) update the README to say "reference-coded (last species)",
or (ii) actually implement sum-to-zero (Helmert / `contr.sum`) so coefficients
are species-symmetric. For L1 the sum-to-zero choice is preferable.

### A2. Library-size offset uses raw cell totals, not pseudobulk-summed gene totals

`fit_tree_nb` computes `cell_totals = X_full.sum(axis=1)` over *all* cells
(including filtered cells outside any pseudobulk), then
`library_sizes = P.T @ cell_totals` (model.py lines ~1019–1022). Cells dropped
from pseudobulks via `min_cells_per_pseudobulk` correctly contribute zero
through `P`, so this is OK. But for the **dispersion** offset `−log(n_cells)`,
`n_cells` is taken from `pb.group_meta["n_cells"]` (count of cells), while the
mean offset is a sum of UMIs — these are scale-incompatible if cells differ in
depth. Not wrong, but the two offsets implicitly assume mean depth is constant
across groups.

**Fix:** Document the assumption; optionally also offer
`disp_offset = −log(mean_cell_depth × n_cells)` for groups with very
heterogeneous cell depth.

### A3. `refit_support` zeros masked coefficients after each Adam step but still applies gradients to them

`model.py` lines ~604–609 and ~835–837. This wastes optimization mass and can
perturb active coefficients via the optimizer's momentum/variance state on
masked dims. Mathematically harmless but slow and noisy near convergence.

**Fix:** Parameterize only the active subset (or apply the mask to
`param.grad` before `optimizer.step()` and reset Adam state for that dim each
step).

### A4. Adam for a (mostly) convex GLM problem

The NB log-likelihood with linear predictor is concave in (α, β) given θ. Using
Adam with `ReduceLROnPlateau` and an early-stop on a tiny absolute loss delta
(`1e-4`) regularly returns parameters appreciably off the MLE, especially in
over-parameterized families. The post-selection Wald inference
(`compute_wald_significance`) computes SEs at the **fitted** β, so
under-convergence directly biases SE/p.

**Fix:** Use L-BFGS for the mean and dispersion refit (PyTorch has
`torch.optim.LBFGS`), or IRLS. Keep Adam only for the L1 selection pass if
speed matters, then refit on the support with L-BFGS to true MLE.

---

## B. Major statistical caveats / errors

### B1. Donor pseudoreplication is not handled

Donors are entered as L1-penalized fixed effects
(`l1_lambdas["donor"]=0.05`). With ~3–4 donors per species (typical for HMBA),
this is the dominant source of correlation and *the* limiting factor for
cross-species inference. The model treats each pseudobulk group as an
independent NB observation; that is wrong in two ways:

- (i) Multiple pseudobulks share a donor (e.g., different cell types from the
  same donor are correlated through donor state) → standard errors are too
  small.
- (ii) Species effects are estimated essentially from a between-donor contrast
  with only a few donors per species; the model gives no honest accounting of
  that limited replication. The Wald SE in `inference.py` uses
  `W = μ/(1 + μ/θ)` which assumes independent pseudobulks.

The `inference.py` docstring acknowledges "no donor-clustered sandwich" but it
is not a minor caveat — it is the dominant source of false positives in
cross-species DE.

**Fix:**

- Add a **cluster-robust (Huber–White) sandwich** SE clustered by donor in
  `compute_wald_significance` (cheap to add: replace `H^{-1}` with
  `H^{-1} (Σ_d s_d s_d') H^{-1}` where `s_d` is the donor-summed score). This
  is the most important single change.
- Alternatively, switch the donor term to a genuine random effect
  (PQL/Laplace) — much more work; or use a "donor-as-replicate" approach
  where you pseudobulk to one observation per (cell_type × species × donor)
  and treat donor variance as the residual scale (DESeq2/edgeR/limma-voom on
  pseudobulks is the field-standard remedy).

### B2. L1 shrinkage on the donor term induces omitted-variable bias on species/clade coefficients

Donor is **perfectly nested in species** (each donor belongs to one species),
so any donor variance that L1 shrinks to zero is absorbed by `species_global`
and `species_tax_*`. With `lambda=0.05` and donor columns being one-hots with
low Fisher norm (few cells per donor for some clades), donor coefficients are
often zeroed → species effects are inflated. This is a *systematic* bias, not
random noise.

**Fix:**

- **Do not L1-penalize the donor term.** Treat donor either as unpenalized
  fixed effect (with the consequent rank-deficiency cost) or as a
  ridge-penalized random effect. The Fisher-calibrated weighting argument
  applies only to the structured tree families; nuisance covariates should be
  unpenalized or weakly ridge-penalized.
- Same comment applies to `batch` if batch is confounded with species/donor
  (it usually is).

### B3. Rank-deficiency among `alpha` (intercept), `tax_global` root column, and `species_global` root column

Each tree's root indicator equals the all-ones vector. With an unpenalized
intercept `alpha[g]` and L1 on the root coefficients, the root contribution
gets shrunk to zero arbitrarily, while the constant is absorbed by `alpha`.
With `global_lambda=0`, the design is rank-deficient and the optimizer's
solution becomes Adam-trajectory-dependent. With L1>0 the problem is
well-posed but the coefficient labeled "tax_global / root" is meaningless — it
absorbs whatever the penalty leaves over.

**Fix:** Drop the root column(s) explicitly (they are aliased to the
intercept), and document that "tax_global / root" is not a real effect.

### B4. Cross-species comparability of counts

Library-size offset normalizes for sequencing depth within a sample but not
for cross-species transcriptome composition differences (ortholog length,
% mitochondrial, ribosomal RNA fraction, ambient RNA differences, mapping
rates). The `species_global` term therefore mixes biological species effect
with composition bias.

**Fix:**

- Use a TMM-style or median-of-ratios offset per (species × donor) instead of
  (or in addition to) raw library size.
- Alternatively, restrict the "library size" to a curated panel of housekeeping
  orthologs as the size factor. This is what is done in published cross-species
  DE (e.g., Bakken et al. cross-mammalian).

### B5. The dispersion offset `−log(n_cells)` assumes cells within a pseudobulk are iid NB

The derivation "Σ NB(μ, θ) = NB(nμ, nθ)" is only true for *independent*
draws. Within a pseudobulk (= one cell type × one donor × one batch), cells
share donor state — they are positively correlated, so the true pseudobulk
variance exceeds the iid prediction. The offset under-corrects for cell-count
differences and *systematically inflates* the apparent log-overdispersion of
large pseudobulks.

**Fix:** Either drop the offset and let `tax_global / species_global`
dispersion coefficients absorb the n-cells effect (acknowledging
confounding), or fit a per-group "intracluster correlation" parameter ρ and
use the design-effect-corrected `n_eff = n / (1 + (n−1)ρ)`.

### B6. Post-selection Wald is acknowledged but the q-values are mislabeled as "screening"

`inference.py` does BH within `("gene", "family")` over (coefficient ×
gene) pairs from the **selected** support (default since iter 11). This
is neither valid FDR over the original universe nor a valid selective
FDR. Calling them "screening q-values" is honest but users will still
interpret them as FDR.

**Fix:**

- Offer a *valid* alternative: data-splitting (fit L1 on half, refit + Wald on
  the other half) or knockoffs for FDR over the coefficient axis. Even a
  debiased-lasso step (Javanmard–Montanari / van de Geer) per gene would be a
  real improvement.
- At minimum, the BH q should be computed over the *full pre-selection
  coefficient × gene grid* (treating non-selected coefs as p=1), which is
  conservative but interpretable.

### B7. Fisher-weight penalty calibration uses `W.mean(dim=1)` (averaged across genes)

`model.py` line 447: `W_mean = W.mean(dim=1)`. This computes a single
per-group Fisher weight averaged across all genes in the chunk, then applies
it to per-gene coefficients. Two issues:

- Genes with very different expression levels (very low Fisher info vs. very
  high) are penalized using the chunk-average, not their own scale →
  low-expression genes are under-penalized, high-expression over-penalized.
- The "chunk" boundary makes the penalty scale depend on `gene_chunk_size` —
  fits are not reproducible across chunk sizes.

**Fix:** Compute `c_j` per gene: `c_j[g] = sqrt(Σ_i W[i,g] x_ij^2)`. The cost
is a per-gene scale per column per chunk, which is cheap.

### B8. Universal-threshold `sqrt(2 log p_family)` applied per-family, but families are jointly fit

The L1 universal threshold is a single `sqrt(2 log p_total)` for the
concatenated design under orthogonality; per-family scaling biases toward
families with more columns and is *not* the right correction when families
are correlated (tree-path indicators are highly correlated within a family
and weakly across).

**Fix:** Either use a single `sqrt(2 log p_total)` or, better, calibrate
`global_lambda` by cross-validation on held-out donors.

### B9. The "smooth L1" `sqrt(β² + 1e-3)` thresholds around |β| ≈ 0.03 while the selection threshold is 0.01

Effects whose true magnitude is between ~0.01 and 0.03 are partially
smoothed (so over-shrunk) and then declared "selected" — they will be biased
toward zero in the refit because the refit only sees coefficients above the
post-Adam threshold. This biases small but real cross-species effects.

**Fix:** Reduce `eps` to ~1e-5 (or use proximal/ISTA which gives true L1) and
align the selection threshold with the smoothing scale.

### B10. Two-stage mean→dispersion fit ignores joint uncertainty

Freezing the mean for the dispersion fit is standard, but the README /
inference module silently uses the dispersion fit's `θ_ig` when computing the
mean Wald SE. The variability of `θ̂` is ignored. This is acceptable for
screening but understated SEs when dispersion is itself uncertain (which is
most clades with few donors).

**Fix:** Either bootstrap (donors) or add a second-order correction (Cox–Reid
adjusted profile likelihood).

---

## C. Minor / hygiene

- `_build_design_matrices` walks `gm.iterrows()` for taxonomy and species
  indexing — O(n_groups · n_levels) Python loop; vectorize with pandas
  merges. Performance, not correctness.
- `_parse_newick` accepts no branch lengths and silently strips them; warn
  instead.
- Adam + `ReduceLROnPlateau(patience=20)` + outer `patience=50` early stop ⇒
  termination governed mostly by LR decay, not by the gradient. Convergence
  is not checked.
- `dispersion_lambda * sqrt(2 log p)` applied to a column-L2-normalized
  design (unweighted) when `p_family > 1` — fine, but inconsistent with the
  Fisher-weighted scaling on the mean side. Document or unify.
- The Wald inference's `ridge="auto"` adds `1e-8 * trace(H)/k` to the
  Hessian; for very small H (low-expression genes) this can be effectively a
  no-op and leave numerical noise. Consider an absolute floor.

---

## Remediation plan (priority-ordered)

| # | Change | Affected file(s) | Effort | Impact |
|---|---|---|---|---|
| 1 | Donor-clustered sandwich SE in `compute_wald_significance` | `inference.py` | 1 day | Largest reduction in false positives |
| 2 | Stop L1-penalizing donor (and probably batch); keep them unpenalized or weakly ridge | `model.py` (`_compute_penalty_scales`, `_fit_gene_chunk`) | 0.5 day | Removes systematic bias on species coefficients |
| 3 | Replace reference coding of `species_tax_<level>` with true sum-to-zero (Helmert) — or fix README | `model.py` lines ~96–133; `README.md` | 0.5 day | Coefficients become species-symmetric and reproducible across species orderings |
| 4 | Drop the root columns from `A_tax_full` and `A_species_full` (aliased with intercept) | `taxonomy_tree.py`, `species_tree.py`, `_build_design_matrices` | 0.5 day | Removes rank deficiency, makes coefficient names meaningful |
| 5 | Per-gene Fisher weights for penalty scaling (`c_j[g]`) | `model.py` line 447 and analogues | 0.5 day | Reproducibility across `gene_chunk_size`; correct shrinkage by gene |
| 6 | Switch refit to L-BFGS and fix the gradient-mask leakage in `_refit_support` | `model.py` | 1 day | Real MLE → valid Wald SE foundation |
| 7 | Replace `library_sizes` with a TMM / housekeeping-ortholog size factor for cross-species comparisons | `fit_tree_nb`, new normalization helper | 1–2 days | Removes confounding of species effects with composition |
| 8 | Add an explicit "intracluster correlation" correction (ρ) to the dispersion offset or drop the offset | `_fit_dispersion_stage`, `disp_offset` | 1 day | Honest dispersion at large pseudobulks |
| 9 | Implement either data-splitting selective inference or a debiased-lasso pass per gene; relabel "q" appropriately | `inference.py` | 2–3 days | Real FDR control |
| 10 | Reduce smooth-L1 `eps` (or switch to proximal/ISTA) and align selection threshold | `model.py` | 0.5 day | Removes small-effect attenuation |
| 11 | Validate with simulation: known tree-structured β, null cross-species comparisons, donor pseudoreplication — confirm Type-I rate is at nominal level after fixes #1–#2 | new `tests/test_calibration.py` | 1–2 days | Empirical justification of the inferential claims |
| 12 | **Rebuild `compute_wald_significance` per-family** (do not concatenate every selected mean family into one joint Hessian) and drop the root columns from the design. See §D below for empirical justification. | `inference.py`, `_build_design_matrices` in `model.py` | 1 day | Restores any usable Wald inference at all |

---

## D. Empirical findings from `notebooks/sim_eval_tree_nb.ipynb`

A simulation that mirrors the **exact spinal-cord taxonomy / species /
donor / batch structure** (1042 pseudobulks, 4 species, 52 donors, 48
batches after coding, 80 genes, planted strong effects at the
species_tax_Class/Subclass/Group levels and elevated dispersion on
selected Subclass nodes) was run end-to-end. Headline results:

* **L1 + refit recovery is good.** Per-family Pearson correlation
  between planted and fitted β: **Class 0.82, Subclass 0.98, Group 0.93**.
  RMSE on planted entries 0.48–1.06; attenuation bias of −0.11 (Class)
  to −0.54 (Group) — consistent with the smooth-L1 issue in **§B9**.
* **Wald inference is non-functional at realistic sparsity.** With the
  package default `global_lambda=0.1` (and even more so at 0.01, which
  is what the spinal-cord notebook uses), L1 selects ~200–250
  coefficients per gene in each of `tax_global` and `species_tax_*`.
  The joint Hessian over the union of selected families is
  **numerically rank-deficient for every gene**:
  * `100% of Wald rows are flagged `unstable=True`` (cond > 1e10).
  * Median per-coefficient SE ≈ 330. Planted β = 1.5 → z ≈ 0.005,
    p ≈ 1.0.
  * **Sensitivity at q < 0.05 = 0%** on planted strong effects across
    every species_tax family. **AUC(−q) ≈ 0.5** (Wald ranking is not
    informative).
* **Dispersion fit works.** Separation AUC between planted-high-disp
  and null Subclass nodes: **0.73**; median estimated log-overdisp on
  planted-high entries +0.4 vs planted +1.0 (downward attenuation,
  consistent with §B5).

### Implication for the priority ordering

Remediation item **#12 (rebuild Wald per-family + drop root columns)**
should be promoted to the **highest priority** alongside #1 (donor
sandwich SE). Without #12 the Wald layer described in `inference.py`
and consumed by the analysis notebook is empirically equivalent to
"all coefficients non-significant" — the q-filter in
`06-0_lasso_divergence_drivers_analysis.ipynb` would mask out every
effect, including planted +1.5-log-fold deviations.

The simulation script lives at
`notebooks/sim_eval_tree_nb.ipynb`; it runs in ~2 minutes on CPU and
should be re-executed after every change to `inference.py` as a
calibration regression test.

### D3. Further iterations: tighter smooth-L1 + orthogonal tree

**Tighter smooth-L1 (`_smooth_l1` eps 1e-3 → 1e-6, new default).**
The smooth approximation `sqrt(β² + ε)` thresholds around `|β|≈√ε`.
At ε=1e-3 this gave a soft threshold ~0.03, attenuating small true
effects. Dropping to 1e-6 (threshold ~0.001) is a strict
improvement on the sim:

| family   | sens@q05 (ε=1e-3) | sens@q05 (ε=1e-6) | Type-I (1e-3) | Type-I (1e-6) | calib scale (1e-3) | (1e-6) |
|----------|--------------------|--------------------|----------------|----------------|---------------------|--------|
| Class    | 0.40               | 0.60               | 18%            | 19%            | ×2.02               | ×1.00  |
| Subclass | 0.80               | 0.90               | 16%            | 20%            | ×1.58               | ×1.00  |
| Group    | 0.40               | 0.60               | 5%             | 5%             | ×1.00               | ×1.00  |

Sensitivity is up 20-50% at similar Type-I, and the empirical-null
inflation factor collapsed to 1.0 for all `species_tax_*` families —
i.e., the per-family Hessian is now well-calibrated *without* needing
post-hoc inflation. The residual Type-I inflation at Class/Subclass
comes from heavy-tailed z-stats (gene-to-gene fit heterogeneity), not
from SE underestimation per se.

**Orthogonal tree (`_build_design_matrices(..., orthogonal_tree=True)`,
opt-in).** Helmert-style residualization of each non-root path-indicator
column against its parent column. Empirically confirms the
rank-deficiency diagnosis: Class empirical-null scale drops 2.02→1.24
even at ε=1e-3, meaning the original parameterization's anti-
conservative SE was indeed driven by ancestor/descendant aliasing.
However Type-I overall did not improve in the sim benchmark
(sibling-pair aliasing within each parent remains, and the planted
ground truth is in the original parameterization so accuracy metrics
become non-comparable). Kept as **opt-in** (`fit_tree_nb(...,
orthogonal_tree=True)`) — interpretable as "deviation of each subtree
from its parent's mean" rather than "absolute path contribution".



### D2. Iteration log (compute_wald_significance overhaul)

The Wald layer has been rebuilt across 5 iterations using
`notebooks/sim_eval_tree_nb.ipynb` as the calibration benchmark. Final
state: well-calibrated for non-aliased families, partially calibrated
for inner tree levels via empirical-null SE inflation.

**Iteration 1 — per-family Hessian + drop root cols + ridge floor**
(`per_family=True`, `drop_root_columns=True`, `absolute_ridge_floor=1e-4`).
Result: 100%→0% unstable, median SE 330→0.04–0.12, AUC 0.5→0.79–0.95,
sensitivity 0→0.5–0.9. But Type-I on null genes still inflated:
Class 48%, Subclass 34%, Group 6%, species_global/donor/batch high.

**Iteration 2 — donor-clustered sandwich SE.** Implemented
`cluster_se=True` (NB-score-residual sandwich, Stata HC1 correction).
On a correctly-specified NB simulation this **made things worse** (refit
absorbs residual variance on the selected support → score residuals
tiny → Ω too small → SE anti-conservative). Reverted default to
`False`; kept as opt-in for real data with donor random effects the
IID NB model doesn't capture.

**Iteration 3 — `hessian_on_full_family=True`** (default). Build the
per-family Hessian over **every** column of the family (not just the
L1-selected support) to remove post-selection bias on the SE. Mild
improvement on Group; Class/Subclass Type-I essentially unchanged
(48% / 34% → 48% / 34%). Confirmed that selection bias on **β̂** is
not the dominant issue.

**Iteration 4 — `pearson_phi=True`** + **`debias_one_step=True`** +
**`report_all_family_cols=True`**. Pearson φ_g = Σ(Y_g−μ_g)²/V_g/(n−p),
floored at 1 (quasi-likelihood SE inflation when dispersion under-fit).
One-step debiased β̂. Reporting Wald for *all* family columns, not just
L1-selected, with a `selected_l1` flag. None of these moved Type-I
materially on null genes because the dispersion fit is approximately
correct on the null subset (φ ≈ 1) and the bias is not at the column
level but at the design-aliasing level.

Empirical SD of β̂ across the 40 null genes vs the model-based SE
revealed the root cause: SE/emp_SD ratios were Class 0.47, Subclass
0.76, donor 0.72, batch 0.76 — **anti-conservative by 1.3–2×** — while
Group was 1.21 (over-conservative, why it calibrated). The
per-family conditional Hessian gives the *profile* SE (as if other
families were fixed), but at inner tree levels (Class ⊃ Subclass ⊃
Group) the families are jointly estimated and partially alias each
other, so the conditional SE is too tight.

**Iteration 5 — joint full Hessian (`joint_full_hessian=True`).**
Build H over ALL columns of ALL families with sum-to-zero anchors and
ridge floor. Confirmed empirically: the joint design is
**rank-deficient** (Class span = ⊕ Subclass spans = ⊕ Group spans),
ridge dominates, all SEs → ~215, all p → 1, AUC → 0.5. Documented this
as a fundamental identifiability limit of the tree-nested
parameterization at multiple levels simultaneously.

**Iteration 6 (final) — `empirical_null_calibration=True`** (default).
Efron-style robust SE inflation: per family, compute the central |z|
quartile of the reported coefs and scale SE so that this matches the
N(0,1) quartile (0.6745). Per-family `se_emp_inflation` is recorded
in the output. Final calibration:

| family               | Type-I (nominal 5%) | Sensitivity | AUC(−q) | emp_inflation |
|----------------------|---------------------|-------------|---------|---------------|
| species_tax_Class    | 18%                 | 0.40        | 0.78    | ×2.02         |
| species_tax_Subclass | 16%                 | 0.80        | 0.94    | ×1.58         |
| species_tax_Group    | 5%                  | 0.40        | 0.89    | ×1.00         |
| species_global       | 17%                 | —           | —       | ×1.00 *(insufficient n)* |
| tax_global           | 0%                  | —           | —       | ×1.00         |
| donor                | 11%                 | —           | —       | ×2.11         |
| batch                | 13%                 | —           | —       | ×1.71         |

Class/Subclass/donor/batch still carry residual inflation (~3× nominal)
because the z-stats are heavy-tailed (gene-to-gene fit heterogeneity),
and IQR-based calibration only matches the central mass. **Treat
q-values at Class/Subclass/donor/batch as a screening signal; use
species_global, tax_global, Group, and final_cluster as the trustworthy
inferential families.** Or use sensitivity-adjusted thresholds (e.g.,
q < 0.01 instead of 0.05) at the inner tree levels.

The fundamental fix (LRT-based tree-level inference that handles the
non-identifiability properly, or a re-parameterization that breaks the
nested aliasing) is a larger architectural change beyond the scope of
this iteration cycle.

### D3. Level-specific empirical-Bayes calibration

The taxonomy basis now has an opt-in full earlier-level residualization:
every level is projected off the span of all earlier levels, and each residual
column is restored to its original L2 norm. The same transform is applied to
`species_tax_*` blocks after removing taxonomy/species nuisance spans. EB fits
enable this automatically.

Raw coefficient vectors remain redundant within sibling/species contrast
spaces, so level comparisons use the invariant fitted contribution
`X_level @ beta_level`. Normalized evolutionary burden is its mean square over
observed pseudobulks and genes. Synthetic truths are canonicalized to the
minimum-norm sum-to-zero coordinates before coefficient scoring.

Thirty known-truth simulations per dense-small/sparse-large regime and 100
donor-split coverage replicates gave:

| method | mean magnitude error | mean log-burden error | donor localization | null coverage | null rejection |
|---|---:|---:|---:|---:|---:|
| fixed L1 | 0.022 | 0.029 | 0.88 | 0.963 | 0.037 |
| Gaussian EB | 0.062 | 0.147 | 0.77 | 0.960 | 0.040 |
| Laplace EB | 0.067 | 0.100 | 0.81 | 0.960 | 0.040 |
| spike-and-slab EB | 0.054 | 0.084 | 0.77 | 0.960 | 0.040 |

All methods localized the known changes in the balanced reconstruction
simulations. Fixed L1 remains the strongest point-reconstruction baseline.
Spike-and-slab is selected only among EB priors because it best balances dense
and sparse burden recovery while exposing a learned change prevalence and slab
magnitude. EB therefore remains opt-in for comparative level-burden analyses;
it does not replace donor-honest held-out intervals or the default L1 fit.

### D4. Fixed-L1 tuning and legacy-basis comparison

A subsequent 30-simulation comparison selected the L1 penalty by mean
negative-binomial likelihood on two unseen donors per species. The selection
stage did not use planted coefficients. Final recovery was measured from each
level's fitted contribution `X_level @ beta_level`, making legacy and
orthogonal coordinates directly comparable.

The prespecified grid was `0.005, 0.01, 0.02, 0.05, 0.09, 0.15, 0.30`.
Pooled validation selected `0.15` for both bases. Relative to `0.05`, the
orthogonal tuned fit improved held-out NLL by 0.0117 per observation (paired
95% CI 0.0046 to 0.0189). Its advantage over `0.09` was only 0.0028 (95% CI
-0.0006 to 0.0062), so the fine distinction between 0.09 and 0.15 is not
resolved.

| method | normalized effect RMSE | mean absolute log-burden error | combined score |
|---|---:|---:|---:|
| orthogonal L1, fixed 0.05 | 0.126 | 0.026 | 0.162 |
| orthogonal L1, tuned 0.15 | 0.113 | 0.047 | 0.167 |
| spike-and-slab EB | 0.107 | 0.094 | 0.207 |
| Laplace EB | 0.108 | 0.106 | 0.220 |
| Gaussian EB | 0.154 | 0.156 | 0.320 |
| legacy L1, historical 0.09 | 0.472 | 1.263 | 1.856 |
| legacy L1, tuned 0.15 | 0.470 | 1.764 | 2.340 |

Thus orthogonalization is the dominant improvement. Predictively tuned L1 is
the better point-effect estimator than fixed 0.05, whereas fixed 0.05 is the
better evolutionary-burden estimator. Spike-and-slab has the lowest
sparse-large normalized effect RMSE (0.101 versus 0.110 for tuned orthogonal
L1), but its sparse burden is more attenuated (ratio 0.868 versus 0.964).
