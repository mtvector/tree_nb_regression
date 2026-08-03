# Tree-Structured Pseudobulk NB Regression

Memory-efficient negative-binomial regression using taxonomy hierarchy and species tree as additive path-indicator design bases.

## Model

For pseudobulk group `i`, gene `g`:

```
Y[i,g] ~ NB(mu[i,g], theta[i,g])

log(mu[i,g]) =
    offset[i]
  + alpha[g]
  + A_tax[k_i, :] @ beta_tax[:, g]                  # taxonomy path indicator
  + A_species[s_i, :] @ beta_species[:, g]          # species-tree path indicator
  + A_species_tax_level[...] @ gamma[:, g]          # species x tax interaction
  + X_batch[i, :] @ delta_batch[:, g]
  + optional donor effect
```

Penalties are weighted L1 (Laplace) calibrated by Fisher information so that
coefficients have comparable cost per unit effect on the linear predictor.

### Sum-to-zero coding of `species_tax_<level>` blocks

For each taxonomy node `n` at level L with species set `S(n)` (those species
that have any cells in the subtree under `n`), the interaction block
`species_tax_<L>` uses **K = |S(n)| centered indicator columns**, one per
species in `S(n)`:

```
X[(s, n)][i] = +(K-1)/K  if  anc(i)=n  and  species(i)=s
             = -1/K      if  anc(i)=n  and  species(i)!=s
             =  0        if  anc(i)!=n
```

Each row's K values sum to 0 (column space is orthogonal to the constant and
to `tax_global[n]`), and a quadratic anchor
`0.5 * SUMZERO_ANCHOR_MU * (sum_s beta[s, n, g])^2` is added to both the L1
optimization and the post-selection refit so that fitted coefficients
satisfy `sum_{s in S(n)} beta[s, n, g] = 0` to floating-point precision.

This makes the parameterization:

- **Symmetric across species under the L1 penalty** (no arbitrary
  "reference" species — permuting species labels yields the same
  |beta| per (species, node, gene)).
- **Restricted to identifiable contrasts** at each node: species not
  present in the subtree get no column, and nodes with |S(n)| < 2 are
  skipped entirely.
- **Interpretable as deviations**: beta[s, n, g] is species s's deviation
  from the cross-species mean at node n for gene g; the mean-across-species
  effect at node n lives entirely in `tax_global[n]`.

The Wald-inference Hessian is augmented with the same quadratic anchor so
post-selection standard errors are well-defined despite the rank-(K-1)
within-block design. See `SUMZERO_ANCHOR_MU` in `model.py`.

## Usage

```python
import anndata as ad
from tree_nb_regression import fit_tree_nb

adata = ad.read_h5ad("path/to/data.h5ad")
# Use raw counts (e.g. from UMIs layer)
adata.X = adata.layers["UMIs"].copy()

res = fit_tree_nb(
    adata,
    taxonomy_cols=["Neighborhood", "Class_V2", "Subclass_V2", "Group_V2", "final_cluster"],
    species_col="species",
    species_tree="(Mouse,((Macaque_mulatta,Macaque_nemestrina),Human));",
    batch_col="batch",
    donor_col="donor_name",
)

# Examine results
print(res.summary())
print(res.diagnostics)

# Get coefficients for a specific family
tax_df = res.get_coefficients_df("tax_global")
```

## Tree-structured dispersion (optional)

Pass `fit_dispersion_tree=True` to also fit lasso‑selected, tree‑structured
**dispersion** deviations after the mean is fit. Each pseudobulk group's
log‑overdispersion (`-log θ`) becomes
`phi0[g] + tree deviations + (optional) -log(n_cells)` so that a *positive*
coefficient at a tree node means **"this clade is more variable across
donors than the gene baseline"** — and conversely for negative.

```python
res = fit_tree_nb(
    adata,
    taxonomy_cols=[...],
    species_col="species",
    species_tree="(Mouse,((Macaque_mulatta,Macaque_nemestrina),Human));",
    donor_col="donor_name",
    fit_dispersion_tree=True,
    dispersion_lambda=0.3,          # independent of global_lambda
    min_replicates_per_node=3,      # mask under-replicated nodes
)

# Inspect dispersion calls
disp_df = res.get_dispersion_df("tax_global")          # rows = tree node ids
calls   = res.call_dispersion("tax_global", threshold=0.2)
calls[calls["above"]]   # nodes called "above baseline" for ≥1 gene
```

Design notes:
- **Two‑stage** by default: mean coefficients (post‑selection refit) are
  frozen before dispersion fitting, so L1‑shrunk mean structure can't leak
  into dispersion as fake heterogeneity.
- **Replicate masking**: dispersion design columns whose subtree spans
  fewer than `min_replicates_per_node` distinct donors are dropped *before*
  L1, since they cannot identify dispersion.
- **Duplicate columns dropped**: when neighboring tree levels yield
  identical indicator vectors (common in shallow slices), only one
  representative is kept to avoid the L1 spreading a single effect across
  redundant columns.
- **Cell‑count offset**: `-log(n_cells)` is added to log‑overdispersion by
  default (`dispersion_cell_offset=True`) so pseudobulk size differences
  don't masquerade as dispersion shifts.
- **Limitation**: only the marginal `tax_global` and `species_global` are
  enabled by default; `species_tax_<level>` interactions on dispersion are
  intentionally omitted (under‑replicated in typical datasets).


## Architecture

- `taxonomy_tree.py` — Builds taxonomy tree from obs columns, returns path-indicator matrix
- `species_tree.py` — Parses Newick/tuple species tree, returns path-indicator matrix
- `pseudobulk.py` — Sparse pseudobulk aggregation (no dense tensor allocation)
- `model.py` — Core NB fitting with chunked gene processing and weighted L1
- `results.py` — Result container with accessors

## Key Design Decisions

1. **No phylogenetic covariance** — Trees are used only as path-indicator design bases
2. **Sparse aggregation** — Only observed combinations are materialized
3. **Fisher-calibrated penalties** — Each column penalized proportional to its Fisher norm
4. **Post-selection refit** — After L1 selects support, refit with weak ridge to reduce bias
5. **Gene chunking** — Processes genes in configurable chunks to limit memory

## Tests

```bash
cd /code/HMBA_Genomics/SpinalCord/xspecies/analysis
python -m pytest tree_nb_regression/tests/test_tree_nb.py -v
```
