# Tree-Structured Pseudobulk NB Regression

Memory-efficient negative-binomial regression using taxonomy and species trees
as additive path-indicator design bases.

> [!WARNING]
> This is research software for exploratory coefficient screening. The current
> Wald statistics are post-selection, use conditional per-family Hessians, and
> do not account fully for donor-level correlation or dependencies between
> nested tree levels. Their `p` and `q` values are not confirmatory error-rate
> guarantees. See [Statistical status](#statistical-status) and `review.md`.

## Installation

The supported reproducible environment is defined in `environment.yml`.

```bash
export TMPDIR=/scratch/tmp MPLCONFIGDIR=/scratch/mpl
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
mamba env create -f environment.yml
mamba activate tree-nb-regression
python -m pip install --no-deps -e .
```

For development checks, install the pinned optional dependencies:

```bash
python -m pip install --cache-dir /scratch/pipcache -e '.[dev]'
```

## Model

For pseudobulk group `i` and gene `g`:

```text
Y[i,g] ~ NB(mu[i,g], theta[i,g])

log(mu[i,g]) =
    offset[i]
  + alpha[g]
  + A_tax[k_i, :] @ beta_tax[:, g]
  + A_species[s_i, :] @ beta_species[:, g]
  + A_species_tax_level[...] @ gamma[:, g]
  + X_batch[i, :] @ delta_batch[:, g]
  + optional donor effect
```

The selection pass applies a smooth, Fisher-scaled L1 penalty to every design
family, including batch and donor when supplied. The Fisher scale is currently
averaged over the genes in each processing chunk. Non-interaction families are
then refitted without an L1 penalty on the selected support using Adam;
`species_tax_*` estimates remain frozen at their penalized values.

The default global penalty is `DEFAULT_GLOBAL_LAMBDA = 0.09`.

### Species-by-taxonomy coding

For taxonomy node `n`, let `S(n)` be the species represented below that node
and `K = |S(n)|`. Nodes with fewer than two species are skipped. For each
remaining species `s`, the design contains a centered column:

```text
X[(s,n)][i] = +(K-1)/K  if group i is species s below node n
              -1/K      if group i is another species below node n
               0        otherwise
```

The columns sum to zero across species for each row. Their coefficients have a
soft quadratic anchor on `sum_s beta[s,n,g]`; therefore coefficient sums are
approximately, not algebraically, zero. The interaction coefficients are
species-symmetric screening effects, while mean effects at the node are carried
by `tax_global`.

## Usage

```python
import anndata as ad

from tree_nb_regression import fit_tree_nb

adata = ad.read_h5ad("/data/counts.h5ad")

res = fit_tree_nb(
    adata,
    taxonomy_cols=[
        "Neighborhood",
        "Class_V2",
        "Subclass_V2",
        "Group_V2",
        "final_cluster",
    ],
    species_col="species",
    species_tree="(Mouse,((Macaque_mulatta,Macaque_nemestrina),Human));",
    counts_layer="UMIs",
    batch_col="batch",
    donor_col="donor_name",
)

print(res.summary())
tax_coefficients = res.get_coefficients_df("tax_global")
```

Raw nonnegative integer counts are required. Prefer `counts_layer` to replacing
`adata.X`, which avoids copying a potentially large matrix.

## Tree-structured dispersion

Pass `fit_dispersion_tree=True` to fit lasso-selected dispersion deviations
after the mean fit. Coefficients parameterize log-overdispersion (`-log theta`),
so positive values denote greater variability than the gene baseline.

```python
res = fit_tree_nb(
    adata,
    taxonomy_cols=["Neighborhood", "Class_V2", "Subclass_V2", "Group_V2", "final_cluster"],
    species_col="species",
    species_tree="(Mouse,((Macaque_mulatta,Macaque_nemestrina),Human));",
    counts_layer="UMIs",
    donor_col="donor_name",
    fit_dispersion_tree=True,
    dispersion_lambda=0.3,
    min_replicates_per_node=3,
)

dispersion = res.get_dispersion_df("tax_global")
calls = res.call_dispersion("tax_global", threshold=0.2)
```

By default, only `tax_global` and `species_global` participate in the
dispersion fit. Exact duplicate columns and columns with too few replicate
groups are removed. The optional `-log(n_cells)` offset assumes independent,
similarly sequenced cells; correlated cells or heterogeneous per-cell depth can
violate that approximation.

## Statistical status

The coefficient-fitting pipeline is useful for exploratory ranking and has
synthetic recovery tests. Important limitations remain:

- pseudobulks from the same donor are treated as independent by Wald inference;
- donor and batch terms participate in L1 support selection;
- root and nested path-indicator columns are not fully identifiable jointly;
- Wald tests condition on other fitted families and ignore selection;
- BH adjustment is performed on the reported selected support, not the full
  pre-selection hypothesis universe;
- fitted dispersion is treated as known when computing mean-effect uncertainty;
- penalty calibration is gene-chunk dependent.

The empirical simulation summarized in `review.md` found inflated Type-I error
for several inner taxonomy levels and nuisance families. Use the legacy Wald
`p` and `q` values only as screening diagnostics; use the donor-honest workflow
below for the currently validated discovery contrast.

## Donor-honest post-selection intervals

`donor_honest_intervals` is the supported route for discovery-oriented
species-by-taxonomy inference. It splits **whole donors within each species**:
the training donors localize `species_tax_*` candidates, while the disjoint
inference donors estimate an identifiable contrast and its interval.

```python
from tree_nb_regression import DonorSelectionConfig, donor_honest_intervals

honest = donor_honest_intervals(
    adata,
    taxonomy_cols=("Neighborhood", "Class_V2", "Subclass_V2", "Group_V2", "final_cluster"),
    species_col="species",
    species_tree="(Mouse,((Macaque_mulatta,Macaque_nemestrina),Human));",
    donor_col="donor_name",
    counts_layer="UMIs",
    selection=DonorSelectionConfig(global_lambda=0.01, max_iter=500),
    random_state=41,
)
discoveries = honest.discoverable.query("q < 0.05")
```

Each returned interval estimates a species' mean donor-level log counts per
million within a taxonomy node minus the equally weighted mean of the other
species at that node. It is deliberately **not** an interval for a raw
path-indicator coefficient. The held-out `p` values are valid conditional on
the training-donor screen; `q` is BH-adjusted across the selected held-out
contrasts.

At least four donors per species are required to form a split with two
held-out donors, but this is a minimum for computation rather than a strong
design. The included calibration uses 12 donors per species. A batch perfectly
confounded with species remains non-identifiable, even with donor splitting.

Run the end-to-end calibration before changing this workflow:

```bash
python - <<'PY'
import json
from tree_nb_regression import run_donor_honest_calibration

summary = run_donor_honest_calibration(n_simulations=300, random_state=202)
print(json.dumps(summary.__dict__ | {"records": "omitted"}, indent=2, default=str))
PY
```

For the committed simulator and seed, the 300-replicate run is expected to
have nominal-95% coverage between 90% and 97.5%, null rejection at or below
6%, and correct Class-A localization in at least 70% of screens. Calibration
records belong in `/results`, not the repository.

## Repository layout

- `taxonomy_tree.py`: taxonomy construction and path-indicator design
- `species_tree.py`: limited Newick/tuple parser and species design
- `pseudobulk.py`: sparse observed-group aggregation
- `model.py`: chunked NB fitting, selection, refit, and dispersion model
- `inference.py`: exploratory Wald statistics and BH adjustment
- `results.py`: result container and tabular accessors
- `tests/`: synthetic unit and integration tests
- `notebooks/`: simulation and spinal-cord analyses
- `review.md`: detailed statistical review and simulation history

## Development

Run checks from the repository root:

```bash
export TMPDIR=/scratch/tmp MPLCONFIGDIR=/scratch/mpl
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
python -m pytest
python -m ruff check .
python -m mypy --strict
python -m nbstripout --verify notebooks/*.ipynb
```

Notebook outputs are intentionally excluded from version control. Store final
capsule artifacts in `/results`; use `/scratch` only for ephemeral intermediates.
After cloning, run `nbstripout --install` once to activate the configured Git
filter locally.
