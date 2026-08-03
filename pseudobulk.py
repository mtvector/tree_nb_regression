"""Pseudobulk aggregation without dense tensor allocation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class PseudobulkData:
    group_meta: pd.DataFrame
    cell_to_group: sparse.csc_matrix
    n_groups: int
    library_sizes: np.ndarray | None


def build_pseudobulk(
    obs: pd.DataFrame,
    taxonomy_col: str,
    species_col: str,
    batch_col: str | None = None,
    donor_col: str | None = None,
    min_cells_per_pseudobulk: int = 10,
) -> PseudobulkData:
    """Build pseudobulk grouping from observed combinations only.

    Returns a sparse cell-to-pseudobulk assignment matrix P such that
    Y = P.T @ X gives pseudobulk counts.
    """
    groupby_cols = [taxonomy_col, species_col]
    if batch_col is not None:
        groupby_cols.append(batch_col)
    if donor_col is not None:
        groupby_cols.append(donor_col)

    obs_reset = obs.reset_index(drop=True)
    group_keys = obs_reset[groupby_cols].astype(str)
    group_labels = group_keys.apply(lambda row: "||".join(row), axis=1)

    group_counts = group_labels.value_counts()
    valid_groups = set(group_counts[group_counts >= min_cells_per_pseudobulk].index)

    valid_mask = group_labels.isin(valid_groups)
    valid_indices = np.where(valid_mask.values)[0]
    valid_group_labels = group_labels.iloc[valid_indices]

    unique_groups = sorted(valid_groups)
    group_to_idx = {g: i for i, g in enumerate(unique_groups)}

    n_cells = len(obs_reset)
    n_groups = len(unique_groups)

    rows = valid_indices
    cols = np.array([group_to_idx[g] for g in valid_group_labels])
    data = np.ones(len(rows), dtype=np.float64)

    cell_to_group = sparse.csc_matrix(
        (data, (rows, cols)), shape=(n_cells, n_groups), dtype=np.float64
    )

    group_meta_rows = []
    for g in unique_groups:
        parts = g.split("||")
        row_dict = {col: parts[i] for i, col in enumerate(groupby_cols)}
        row_dict["group_label"] = g
        row_dict["n_cells"] = int(group_counts[g])
        group_meta_rows.append(row_dict)
    group_meta = pd.DataFrame(group_meta_rows)

    return PseudobulkData(
        group_meta=group_meta,
        cell_to_group=cell_to_group,
        n_groups=n_groups,
        library_sizes=None,
    )


def aggregate_chunk(
    X_chunk: sparse.spmatrix | np.ndarray,
    cell_to_group: sparse.spmatrix,
) -> np.ndarray:
    """Aggregate a gene chunk: Y = P.T @ X_chunk."""
    if sparse.issparse(X_chunk):
        X_chunk = X_chunk.tocsc()
    P_t = cell_to_group.T.tocsr()
    Y = P_t @ X_chunk
    if sparse.issparse(Y):
        Y = Y.toarray()
    return np.asarray(Y, dtype=np.float64)
