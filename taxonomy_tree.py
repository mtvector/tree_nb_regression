"""Taxonomy tree construction from obs columns."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class TaxonomyTree:
    node_table: pd.DataFrame
    leaf_mapping: dict[str, list[str]]
    A_tax_leaf: sparse.csr_matrix
    leaf_ids: list[str]
    node_ids: list[str]
    levels: list[str]


def build_taxonomy_tree_from_obs(
    obs: pd.DataFrame,
    taxonomy_cols: list[str],
) -> TaxonomyTree:
    """Build a taxonomy tree from obs columns defining root-to-leaf paths.

    Each row in obs defines a path through the taxonomy columns (from root to leaf).
    Nodes are disambiguated by full path if a label appears under multiple parents.
    """
    taxonomy_cols = list(taxonomy_cols)
    paths_df = obs[taxonomy_cols].drop_duplicates().reset_index(drop=True)

    nodes: list[dict[str, Any]] = []
    node_id_set: set[str] = set()
    label_parents: dict[str, set[str]] = {}

    for _, row in paths_df.iterrows():
        parent_id = "root"
        for level_idx, col in enumerate(taxonomy_cols):
            label = str(row[col])
            path_key = "/".join(str(row[c]) for c in taxonomy_cols[: level_idx + 1])
            node_id = path_key

            if node_id not in node_id_set:
                nodes.append(
                    {
                        "node_id": node_id,
                        "label": label,
                        "level": col,
                        "level_idx": level_idx,
                        "parent_id": parent_id if level_idx > 0 else None,
                        "path": path_key,
                    }
                )
                node_id_set.add(node_id)

            if label not in label_parents:
                label_parents[label] = set()
            label_parents[label].add(parent_id)
            parent_id = node_id

    node_table = pd.DataFrame(nodes)
    node_table = node_table.sort_values(["level_idx", "node_id"]).reset_index(drop=True)

    leaf_col = taxonomy_cols[-1]
    leaf_nodes = node_table[node_table["level"] == leaf_col]
    leaf_ids = list(leaf_nodes["node_id"].values)
    all_node_ids = list(node_table["node_id"].values)
    node_to_idx = {nid: i for i, nid in enumerate(all_node_ids)}

    leaf_mapping: dict[str, list[str]] = {}
    for _, leaf_row in leaf_nodes.iterrows():
        path_parts = leaf_row["path"].split("/")
        ancestor_ids = []
        for k in range(1, len(path_parts) + 1):
            ancestor_ids.append("/".join(path_parts[:k]))
        leaf_mapping[leaf_row["node_id"]] = ancestor_ids

    n_leaves = len(leaf_ids)
    n_nodes = len(all_node_ids)
    rows, cols, data = [], [], []
    for leaf_idx, leaf_id in enumerate(leaf_ids):
        for ancestor_id in leaf_mapping[leaf_id]:
            if ancestor_id in node_to_idx:
                rows.append(leaf_idx)
                cols.append(node_to_idx[ancestor_id])
                data.append(1.0)

    A_tax_leaf = sparse.csr_matrix(
        (data, (rows, cols)), shape=(n_leaves, n_nodes), dtype=np.float64
    )

    return TaxonomyTree(
        node_table=node_table,
        leaf_mapping=leaf_mapping,
        A_tax_leaf=A_tax_leaf,
        leaf_ids=leaf_ids,
        node_ids=all_node_ids,
        levels=taxonomy_cols,
    )
