"""Species tree construction from Newick string or nested tuples."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd
from scipy import sparse


NestedTuple = Union[str, tuple]


@dataclass
class SpeciesTreeDesign:
    A_species: sparse.csr_matrix
    species_order: list[str]
    node_ids: list[str]
    node_table: pd.DataFrame


def _parse_newick(newick: str) -> tuple:
    """Parse a simple Newick string (no branch lengths) into nested tuples."""
    newick = newick.strip().rstrip(";").strip()

    def _parse_subtree(s: str, pos: int) -> tuple[Any, int]:
        if s[pos] == "(":
            pos += 1
            children = []
            while True:
                child, pos = _parse_subtree(s, pos)
                children.append(child)
                if pos < len(s) and s[pos] == ",":
                    pos += 1
                elif pos < len(s) and s[pos] == ")":
                    pos += 1
                    break
                else:
                    break
            label = ""
            while pos < len(s) and s[pos] not in (",", ")", ";", ":"):
                label += s[pos]
                pos += 1
            if pos < len(s) and s[pos] == ":":
                while pos < len(s) and s[pos] not in (",", ")", ";"):
                    pos += 1
            return (tuple(children), label.strip()), pos
        else:
            label = ""
            while pos < len(s) and s[pos] not in (",", ")", ";", ":"):
                label += s[pos]
                pos += 1
            if pos < len(s) and s[pos] == ":":
                while pos < len(s) and s[pos] not in (",", ")", ";"):
                    pos += 1
            return label.strip(), pos

    result, _ = _parse_subtree(newick, 0)
    return result


def _collect_nodes(
    tree: tuple | str,
    parent_id: str | None,
    nodes: list[dict],
    counter: list[int],
) -> None:
    """Recursively collect nodes from nested tuple tree."""
    if isinstance(tree, str):
        nodes.append(
            {
                "node_id": tree,
                "label": tree,
                "is_leaf": True,
                "parent_id": parent_id,
            }
        )
    else:
        children, label = tree
        if label:
            node_id = label
        else:
            node_id = f"_clade_{counter[0]}"
            counter[0] += 1
        nodes.append(
            {
                "node_id": node_id,
                "label": label or node_id,
                "is_leaf": False,
                "parent_id": parent_id,
            }
        )
        for child in children:
            _collect_nodes(child, node_id, nodes, counter)


def _get_ancestors(node_id: str, parent_map: dict[str, str | None]) -> list[str]:
    """Get path from root to node (inclusive), in root-first order."""
    path = []
    current = node_id
    while current is not None:
        path.append(current)
        current = parent_map.get(current)
    return list(reversed(path))


def build_species_tree_design(
    species_tree: str | tuple,
    observed_species: list[str],
) -> SpeciesTreeDesign:
    """Build species path-indicator design matrix from a tree.

    Parameters
    ----------
    species_tree : Newick string or nested tuple
    observed_species : list of species labels that must be leaves in the tree
    """
    if isinstance(species_tree, str):
        parsed = _parse_newick(species_tree)
    else:
        parsed = species_tree

    nodes: list[dict] = []
    counter = [0]
    _collect_nodes(parsed, None, nodes, counter)
    node_table = pd.DataFrame(nodes)

    parent_map = {row["node_id"]: row["parent_id"] for _, row in node_table.iterrows()}
    leaf_ids = set(node_table[node_table["is_leaf"]]["node_id"])

    missing = set(observed_species) - leaf_ids
    if missing:
        raise ValueError(f"Species not found in tree: {missing}")

    all_node_ids = list(node_table["node_id"].values)
    node_to_idx = {nid: i for i, nid in enumerate(all_node_ids)}

    n_species = len(observed_species)
    n_nodes = len(all_node_ids)
    rows, cols, data = [], [], []

    for sp_idx, sp in enumerate(observed_species):
        ancestors = _get_ancestors(sp, parent_map)
        for anc in ancestors:
            rows.append(sp_idx)
            cols.append(node_to_idx[anc])
            data.append(1.0)

    A_species = sparse.csr_matrix(
        (data, (rows, cols)), shape=(n_species, n_nodes), dtype=np.float64
    )

    return SpeciesTreeDesign(
        A_species=A_species,
        species_order=observed_species,
        node_ids=all_node_ids,
        node_table=node_table,
    )
