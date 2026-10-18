"""Read-only graph schema an input adapter exposes to Stage 1: node types, typed edges and canonical target IDs."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GraphSchema:
    node_counts: dict      # node type -> number of rows
    edges: dict            # (src_type, edge_type, dst_type) -> (src_rows, dst_rows) in canonical orientation
    target_type: str
    target_ids: tuple      # canonical target ID (the ID run_comce.py accepts) of every target-type row

    def relation(self, triple):
        triple = tuple(map(str, triple))
        if triple not in self.edges:
            raise KeyError(f'relation {list(triple)} is not in the graph; available: {[list(t) for t in self.edges]}')
        source, destination = self.edges[triple]
        return np.asarray(source, np.int64), np.asarray(destination, np.int64)

    def target_rows(self, ids):
        """Rows of canonical target IDs; IDs that name no vertex of this graph are returned separately."""
        index = {c: i for i, c in enumerate(self.target_ids)}
        rows, unresolved = [], []
        for c in map(str, ids):
            (rows.append(index[c]) if c in index else unresolved.append(c))
        return np.asarray(rows, np.int64), unresolved
