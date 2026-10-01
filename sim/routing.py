"""Graph utilities for fast vectorized routing.

The drive graph can have parallel edges (same u, v). scipy's csgraph needs one
entry per (u, v) pair, so edges are grouped into *pairs*. For each cost vector
we pick the cheapest edge of every pair, build a CSR matrix on pairs and run
``scipy.sparse.csgraph.dijkstra`` from many roots at once.

Paths are extracted from predecessor arrays for many legs at once (one numpy
step per edge along the path), producing a padded ``(n_legs, L)`` matrix of
edge indices with ``-1`` padding.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

MIN_COST = 1e-3  # seconds; csgraph treats explicit zeros inconsistently, keep costs positive


class PairGraph:
    """Fixed structure of the directed graph collapsed to unique (u, v) pairs."""

    def __init__(self, n_nodes: int, eu: np.ndarray, ev: np.ndarray) -> None:
        self.n = int(n_nodes)
        self.eu = eu.astype(np.int64)
        self.ev = ev.astype(np.int64)
        keys = self.eu * self.n + self.ev
        uniq, inv = np.unique(keys, return_inverse=True)
        self.pair_keys = uniq  # sorted
        self.pair_of_edge = inv.astype(np.int32)
        self.n_pairs = len(uniq)
        pu = (uniq // self.n).astype(np.int32)
        pv = (uniq % self.n).astype(np.int32)
        self.pair_u = pu
        self.pair_v = pv
        # CSR structure (rows = u), pairs already sorted by (u, v)
        self.indptr = np.zeros(self.n + 1, dtype=np.int32)
        np.add.at(self.indptr, pu + 1, 1)
        self.indptr = np.cumsum(self.indptr).astype(np.int32)
        self.indices = pv
        # transposed CSR structure (rows = v)
        self.t_perm = np.lexsort((pu, pv))
        self.t_indptr = np.zeros(self.n + 1, dtype=np.int32)
        np.add.at(self.t_indptr, pv + 1, 1)
        self.t_indptr = np.cumsum(self.t_indptr).astype(np.int32)
        self.t_indices = pu[self.t_perm]
        # edges sorted by pair for best-edge selection
        self._edge_order = np.argsort(self.pair_of_edge, kind="stable")
        self._pair_start = np.searchsorted(self.pair_of_edge[self._edge_order], np.arange(self.n_pairs))
        self.has_parallel = len(eu) != self.n_pairs

    def best_edges(self, edge_cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (pair_cost, pair_best_edge) for an edge cost vector."""
        if not self.has_parallel:
            order = self._edge_order
            return np.maximum(edge_cost[order], MIN_COST), order.astype(np.int32)
        order = np.lexsort((edge_cost, self.pair_of_edge))
        first = np.searchsorted(self.pair_of_edge[order], np.arange(self.n_pairs))
        best = order[first].astype(np.int32)
        return np.maximum(edge_cost[best], MIN_COST), best

    def matrices(self, pair_cost: np.ndarray) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        fwd = sp.csr_matrix((pair_cost, self.indices, self.indptr), shape=(self.n, self.n))
        rev = sp.csr_matrix((pair_cost[self.t_perm], self.t_indices, self.t_indptr), shape=(self.n, self.n))
        return fwd, rev

    def pair_index(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        keys = a.astype(np.int64) * self.n + b.astype(np.int64)
        return np.searchsorted(self.pair_keys, keys)


def trees(matrix: sp.csr_matrix, roots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dijkstra from many roots. Returns (dist float32 (k, N), pred int32 (k, N))."""
    if len(roots) == 0:
        n = matrix.shape[0]
        return np.zeros((0, n), np.float32), np.zeros((0, n), np.int32)
    dist, pred = dijkstra(matrix, directed=True, indices=np.asarray(roots, dtype=np.int64), return_predecessors=True)
    return dist.astype(np.float32), pred.astype(np.int32)


def walk_to_root(
    pred_flat: np.ndarray,
    n_nodes: int,
    rows: np.ndarray,
    start: np.ndarray,
    root: np.ndarray,
    graph: PairGraph,
    best_edge: np.ndarray,
    best_row: np.ndarray,
    forward: bool,
    max_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Follow predecessor pointers from ``start`` until ``root`` for many legs.

    ``forward=True``: tree on the reversed graph (rooted at the destination),
    pred gives the next node towards the root; edges come out in travel order.
    ``forward=False``: tree on the forward graph rooted at the origin, walking
    back from the destination; edges come out reversed and are flipped here.

    ``best_edge`` is a 2D array (n_cost_sets, n_pairs); ``best_row`` selects the
    cost set per leg (routing period). Returns (paths (n, L) int32, lengths, ok).
    """
    n = len(start)
    cur = start.astype(np.int64).copy()
    lengths = np.zeros(n, dtype=np.int32)
    ok = np.ones(n, dtype=bool)
    cols: list[np.ndarray] = []
    active = np.nonzero(cur != root)[0]
    base = rows.astype(np.int64) * n_nodes
    step = 0
    while len(active) and step < max_steps:
        c = cur[active]
        nxt = pred_flat[base[active] + c]
        bad = nxt < 0
        if bad.any():
            ok[active[bad]] = False
            active = active[~bad]
            c = c[~bad]
            nxt = nxt[~bad]
        if forward:
            pidx = graph.pair_index(c, nxt)
        else:
            pidx = graph.pair_index(nxt, c)
        e = best_edge[best_row[active], pidx]
        col = np.full(n, -1, dtype=np.int32)
        col[active] = e
        cols.append(col)
        lengths[active] += 1
        cur[active] = nxt
        active = active[nxt != root[active]]
        step += 1
    if len(active):
        ok[active] = False
    if not cols:
        return np.full((n, 0), -1, dtype=np.int32), lengths, ok
    paths = np.stack(cols, axis=1)
    if not forward:
        paths = reverse_rows(paths, lengths)
    return paths, lengths, ok


def reverse_rows(paths: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    n, L = paths.shape
    if L == 0:
        return paths
    j = np.arange(L)[None, :]
    src = lengths[:, None] - 1 - j
    valid = src >= 0
    out = np.full_like(paths, -1)
    rr = np.broadcast_to(np.arange(n)[:, None], (n, L))
    out[valid] = paths[rr[valid], src[valid]]
    return out


def concat_paths(parts: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate per-row padded path segments. parts = [(paths, lengths), ...]."""
    n = len(parts[0][1])
    total = np.zeros(n, dtype=np.int32)
    for _, ln in parts:
        total += ln
    L = int(total.max()) if n else 0
    out = np.full((n, max(L, 0)), -1, dtype=np.int32)
    offset = np.zeros(n, dtype=np.int32)
    for paths, ln in parts:
        if paths.shape[1] == 0:
            continue
        j = np.arange(paths.shape[1])[None, :]
        valid = j < ln[:, None]
        rr, cc = np.nonzero(valid)
        out[rr, offset[rr] + cc] = paths[rr, cc]
        offset += ln
    return out, total
