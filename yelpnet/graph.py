"""Whole-network stats on a sparse matrix (scales to millions of edges) and subgraph sampling."""
import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components


def adjacency(edges, n):
    a = coo_matrix((np.ones(len(edges), np.int8), edges.T), shape=(n, n)).tocsr()
    return a + a.T


def degrees(a):
    return np.diff(a.indptr)


def ccdf(values):
    """P(X > k) at every distinct value k."""
    k, cnt = np.unique(values, return_counts=True)
    return k, 1.0 - np.cumsum(cnt) / len(values)


def log_binned_pk(deg, nbins=30):
    """P(k) with geometrically growing bins, returned at the geometric bin centres."""
    edges = np.unique(np.floor(np.logspace(0, np.log10(deg.max() + 1), nbins)))
    cnt, _ = np.histogram(deg, bins=edges)
    keep = cnt > 0
    return np.sqrt(edges[:-1] * edges[1:])[keep], (cnt / np.diff(edges) / deg.size)[keep]


def powerlaw_gamma(deg, kmin):
    """Discrete MLE of the power-law exponent on k >= kmin (Clauset, Shalizi, Newman 2009, eq. 3.7)."""
    tail = deg[deg >= kmin]
    return float(1 + tail.size / np.log(tail / (kmin - 0.5)).sum())


def summary(a, kmin):
    deg = degrees(a)
    active = deg > 0  # users with at least one friend
    _, labels = connected_components(a, directed=False)
    sizes = np.bincount(labels[active])
    n = int(active.sum())
    return {
        "nodes": n,
        "edges": int(a.nnz // 2),
        "density": a.nnz / (n * (n - 1)),
        "mean_degree": float(deg[active].mean()),
        "median_degree": float(np.median(deg[active])),
        "max_degree": int(deg.max()),
        f"powerlaw_gamma_k>={kmin}": powerlaw_gamma(deg[active], kmin),
        "components": int((sizes > 0).sum()),
        "largest_component_frac": float(sizes.max() / n),
        "second_component_size": int(np.sort(sizes)[-2]),
    }


def snowball(a, n, seed=0):
    """Subgraph induced by the first n nodes reached by BFS from a random node in the largest component."""
    _, labels = connected_components(a, directed=False)
    deg = degrees(a)
    lcc = np.bincount(labels[deg > 0]).argmax()
    start = np.random.default_rng(seed).choice(np.flatnonzero(labels == lcc))
    nodes = breadth_first_order(a, start, directed=False, return_predecessors=False)[:n]
    return nx.from_scipy_sparse_array(a[nodes][:, nodes])
