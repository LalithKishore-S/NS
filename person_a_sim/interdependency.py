"""Couple two layers with dependency links.

Buldyrev et al., Nature 2010  -> one-to-one random bijection (q = 1).
Parshani, Buldyrev & Havlin, PRL 2010 -> only a fraction q of nodes are dependent.
"""
import numpy as np


def couple(n: int, dependency_fraction: float = 1.0, seed: int = 0) -> np.ndarray:
    """Return dep[i] = index of B-node that A-node i depends on (and vice versa), or -1.

    Links are bidirectional and one-to-one (no node has two partners).
    """
    rng = np.random.default_rng(seed)
    dep = np.full(n, -1, dtype=np.int64)
    perm = rng.permutation(n)                # random A->B bijection
    n_dep = int(round(dependency_fraction * n))
    a_nodes = rng.choice(n, n_dep, replace=False)
    dep[a_nodes] = perm[a_nodes]
    return dep


def inverse(dep: np.ndarray) -> np.ndarray:
    """inv[j] = A-node that B-node j depends on, or -1."""
    inv = np.full(len(dep), -1, dtype=np.int64)
    has = dep >= 0
    inv[dep[has]] = np.nonzero(has)[0]
    return inv
