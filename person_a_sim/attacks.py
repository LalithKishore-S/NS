"""Initial-attack strategies: order layer-A nodes worst-first for a given strategy.
The attack is part of the ENVIRONMENT (how damage starts), never a model feature - no leakage."""
import numpy as np
import networkx as nx


def attack_order(g: nx.Graph, strategy: str, seed: int = 0) -> np.ndarray:
    """Return node indices in the order they should be attacked (first = hit first)."""
    n = g.number_of_nodes()
    rng = np.random.default_rng(seed)
    if strategy == "random":
        return rng.permutation(n)
    if strategy == "degree":
        deg = np.array([d for _, d in g.degree()], dtype=float)
        return np.argsort(-(deg + rng.random(n) * 1e-6))
    if strategy == "betweenness":
        bc = nx.betweenness_centrality(g, k=min(n, 200), seed=int(seed))
        return np.argsort(-np.array([bc[i] for i in range(n)]))
    if strategy == "kcore":
        core = nx.core_number(g)
        return np.argsort(-np.array([core[i] for i in range(n)], dtype=float))
    raise ValueError(f"unknown attack strategy {strategy!r}")
