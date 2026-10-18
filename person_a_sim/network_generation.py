"""Stage 1: ER / BA / WS / SBM graphs at matched N and average degree <k>."""
import networkx as nx
import numpy as np


def make_graph(family: str, n: int = 1000, k: float = 4.0, seed: int = 0) -> nx.Graph:
    if family == "ER":   # exactly n*k/2 edges -> <k> exactly k
        g = nx.gnm_random_graph(n, int(n * k / 2), seed=seed)
    elif family == "BA":  # each new node attaches m edges -> <k> ~ 2m
        g = nx.barabasi_albert_graph(n, int(round(k / 2)), seed=seed)
    elif family == "WS":  # ring lattice, each node k neighbours, 10% rewired
        g = nx.watts_strogatz_graph(n, int(k), 0.1, seed=seed)
    elif family == "SBM":  # 4 communities, 90% of edges inside a community
        b = 4
        size = n // b
        p_in = 0.9 * k / (size - 1)
        p_out = 0.1 * k / (n - size)
        probs = [[p_in if i == j else p_out for j in range(b)] for i in range(b)]
        g = nx.stochastic_block_model([size] * b, probs, seed=seed)
        g = nx.Graph(g)  # drop block metadata
    else:
        raise ValueError(family)
    return g


def avg_degree(g: nx.Graph) -> float:
    return 2 * g.number_of_edges() / g.number_of_nodes()


def graph_stats(g: nx.Graph) -> dict:
    degs = np.array([d for _, d in g.degree()])
    gcc = max(nx.connected_components(g), key=len)
    return {
        "avg_degree": avg_degree(g),
        "max_degree": int(degs.max()),
        "degree_std": float(degs.std()),
        "clustering": nx.average_clustering(g),
        "gcc_frac": len(gcc) / g.number_of_nodes(),
    }
