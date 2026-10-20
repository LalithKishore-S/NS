"""Stage 2: interdependent percolation cascade (Buldyrev et al., Nature 2010).

Rules (Buldyrev 2010, Fig. 2):
  t=0  remove a random fraction p of layer-A nodes.
  each iteration t = 1, 2, ...:
    (a) A-nodes outside A's giant component fail; their B-partners fail.
    (b) B-nodes outside B's giant component fail; their A-partners fail.
  stop when an iteration changes nothing.  Survivors = mutually connected giant component.
Failures are irreversible.  We record the iteration at which every node failed.
"""
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from person_a_sim.interdependency import inverse


def to_csr(g: nx.Graph) -> csr_matrix:
    return nx.to_scipy_sparse_array(g, nodelist=range(g.number_of_nodes()), format="csr")


def giant_mask(adj: csr_matrix, alive: np.ndarray) -> np.ndarray:
    """Boolean mask of nodes in the largest connected component of the alive subgraph."""
    idx = np.nonzero(alive)[0]
    out = np.zeros_like(alive)
    if len(idx) == 0:
        return out
    _, lab = connected_components(adj[idx][:, idx], directed=False)
    out[idx[lab == np.bincount(lab).argmax()]] = True
    return out


def simulate(adj_A, adj_B, dep, p: float, seed: int = 0, order=None, max_iter: int = 10_000) -> dict:
    """Run one cascade. Returns failure times and the per-iteration curve.

    fail_t_A[i] = HALF-STEP at which A-node i failed, -1 = survived.
      0 = initial attack, 2t-1 = during A-pruning of round t, 2t = during B-pruning of round t.
      Round of failure = ceil(fail_t / 2).  Half-steps double time resolution (R1).
    """
    n = adj_A.shape[0]
    inv = inverse(dep)
    rng = np.random.default_rng(seed)
    alive_A = np.ones(n, bool)
    alive_B = np.ones(n, bool)
    fail_A = np.full(n, -1, np.int64)
    fail_B = np.full(n, -1, np.int64)

    k = int(round(p * n))
    hit = np.asarray(order)[:k] if order is not None else rng.choice(n, k, replace=False)
    alive_A[hit] = False
    fail_A[hit] = 0
    curve = [(0, alive_A.mean(), alive_B.mean())]

    for t in range(1, max_iter + 1):
        before = alive_A.sum() + alive_B.sum()
        # (a) A pruning, then dependency A -> B
        alive_A &= giant_mask(adj_A, alive_A)
        dead_A = ~alive_A & (dep >= 0)
        alive_B[dep[dead_A]] = False
        fail_A[(~alive_A) & (fail_A < 0)] = 2 * t - 1      # half-step resolution
        fail_B[(~alive_B) & (fail_B < 0)] = 2 * t - 1
        # (b) B pruning, then dependency B -> A
        alive_B &= giant_mask(adj_B, alive_B)
        dead_B = ~alive_B & (inv >= 0)
        alive_A[inv[dead_B]] = False

        fail_A[(~alive_A) & (fail_A < 0)] = 2 * t
        fail_B[(~alive_B) & (fail_B < 0)] = 2 * t
        if alive_A.sum() + alive_B.sum() == before:   # no change -> stable
            break
        curve.append((t, alive_A.mean(), alive_B.mean()))

    return {
        "fail_t_A": fail_A, "fail_t_B": fail_B,
        "curve": np.array(curve),              # columns: iteration, frac_alive_A, frac_alive_B
        "n_iter": int(curve[-1][0]),          # rounds
        "n_half": int(max(fail_A.max(), fail_B.max())),  # last half-step with a failure
        "S_final": float(alive_A.mean()),       # mutual giant component size (A side)
    }
