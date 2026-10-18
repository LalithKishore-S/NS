"""Stage 3: survival-vs-attack sweeps, p_c estimation, early-warning window (risk R1).

p_c definition: attack fraction where P(collapse) = 0.5 (linear interpolation).
Sharpness: 1 / (p_90 - p_10), the inverse width of the window where P(collapse) goes 10% -> 90%.
Lead time (R1): half-steps a collapsing cascade spends ABOVE the collapse threshold, i.e.
how long it still "looks alive" and can be observed before it is obviously dead.
"""
import numpy as np
import pandas as pd

from person_a_sim.network_generation import make_graph
from person_a_sim.interdependency import couple
from person_a_sim.cascade_simulator import simulate, to_csr


def alive_curve(fail_t: np.ndarray) -> np.ndarray:
    """Fraction alive after each half-step h = 0..H (from per-node failure times)."""
    H = max(int(fail_t.max()), 0)
    return np.array([np.mean((fail_t < 0) | (fail_t > h)) for h in range(H + 1)])


def lead_time(fail_t_A: np.ndarray, threshold: float) -> int:
    c = alive_curve(fail_t_A)
    below = np.nonzero(c < threshold)[0]
    return int(below[0]) if len(below) else len(c)


def sweep(family, n_graphs, grid, n_seeds=3, n=1000, k=4.0, q=1.0, threshold=0.25, base_seed=0):
    rows = []
    for g in range(n_graphs):
        s = base_seed + 1000 * g
        A, B = to_csr(make_graph(family, n, k, s)), to_csr(make_graph(family, n, k, s + 1))
        dep = couple(n, q, seed=s + 2)
        for p in grid:
            for r in range(n_seeds):
                out = simulate(A, B, dep, p, seed=s + 3 + r)
                rows.append({"family": family, "graph_id": f"{family}_{g}", "n": n, "p": round(p, 4),
                             "seed": r, "S_final": out["S_final"], "rounds": out["n_iter"],
                             "half_steps": out["n_half"], "collapsed": out["S_final"] < threshold,
                             "lead": lead_time(out["fail_t_A"], threshold)})
    return pd.DataFrame(rows)


def crossing(p, y, level):
    """First p where y crosses `level` (y increasing in p), linear interpolation."""
    idx = np.nonzero(y >= level)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    if i == 0:
        return p[0]
    return p[i - 1] + (level - y[i - 1]) * (p[i] - p[i - 1]) / (y[i] - y[i - 1])


def estimate_pc(df: pd.DataFrame) -> dict:
    curve = df.groupby("p")["collapsed"].mean()
    p, y = curve.index.values, np.maximum.accumulate(curve.values)   # enforce monotone
    pc, p10, p90 = crossing(p, y, .5), crossing(p, y, .1), crossing(p, y, .9)
    return {"p_c": pc, "p10": p10, "p90": p90, "sharpness": 1 / max(p90 - p10, 1e-9)}


def bootstrap_pc(df: pd.DataFrame, n_boot=200, seed=0) -> tuple[float, float]:
    """95% CI of p_c, resampling whole graphs (graph instance is the unit)."""
    rng = np.random.default_rng(seed)
    gids = df.graph_id.unique()
    groups = {g: d for g, d in df.groupby("graph_id")}
    pcs = [estimate_pc(pd.concat([groups[g] for g in rng.choice(gids, len(gids))]))["p_c"] for _ in range(n_boot)]
    return tuple(np.nanpercentile(pcs, [2.5, 97.5]))
