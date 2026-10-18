"""Stage 4: early-observation features at a fixed horizon h (half-steps).

Uses ONLY rows with half-step <= h (leakage rule).  Inputs are the aggregate failure curves of
trajectories.parquet — no adjacency, no dependency map, no node ids.

Feature groups (plan §9 Stage 4):
  state     : p, failed_A, failed_B                               -> Model A (state-only baseline)
  temporal  : rates, acceleration, active steps, stopped, EWS     -> Scheffer 2009 / Dakos 2012 / Deb 2022
  cross     : layer-B vs layer-A failure balance                  -> interdependence
  timing    : distribution of failure times within the window     -> node-resolved (aggregated)
"""
import numpy as np
import pandas as pd

STATE = ["p", "failed_A", "failed_B"]
TEMPORAL = ["rate_last", "rate_mean", "accel", "n_active", "stopped",
            "ews_var", "ews_ac1", "ews_reliable"]
# NOTE: failed_B - failed_A is identically 0 at even h when dependency_fraction = 1 (after each full round
# the two layers' alive sets mirror each other), so cross-layer information lives in the half-step split.
CROSS = ["cross_ratio_last", "cross_share_B"]
TIMING = ["t_mean", "t_std", "conc_last"]
GROUPS = {"state": STATE, "temporal": TEMPORAL, "cross": CROSS, "timing": TIMING}
ALL = STATE + TEMPORAL + CROSS + TIMING
MIN_EWS_POINTS = 4   # fewer points -> variance / lag-1 AC flagged unreliable (plan §9)


def _lag1_ac(x: np.ndarray) -> float:
    if len(x) < 3 or np.std(x[:-1]) == 0 or np.std(x[1:]) == 0:
        return 0.0
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def features_one(traj: pd.DataFrame, p: float, h: int) -> dict:
    """traj: rows of ONE cascade, any length. Only h' <= h is used; missing steps = cascade already stable."""
    t = traj[traj.h <= h].sort_values("h")
    nA = np.zeros(h + 1); nB = np.zeros(h + 1)
    nA[t.h.values] = t.new_fail_A.values
    nB[t.h.values] = t.new_fail_B.values
    stopped = float(len(t) < h + 1 or (nA[-1] == 0 and nB[-1] == 0))
    cas_A = nA.copy(); cas_A[0] = 0                     # cascade failures (exclude the attack itself)
    tot = cas_A + nB                                    # all cascade failures per half-step
    series = tot[1:]                                    # failure-rate time series after the attack
    failed_A, failed_B = nA.sum(), nB.sum()
    rate_last = series[-1] if len(series) else 0.0
    rates = series
    accel = float(rates[-1] - rates[-2]) if len(rates) >= 2 else 0.0
    w = np.arange(1, h + 1)
    mass = tot[1:].sum()
    return {
        "p": p, "failed_A": failed_A, "failed_B": failed_B,
        "rate_last": rate_last, "rate_mean": float(rates.mean()) if len(rates) else 0.0, "accel": accel,
        "n_active": float((series > 0).sum()), "stopped": stopped,
        "ews_var": float(np.var(series)), "ews_ac1": _lag1_ac(series),
        "ews_reliable": float(len(series) >= MIN_EWS_POINTS),
        "cross_ratio_last": float(nB[-1] / (nA[-1] + nB[-1])) if (nA[-1] + nB[-1]) > 0 else 0.0,
        "cross_share_B": float(nB.sum() / mass) if mass > 0 else 0.0,
        "t_mean": float((w * tot[1:]).sum() / mass) if mass > 0 else 0.0,
        "t_std": float(np.sqrt((((w - (w * tot[1:]).sum() / mass) ** 2) * tot[1:]).sum() / mass)) if mass > 0 else 0.0,
        "conc_last": float(tot[-1] / mass) if mass > 0 else 0.0,
    }


def extract(traj: pd.DataFrame, h: int) -> pd.DataFrame:
    rows = []
    for (gid, seed, p), d in traj.groupby(["graph_id", "seed", "p"], sort=False):
        rows.append({"graph_id": gid, "seed": seed, **features_one(d, p, h), "h": h})
    return pd.DataFrame(rows)
