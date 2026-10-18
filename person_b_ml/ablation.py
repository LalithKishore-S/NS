"""Stage 6 / E6: feature-group ablations and graph-level bootstrap uncertainty.

Ablation: start from Model B (state + all dynamics) and remove ONE group at a time; also state + ONE group.
Bootstrap: resample GRAPHS with replacement (the graph instance is the unit of independence, plan §12),
recompute the metric on the resampled cases -> 95% CI that reflects which graphs were drawn.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from person_b_ml.evaluation import group_folds
from person_b_ml.feature_extraction import STATE, TEMPORAL, CROSS, TIMING
from person_b_ml.models import make_model

EWS = ["ews_var", "ews_ac1"]
RATE = [f for f in TEMPORAL if f not in EWS + ["ews_reliable"]]
DYN_GROUPS = {"rate": RATE, "ews": EWS, "cross": CROSS, "timing": TIMING}
B_FULL = STATE + RATE + EWS + CROSS + TIMING


def ablation_sets() -> dict:
    sets = {"A_state": STATE, "B_full": B_FULL}
    for g, cols in DYN_GROUPS.items():
        sets[f"B_minus_{g}"] = [c for c in B_FULL if c not in cols]
        sets[f"state_plus_{g}"] = STATE + cols
    return sets


def oof_within(feat, sets, model, seed):
    feat = feat.reset_index(drop=True)
    y = feat.collapsed.astype(int).values
    out = {k: np.zeros(len(feat)) for k in sets}
    for tr, te in group_folds(feat, y, n_splits=5, seed=seed):
        for k, cols in sets.items():
            out[k][te] = make_model(model, seed).fit(feat.iloc[tr][cols].values, y[tr]).predict_proba(feat.iloc[te][cols].values)[:, 1]
    return out


def lofo(feat, held, sets, model, seed):
    tr, te = feat[feat.family != held], feat[feat.family == held].reset_index(drop=True)
    y = tr.collapsed.astype(int).values
    return te, {k: make_model(model, seed).fit(tr[c].values, y).predict_proba(te[c].values)[:, 1] for k, c in sets.items()}


def bootstrap_auc(df: pd.DataFrame, prob: np.ndarray, mask: np.ndarray, n_boot=1000, seed=0):
    """Graph-level bootstrap of ROC-AUC over the cases selected by mask. Returns (auc, lo, hi)."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"g": df.graph_id.values, "y": df.collapsed.astype(int).values, "p": prob})[mask]
    by_g = {g: x for g, x in d.groupby("g")}
    gids = np.array(list(by_g))
    point = roc_auc_score(d.y, d.p)
    stats = []
    for _ in range(n_boot):
        s = pd.concat([by_g[g] for g in rng.choice(gids, len(gids))])
        if s.y.nunique() == 2:
            stats.append(roc_auc_score(s.y, s.p))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return point, lo, hi


def bootstrap_diff(df, prob_a, prob_b, mask, n_boot=1000, seed=0):
    """Paired graph bootstrap of AUC(b) - AUC(a). Returns (diff, lo, hi, P(diff <= 0))."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"g": df.graph_id.values, "y": df.collapsed.astype(int).values, "a": prob_a, "b": prob_b})[mask]
    by_g = {g: x for g, x in d.groupby("g")}
    gids = np.array(list(by_g))
    diffs = []
    for _ in range(n_boot):
        s = pd.concat([by_g[g] for g in rng.choice(gids, len(gids))])
        if s.y.nunique() == 2:
            diffs.append(roc_auc_score(s.y, s.b) - roc_auc_score(s.y, s.a))
    diffs = np.array(diffs)
    return roc_auc_score(d.y, d.b) - roc_auc_score(d.y, d.a), *np.percentile(diffs, [2.5, 97.5]), float((diffs <= 0).mean())
