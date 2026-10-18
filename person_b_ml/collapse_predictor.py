"""Stage 5-6 / E3-E5: three nested collapse models (plan §11).

  Model A  state-only        : p, failed_A, failed_B                      (mandatory baseline)
  Model B  dynamics-aware    : A + temporal + EWS + cross-layer + timing
  Model C  topology-informed : B + PREDICTED family probabilities (never the true family — it is hidden)

Model C leakage control: the mechanism classifier inside C is fitted on the TRAINING graphs only.
Training rows get cross-fitted (out-of-fold) probabilities, so C never learns from over-confident in-sample
probabilities; test rows get probabilities from a classifier fitted on all training graphs.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from person_b_ml.evaluation import group_folds, binary_metrics
from person_b_ml.feature_extraction import STATE, TEMPORAL, CROSS, TIMING
from person_b_ml.models import make_model

DYN = [f for f in TEMPORAL if f != "ews_reliable"] + CROSS + TIMING
FEATURE_SETS = {"A_state": STATE, "B_dynamics": STATE + DYN, "C_topology": STATE + DYN}   # C adds probs
MECH_INPUT = [f for f in STATE + DYN]


def _family_probs(train: pd.DataFrame, test: pd.DataFrame, seed: int):
    """Cross-fitted family probabilities for train rows, fitted-on-all-train probabilities for test rows.
    Classes = families present in TRAIN (3 under leave-one-family-out, 4 otherwise)."""
    le = LabelEncoder().fit(train.family)
    y = le.transform(train.family)
    k = len(le.classes_)
    Xtr, Xte = train[MECH_INPUT].values, test[MECH_INPUT].values
    p_tr = np.zeros((len(train), k))
    for itr, ite in group_folds(train.reset_index(drop=True), y, n_splits=3, seed=seed):
        p_tr[ite] = make_model("RF", seed, k).fit(Xtr[itr], y[itr]).predict_proba(Xtr[ite])
    p_te = make_model("RF", seed, k).fit(Xtr, y).predict_proba(Xte)
    cols = [f"prob_{c}" for c in le.classes_]
    return pd.DataFrame(p_tr, columns=cols), pd.DataFrame(p_te, columns=cols)


def fit_predict_all(train: pd.DataFrame, test: pd.DataFrame, models, seed: int) -> dict:
    """Fit every (feature set, model) on one split. Family probabilities for C are computed ONCE per split."""
    ptr, pte = _family_probs(train, test, seed)
    y = train.collapsed.astype(int).values
    out = {}
    for ms, feats in FEATURE_SETS.items():
        Xtr, Xte = train[feats].reset_index(drop=True), test[feats].reset_index(drop=True)
        if ms == "C_topology":
            Xtr, Xte = pd.concat([Xtr, ptr], axis=1), pd.concat([Xte, pte], axis=1)
        for m in models:
            out[(ms, m)] = make_model(m, seed, 2).fit(Xtr.values, y).predict_proba(Xte.values)[:, 1]
    return out


def eval_masks(df: pd.DataFrame, threshold: float, pc: dict, window: float = 0.04) -> dict:
    """ALL cases; UND = not yet visibly collapsed at h (D8); CRIT = UND and within ±window of the family's p_c (D9).
    p_c is used only to choose WHICH cases are scored, never as a model input."""
    und = ((1 - df.failed_A) >= threshold).values
    crit = und & (np.abs(df.p - df.family.map(pc)) <= window).values
    return {"all": np.ones(len(df), bool), "und": und, "crit": crit}


def score(test: pd.DataFrame, prob: np.ndarray, masks: dict) -> dict:
    y = test.collapsed.astype(int).values
    out = {}
    for name, mk in masks.items():
        out.update({f"{name}_{k}": v for k, v in binary_metrics(y[mk], prob[mk]).items()})
        out[f"{name}_n"], out[f"{name}_prev"] = int(mk.sum()), float(y[mk].mean()) if mk.any() else np.nan
    return out


def within_distribution(feat: pd.DataFrame, models, seed: int) -> dict:
    """5-fold grouped CV over graphs. Returns {(set, model): out-of-fold probabilities aligned with feat}."""
    feat = feat.reset_index(drop=True)
    probs = {}
    for tr, te in group_folds(feat, feat.collapsed.astype(int), n_splits=5, seed=seed):
        for key, p in fit_predict_all(feat.iloc[tr], feat.iloc[te], models, seed).items():
            probs.setdefault(key, np.zeros(len(feat)))[te] = p
    return probs


def leave_one_family_out(feat: pd.DataFrame, held_out: str, models, seed: int):
    """Train on the other three families' graphs, test on every graph of `held_out`.
    (D1: both layers share a family, so 'either layer is the held-out family' == family == held_out.)"""
    tr, te = feat[feat.family != held_out], feat[feat.family == held_out]
    assert held_out not in set(tr.family)
    return te.reset_index(drop=True), fit_predict_all(tr, te, models, seed)
