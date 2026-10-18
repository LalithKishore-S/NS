"""Leakage-safe splitting and metrics (plan §9 Stage 6, §12).

Every split is by graph_id: all cascades of one graph (every p, every seed) stay on one side.
"""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, average_precision_score


def group_folds(df, y, n_splits=5, seed=0):
    """Stratified K-fold over GRAPHS. Yields (train_idx, test_idx)."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in cv.split(df, y, groups=df.graph_id):
        assert not set(df.graph_id.iloc[tr]) & set(df.graph_id.iloc[te]), "graph leaked across split"
        yield tr, te


def multiclass_metrics(y_true, y_pred) -> dict:
    return {"macro_f1": f1_score(y_true, y_pred, average="macro"),
            "balanced_acc": balanced_accuracy_score(y_true, y_pred)}


def binary_metrics(y_true, prob, thr=0.5) -> dict:
    if len(np.unique(y_true)) < 2:
        return {"roc_auc": np.nan, "pr_auc": np.nan, "macro_f1": np.nan, "balanced_acc": np.nan}
    pred = (prob >= thr).astype(int)
    return {"roc_auc": roc_auc_score(y_true, prob), "pr_auc": average_precision_score(y_true, prob),
            "macro_f1": f1_score(y_true, pred, average="macro"),
            "balanced_acc": balanced_accuracy_score(y_true, pred)}


def mean_ci(x) -> tuple[float, float]:
    """Mean and 95% normal-approx half-width over seeds."""
    x = np.asarray(x, float)
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
