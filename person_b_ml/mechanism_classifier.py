"""Stage 5 / E2: predict the hidden generating family (ER/BA/WS/SBM) from ONE partial cascade.

Evaluated held-out-INSTANCE only (plan §10: a 4-class model cannot be tested on a family it never saw).
Out-of-fold probabilities are also returned: Model C (Step 6) uses them as 'inferred topology'.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from common.contracts import FAMILIES
from person_b_ml.evaluation import group_folds, multiclass_metrics
from person_b_ml.feature_extraction import ALL
from person_b_ml.models import make_model

MECH_FEATURES = [f for f in ALL if f != "ews_reliable"]   # constant within a horizon


def run_mechanism(feat: pd.DataFrame, model: str, seed: int, features=MECH_FEATURES):
    """Returns (metrics dict, out-of-fold probability DataFrame with columns prob_ER..prob_SBM)."""
    le = LabelEncoder().fit(list(FAMILIES))
    y = le.transform(feat.family)
    X = feat[features].values
    oof = np.zeros((len(feat), len(FAMILIES)))
    for tr, te in group_folds(feat, y, n_splits=5, seed=seed):
        m = make_model(model, seed, n_classes=4).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])
    pred = oof.argmax(1)
    probs = pd.DataFrame(oof, columns=[f"prob_{c}" for c in le.classes_], index=feat.index)
    return multiclass_metrics(y, pred), probs, le.inverse_transform(pred)
