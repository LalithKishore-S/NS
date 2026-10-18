"""The three classical models used everywhere (plan §3): LR, RF, XGBoost. CPU only."""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

MODEL_NAMES = ["LR", "RF", "XGB"]


def make_model(name: str, seed: int = 0, n_classes: int = 2):
    if name == "LR":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    if name == "RF":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    if name == "XGB":
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, n_jobs=2, random_state=seed, verbosity=0,
                             objective="binary:logistic" if n_classes == 2 else "multi:softprob")
    raise ValueError(name)
