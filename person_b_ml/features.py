"""Person B entry point: read trajectories, write features_{h}.parquet per horizon (joined with labels)."""
import pandas as pd
from common import paths
from common.contracts import ExperimentConfig
from person_b_ml.feature_extraction import extract


def extract_features(cfg: ExperimentConfig) -> None:
    d = paths.artifact_dir(cfg.run_id)
    traj = pd.read_parquet(d / paths.TRAJECTORIES)
    outc = pd.read_parquet(d / paths.OUTCOMES)[["graph_id", "seed", "p", "family", "collapsed", "S_final"]]
    for h in cfg.horizons:
        f = extract(traj, h).merge(outc, on=["graph_id", "seed", "p"], validate="one_to_one")
        f.to_parquet(d / paths.features_file(h))
