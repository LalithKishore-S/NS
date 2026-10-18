import pandas as pd
from common.contracts import ExperimentConfig
from common.dummy import make_dummy_artifacts
from common import paths
from person_b_ml.features import extract_features

def test_dummy_roundtrip_through_features():
    cfg = ExperimentConfig(run_id="pytest_dummy", n_graphs_per_family=2, n_seeds=1, attack_grid=[0.2, 0.4])
    make_dummy_artifacts(cfg)
    extract_features(cfg)
    d = paths.artifact_dir(cfg.run_id)
    for h in cfg.horizons:
        f = pd.read_parquet(d / paths.features_file(h))
        assert len(f) == 4 * 2 * 2 * 1 and f.notna().all().all()
