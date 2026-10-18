"""Leakage rules (plan §5, §12)."""
import numpy as np
import pandas as pd
from common.contracts import ExperimentConfig
from common import paths
from person_a_sim.pipeline import build_networks_and_cascades
from person_b_ml.feature_extraction import features_one


def test_features_ignore_future():
    """Changing anything AFTER horizon h must not change features at h."""
    h = 4
    base = pd.DataFrame({"h": range(10), "new_fail_A": [.3, .1, .05, .05, .02, .01, .01, .2, .2, .06],
                         "new_fail_B": [0, .1, .05, .05, .02, .01, .01, .2, .2, .06]})
    alt = base.copy()
    alt.loc[alt.h > h, ["new_fail_A", "new_fail_B"]] = 0.0          # completely different future
    alt = alt[alt.h <= 6]                                             # and a different length
    assert features_one(base, .3, h) == features_one(alt, .3, h)


def test_artifacts_hold_no_wiring():
    """No adjacency / dependency / node-level data may be written (Person B can't read what isn't there)."""
    cfg = ExperimentConfig(run_id="pytest_leak", n_nodes=200, n_graphs_per_family=1, n_seeds=1, attack_grid=[0.3])
    build_networks_and_cascades(cfg)
    d = paths.artifact_dir(cfg.run_id)
    traj = pd.read_parquet(d / paths.TRAJECTORIES)
    assert set(traj.columns) == {"graph_id", "seed", "p", "h", "alive_A", "alive_B", "new_fail_A", "new_fail_B"}
    for f in d.iterdir():
        assert not any(k in f.name.lower() for k in ("adj", "edge", "dep", "graph.")), f.name
