"""Build the two datasets (random + degree attack) + features, from configs/grid_attacks.yaml."""
import sys, os, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.contracts import ExperimentConfig
from person_a_sim.pipeline import build_networks_and_cascades
from person_b_ml.features import extract_features

cfgs = [ExperimentConfig(**d) for d in yaml.safe_load(open("configs/grid_attacks.yaml"))]
for c in cfgs:
    print(f"building {c.run_id} (attack={c.attack}) ...", flush=True)
    build_networks_and_cascades(c)
    extract_features(c)
    print(f"  done {c.run_id}", flush=True)
print("ALL DATASETS BUILT")
