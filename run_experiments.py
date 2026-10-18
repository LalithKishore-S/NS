"""One command reproduces every result:  python run_experiments.py   (or: make experiments)

For each config in configs/grid.yaml: Person A pipeline -> Person B features -> experiment scripts.
Writes results/run_info.json with the git commit + config that produced the results (plan §8).
"""
import json, runpy, subprocess, sys, time
from pathlib import Path
import yaml

from common.contracts import ExperimentConfig
from person_a_sim.pipeline import build_networks_and_cascades
from person_b_ml.features import extract_features

EXPERIMENTS = [
    "experiments/00_network_stats.py",          # Step 1: matched families
    "experiments/01_buldyrev_check.py",         # Step 2: simulator vs theory
    "experiments/02_threshold_and_R1.py",       # E1 + R1
    "experiments/04_mechanism_id.py",           # E2
    "experiments/05_collapse_prediction.py",    # E3 E4 E5
    "experiments/06_collapse_report.py",        #   tables + figure
    "experiments/07_ablation_bootstrap.py",     # E6 + graph-bootstrap CIs
    "experiments/08_threshold_validation.py",   # E7
]

if __name__ == "__main__":
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    cfgs = [ExperimentConfig(**c) for c in yaml.safe_load(open("configs/grid.yaml"))]
    t0 = time.time()
    for cfg in cfgs:
        print(f"== {cfg.run_id}: cascades"); build_networks_and_cascades(cfg)
        print(f"== {cfg.run_id}: features"); extract_features(cfg)
    only = sys.argv[1:]  # optional: run a subset, e.g. python run_experiments.py 04 05
    for script in EXPERIMENTS:
        if only and not any(Path(script).name.startswith(o) for o in only):
            continue
        print(f"== {script}", flush=True)
        runpy.run_path(script, run_name="__main__")
    json.dump({"git_commit": commit, "configs": [c.model_dump() for c in cfgs],
               "runtime_s": round(time.time() - t0)}, open("results/run_info.json", "w"), indent=2)
    print(f"done in {time.time() - t0:.0f}s")
