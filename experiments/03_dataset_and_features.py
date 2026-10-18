"""Step 4: build the main dataset (configs/grid.yaml) and features at h = 2,4,6,8; print sanity checks."""
import time, yaml
import pandas as pd
from common.contracts import ExperimentConfig
from common import paths
from person_a_sim.pipeline import build_networks_and_cascades
from person_b_ml.features import extract_features
from person_b_ml.feature_extraction import ALL

cfg = ExperimentConfig(**yaml.safe_load(open("configs/grid.yaml"))[0])
t0 = time.time()
reps = build_networks_and_cascades(cfg); t1 = time.time()
extract_features(cfg); t2 = time.time()
print(f"cascades {t1-t0:.0f}s, features {t2-t1:.0f}s")
print("p_c:", {r.family_pair: round(r.p_c_estimate, 3) for r in reps})

d = paths.artifact_dir(cfg.run_id)
out = pd.read_parquet(d / paths.OUTCOMES)
print("\ncollapse prevalence by family:\n", out.groupby("family").collapsed.mean().round(3).to_string())
print("overall prevalence:", round(out.collapsed.mean(), 3), "| n cascades:", len(out))

rows = []
for h in cfg.horizons:
    f = pd.read_parquet(d / paths.features_file(h))
    done = (1 - f.failed_A) < cfg.collapse_threshold          # already visibly collapsed at h
    rows.append({"h": h, "already_collapsed": done.mean(), "already_stopped": f.stopped.mean(),
                 "undecided_collapsing": (f.collapsed & ~done).mean(), "ews_reliable": f.ews_reliable.mean()})
print("\nwhat an observer sees at each horizon:\n", pd.DataFrame(rows).round(3).to_string(index=False))

f4 = pd.read_parquet(d / paths.features_file(4))
print("\nfeature means at h=4 by family:\n", f4.groupby("family")[ALL].mean().T.round(3).to_string())
