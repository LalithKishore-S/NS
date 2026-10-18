"""E3 (A vs B), E4 (B vs C), E5 (leave-one-family-out) — the headline experiments.

Three scored subsets (D8, D9): all / und (not yet collapsed at h) / crit (und AND near the family's p_c).
Generalization gap for family F = within-distribution score on F's graphs − score when F was never seen in training.
"""
import json, time, yaml
import numpy as np
import pandas as pd

from common.contracts import ExperimentConfig, MetricsRow, FAMILIES
from common import paths
from person_b_ml.collapse_predictor import within_distribution, leave_one_family_out, score, eval_masks

cfg = ExperimentConfig(**yaml.safe_load(open("configs/grid.yaml"))[0])
d = paths.artifact_dir(cfg.run_id)
THR, SEEDS, MODELS = cfg.collapse_threshold, [0, 1, 2], ["LR", "RF", "XGB"]
pc = {r["family_pair"].split("-")[0]: r["p_c_estimate"] for r in json.load(open(d / paths.THRESHOLD_REPORT))}
t0 = time.time()
rows, mrows = [], []


def add(setting, held, fam, h, key, s, sc):
    rows.append({"setting": setting, "held_out": held, "family": fam, "h": h, "set": key[0], "model": key[1], "seed": s, **sc})


for h in cfg.horizons:
    feat = pd.read_parquet(d / paths.features_file(h)).reset_index(drop=True)
    masks = eval_masks(feat, THR, pc)
    for s in SEEDS:
        for key, prob in within_distribution(feat, MODELS, s).items():
            sc = score(feat, prob, masks); add("within", None, "ALL", h, key, s, sc)
            mrows.append(MetricsRow(experiment="E3E4_within", model="/".join(key), observation_fraction=h, seed=s,
                                    roc_auc=sc["crit_roc_auc"], pr_auc=sc["crit_pr_auc"],
                                    macro_f1=sc["crit_macro_f1"], balanced_acc=sc["crit_balanced_acc"]).model_dump())
            for f in FAMILIES:
                idx = (feat.family == f).values
                add("within", None, f, h, key, s, score(feat[idx], prob[idx], {k: v[idx] for k, v in masks.items()}))
        for f in FAMILIES:
            te, probs = leave_one_family_out(feat, f, MODELS, s)
            tm = eval_masks(te, THR, pc)
            for key, prob in probs.items():
                sc = score(te, prob, tm); add("lofo", f, f, h, key, s, sc)
                mrows.append(MetricsRow(experiment="E5_lofo", model="/".join(key), observation_fraction=h,
                                        held_out_family=f, seed=s, roc_auc=sc["crit_roc_auc"], pr_auc=sc["crit_pr_auc"],
                                        macro_f1=sc["crit_macro_f1"], balanced_acc=sc["crit_balanced_acc"]).model_dump())
        print(f"h={h} seed={s} done ({time.time()-t0:.0f}s)", flush=True)

pd.DataFrame(rows).to_csv("results/E345_all_scores.csv", index=False)
json.dump(mrows, open("results/E345_metrics.json", "w"), indent=2)
print(f"total {time.time()-t0:.0f}s")
