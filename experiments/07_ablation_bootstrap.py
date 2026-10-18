"""E6: feature-group ablations (within-distribution AND held-out WS) + graph-bootstrap CIs for the headline numbers.
XGB, crit cases (D9), seeds 0-2 for ablations; bootstrap on seed 0."""
import json, yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.contracts import ExperimentConfig, FAMILIES
from common import paths
from common.plotting import style
from person_b_ml.ablation import ablation_sets, oof_within, lofo, bootstrap_auc, bootstrap_diff
from person_b_ml.collapse_predictor import eval_masks
from sklearn.metrics import roc_auc_score

cfg = ExperimentConfig(**yaml.safe_load(open("configs/grid.yaml"))[0])
d = paths.artifact_dir(cfg.run_id)
pc = {r["family_pair"].split("-")[0]: r["p_c_estimate"] for r in json.load(open(d / paths.THRESHOLD_REPORT))}
SETS = ablation_sets()
rows, boot = [], []
for h in cfg.horizons:
    feat = pd.read_parquet(d / paths.features_file(h)).reset_index(drop=True)
    crit = eval_masks(feat, cfg.collapse_threshold, pc)["crit"]
    for s in [0, 1, 2]:
        oof = oof_within(feat, SETS, "XGB", s)
        y = feat.collapsed.astype(int).values
        for k, p in oof.items():
            rows.append({"setting": "within", "h": h, "set": k, "seed": s, "auc": roc_auc_score(y[crit], p[crit])})
        te, lp = lofo(feat, "WS", SETS, "XGB", s)
        tcrit = eval_masks(te, cfg.collapse_threshold, pc)["crit"]
        for k, p in lp.items():
            rows.append({"setting": "lofo_WS", "h": h, "set": k, "seed": s,
                         "auc": roc_auc_score(te.collapsed.astype(int)[tcrit], p[tcrit])})
        if s == 0:   # graph-bootstrap CIs for the headline comparisons
            for name, (D, P, M) in {"within": (feat, oof, crit), "lofo_WS": (te, lp, tcrit)}.items():
                for k in ["A_state", "B_full"]:
                    a, lo, hi = bootstrap_auc(D, P[k], M)
                    boot.append({"setting": name, "h": h, "what": f"AUC {k}", "value": a, "lo": lo, "hi": hi})
                dv, lo, hi, pneg = bootstrap_diff(D, P["A_state"], P["B_full"], M)
                boot.append({"setting": name, "h": h, "what": "AUC B - A", "value": dv, "lo": lo, "hi": hi, "P(diff<=0)": pneg})
    print(f"h={h} done", flush=True)

ab = pd.DataFrame(rows); ab.to_csv("results/E6_ablation.csv", index=False)
bt = pd.DataFrame(boot); bt.to_csv("results/bootstrap_CIs.csv", index=False)
tab = ab.groupby(["setting", "set", "h"]).auc.mean().unstack().round(3)
print("\nE6 ablation (XGB, crit ROC-AUC, mean of 3 seeds):\n", tab.to_string())
print("\nGraph-bootstrap 95% CIs (seed 0):\n", bt.round(3).to_string(index=False))

# figure: drop in AUC when a group is removed, and AUC with state + one group, h=4
h0 = 4
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
groups = ["rate", "ews", "cross", "timing"]
for ax, setting, title in zip(axes, ["within", "lofo_WS"], ["seen families (within)", "held-out WS"]):
    t = ab[(ab.setting == setting) & (ab.h == h0)].groupby("set").auc.mean()
    full, base = t["B_full"], t["A_state"]
    drop = [full - t[f"B_minus_{g}"] for g in groups]
    alone = [t[f"state_plus_{g}"] - base for g in groups]
    yv = np.arange(len(groups))
    ax.barh(yv + 0.18, alone, height=0.34, color="#2a78d6", label="gain: state + this group only")
    ax.barh(yv - 0.18, drop, height=0.34, color="#eb6834", label="loss: full B minus this group")
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_yticks(yv, ["failure rates", "EWS (var, AC1)", "cross-layer", "failure timing"])
    ax.set(xlabel=f"Δ ROC-AUC at h={h0}", title=f"E6: {title}")
    style(ax)
axes[0].legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig("results/figures/E6_ablation.png", dpi=150)
