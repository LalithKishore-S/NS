"""E2: is the hidden generating family identifiable from a partial cascade, and when?

Held-out-instance (5-fold grouped by graph_id), 3 seeds, LR / RF / XGB, horizons h = 2,4,6,8.
Also: identifiability vs attack fraction p (R2: where is the boundary?) and confusion matrix.
Saves out-of-fold family probabilities for Model C (Step 6).
"""
import json, yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

from common.contracts import ExperimentConfig, MetricsRow, FAMILIES
from common import paths
from common.plotting import FAMILY_COLOR, style
from person_b_ml.mechanism_classifier import run_mechanism
from person_b_ml.evaluation import mean_ci
from person_b_ml.models import MODEL_NAMES

cfg = ExperimentConfig(**yaml.safe_load(open("configs/grid.yaml"))[0])
d = paths.artifact_dir(cfg.run_id)
SEEDS = [0, 1, 2]
rows, keep = [], {}
for h in cfg.horizons:
    feat = pd.read_parquet(d / paths.features_file(h)).reset_index(drop=True)
    for m in MODEL_NAMES:
        for s in SEEDS:
            met, probs, pred = run_mechanism(feat, m, s)
            rows.append(MetricsRow(experiment="E2_mechanism", model=m, observation_fraction=h, seed=s,
                                   macro_f1=met["macro_f1"], balanced_acc=met["balanced_acc"]).model_dump())
            if m == "XGB" and s == 0:
                keep[h] = (feat, probs, pred)
                probs.assign(graph_id=feat.graph_id, seed=feat.seed, p=feat.p).to_parquet(d / f"mech_probs_h{h}.parquet")
    print(f"h={h} done")

res = pd.DataFrame(rows)
json.dump(rows, open("results/E2_metrics.json", "w"), indent=2)
tab = res.groupby(["observation_fraction", "model"]).macro_f1.apply(lambda x: "%.3f ± %.3f" % mean_ci(x)).unstack()
tab.index.name = "h (half-steps)"
print("\nE2 macro-F1 (mean ± 95% CI over 3 seeds; chance = 0.25):\n", tab.to_string())
tab.to_csv("results/E2_table.csv")

# confusion at h=4 (XGB, seed 0)
feat, probs, pred = keep[4]
cm = confusion_matrix(feat.family, pred, labels=list(FAMILIES), normalize="true")
print("\nconfusion h=4 (rows=true, cols=pred):\n", pd.DataFrame(cm, index=FAMILIES, columns=FAMILIES).round(2).to_string())

# identifiability vs p (R2 boundary), XGB seed 0
bins = []
for h in cfg.horizons:
    feat, _, pred = keep[h]
    for p, g in feat.assign(pred=pred).groupby("p"):
        bins.append({"h": h, "p": p, "bal_acc": balanced_accuracy_score(g.family, g.pred)})
bp = pd.DataFrame(bins)
bp.to_csv("results/E2_by_p.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for m, c in zip(MODEL_NAMES, ["#2a78d6", "#eb6834", "#1baf7a"]):
    g = res[res.model == m].groupby("observation_fraction").macro_f1
    axes[0].errorbar(g.mean().index, g.mean(), yerr=1.96 * g.std() / np.sqrt(len(SEEDS)), color=c, lw=2,
                     marker="o", ms=5, capsize=3, label=m)
axes[0].axhline(0.25, color="#888", ls="--", lw=1); axes[0].text(2, 0.27, "chance", color="#555", fontsize=8)
axes[0].set(xlabel="observation horizon h (half-steps)", ylabel="macro-F1", title="E2: family identification", ylim=(0, 1))
shades = ["#cde2fb", "#86b6ef", "#2a78d6", "#104281"]
for h, c in zip(cfg.horizons, shades):
    t = bp[bp.h == h]
    axes[1].plot(t.p, t.bal_acc, color=c, lw=2, label=f"h={h}")
pcs = json.load(open(d / paths.THRESHOLD_REPORT))
for r in pcs:
    f = r["family_pair"].split("-")[0]
    axes[1].axvline(r["p_c_estimate"], color=FAMILY_COLOR[f], ls=":", lw=1.2)
    axes[1].text(r["p_c_estimate"], 0.05 + 0.07 * FAMILIES.index(f), f" $p_c$ {f}", color="#555", fontsize=7)
axes[1].axhline(0.25, color="#888", ls="--", lw=1)
axes[1].set(xlabel="attack fraction p", ylabel="balanced accuracy (XGB)", title="Identifiability vs attack size", ylim=(0, 1))
for a in axes: style(a); a.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("results/figures/E2_mechanism_id.png", dpi=150)
