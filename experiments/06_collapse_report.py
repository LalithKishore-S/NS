"""Tables + figures for E3/E4/E5 from results/E345_all_scores.csv."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common.plotting import FAMILY_ORDER, style

r = pd.read_csv("results/E345_all_scores.csv")
SET_COLOR = {"A_state": "#9ec5f4", "B_dynamics": "#2a78d6", "C_topology": "#eb6834"}
SET_LABEL = {"A_state": "A state-only", "B_dynamics": "B + dynamics", "C_topology": "C + inferred topology"}


def ci(x):
    x = np.asarray(x, float); return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def fmt(x):
    m, c = ci(x); return f"{m:.3f} ± {c:.3f}"


w = r[(r.setting == "within") & (r.family == "ALL")]
for sub in ["crit", "und", "all"]:
    t = w.groupby(["h", "model", "set"])[f"{sub}_roc_auc"].apply(fmt).unstack()
    print(f"\nE3/E4 within-distribution ROC-AUC on '{sub}' cases:\n{t.to_string()}")
    t.to_csv(f"results/E34_auc_{sub}.csv")
prev = w.groupby("h")[["crit_n", "crit_prev", "und_n", "und_prev", "all_prev"]].mean().round(3)
print("\nsubset sizes / prevalence:\n", prev.to_string())

# paired improvement B-A and C-B per seed (same folds -> paired)
piv = w.pivot_table(index=["h", "model", "seed"], columns="set", values="crit_roc_auc")
gain = pd.DataFrame({"B-A": piv.B_dynamics - piv.A_state, "C-B": piv.C_topology - piv.B_dynamics})
print("\npaired AUC gains on crit (mean ± CI over seeds):\n",
      gain.groupby(["h", "model"]).agg(fmt).to_string())

# E5 generalization gap
wf = r[(r.setting == "within") & (r.family != "ALL")]
lo = r[r.setting == "lofo"]
key = ["family", "h", "set", "model", "seed"]
g = wf[key + ["crit_roc_auc"]].merge(lo[key + ["crit_roc_auc"]], on=key, suffixes=("_within", "_lofo"))
g["gap"] = g.crit_roc_auc_within - g.crit_roc_auc_lofo
tab = g[g.model == "XGB"].groupby(["family", "h", "set"])[["crit_roc_auc_within", "crit_roc_auc_lofo", "gap"]].mean().round(3)
print("\nE5 leave-one-family-out (XGB, crit cases):\n", tab.to_string())
tab.to_csv("results/E5_gap.csv")

# ---- figure 1: AUC vs horizon (XGB, crit)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
x = w[w.model == "XGB"]
for s in SET_COLOR:
    gg = x[x.set == s].groupby("h").crit_roc_auc
    axes[0].errorbar(gg.mean().index, gg.mean(), yerr=1.96 * gg.std() / np.sqrt(3), color=SET_COLOR[s], lw=2,
                     marker="o", ms=5, capsize=3, label=SET_LABEL[s])
axes[0].set(xlabel="observation horizon h (half-steps)", ylabel="ROC-AUC", title="E3/E4: near-critical, not-yet-collapsed")
# ---- figure 2: within vs held-out family at h=4
h0 = 4
gx = g[(g.model == "XGB") & (g.h == h0)].groupby(["family", "set"])[["crit_roc_auc_within", "crit_roc_auc_lofo"]].mean()
for i, f in enumerate(FAMILY_ORDER):
    for j, s in enumerate(SET_COLOR):
        yv = i + (j - 1) * 0.22
        a, b = gx.loc[(f, s)]
        axes[1].plot([b, a], [yv, yv], color=SET_COLOR[s], lw=2)
        axes[1].plot(a, yv, "o", color=SET_COLOR[s], ms=7, mec="white", mew=1.5, label=SET_LABEL[s] if i == 0 else None)
        axes[1].plot(b, yv, "o", color="white", mec=SET_COLOR[s], mew=2, ms=7)
axes[1].set_yticks(range(4), FAMILY_ORDER)
axes[1].set(xlabel=f"ROC-AUC at h={h0}   (● seen in training   ○ held out)", title="E5: leave-one-family-out")
for a in axes: style(a)
axes[0].legend(frameon=False, fontsize=8); axes[1].legend(frameon=False, fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig("results/figures/E345_collapse.png", dpi=150)
