"""E1 + E7 + R1 go/no-go.
(1) survival vs p for all four families (N=1000)  -> p_c, sharpness, 95% CI
(2) cascade length & lead time near p_c for N = 500..4000  -> does the plateau grow? (Zhou 2014: <tau> ~ N^(1/3))
"""
import json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.plotting import FAMILY_COLOR, FAMILY_ORDER, style
from common.contracts import ThresholdReport
from person_a_sim.threshold_analysis import sweep, estimate_pc, bootstrap_pc

t0 = time.time()
GRID = np.round(np.arange(0.0, 0.62, 0.02), 2)
df = pd.concat([sweep(f, n_graphs=20, grid=GRID, n_seeds=3) for f in FAMILY_ORDER])
df.to_csv("results/E1_sweep.csv", index=False)
print(f"sweep: {len(df)} cascades, {time.time()-t0:.0f}s")

reports, summary = [], []
for f in FAMILY_ORDER:
    d = df[df.family == f]
    est, ci = estimate_pc(d), bootstrap_pc(d)
    reports.append(ThresholdReport(family_pair=f"{f}-{f}", p_c_estimate=est["p_c"],
                                   transition_sharpness=est["sharpness"], seed=0).model_dump())
    summary.append({"family": f, **{k: round(v, 3) for k, v in est.items()}, "pc_CI95": [round(x, 3) for x in ci]})
json.dump(reports, open("results/threshold_report.json", "w"), indent=2)
print(pd.DataFrame(summary).to_string(index=False))

# ---- Figure E1: survival and collapse probability vs p
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for f in FAMILY_ORDER:
    g = df[df.family == f].groupby("p")
    m, s = g.S_final.mean(), g.S_final.std()
    axes[0].plot(m.index, m, color=FAMILY_COLOR[f], lw=2, label=f)
    axes[0].fill_between(m.index, m - s, m + s, color=FAMILY_COLOR[f], alpha=.15, lw=0)
    axes[1].plot(m.index, g.collapsed.mean(), color=FAMILY_COLOR[f], lw=2, marker="o", ms=3, label=f)
axes[0].axvline(1 - 2.4554 / 4, color="#888", ls="--", lw=1)
axes[0].text(1 - 2.4554 / 4 + .005, .8, "ER theory\n(Buldyrev 2010)", fontsize=8, color="#555")
axes[0].set(xlabel="attack fraction p (layer A removed)", ylabel="final surviving fraction  S", title="Survival vs attack")
axes[1].set(xlabel="attack fraction p", ylabel="P(collapse)  [S < 0.25]", title="Collapse probability")
for a in axes: style(a); a.legend(frameon=False)
fig.tight_layout(); fig.savefig("results/figures/E1_survival_vs_p.png", dpi=150)

# ---- R1: cascade length near p_c vs N
pc = {r["family"]: r["p_c"] for r in summary}
rows = []
for n in [500, 1000, 2000, 4000]:
    for f in FAMILY_ORDER:
        win = np.round(pc[f] + np.array([-0.02, -0.01, 0, 0.01, 0.02]), 3)
        rows.append(sweep(f, n_graphs=8, grid=win, n_seeds=3, n=n, base_seed=7))
r1 = pd.concat(rows)
r1.to_csv("results/R1_window.csv", index=False)
tab = r1.groupby(["family", "n"]).agg(half_mean=("half_steps", "mean"), half_max=("half_steps", "max"),
                                      P_collapse=("collapsed", "mean")).round(2)
lead = r1[r1.collapsed].groupby(["family", "n"]).lead.median().rename("lead_median_collapsed")
tab = tab.join(lead)
print(tab.to_string())
tab.to_csv("results/R1_summary.csv")

fig, ax = plt.subplots(figsize=(5, 3.8))
for f in FAMILY_ORDER:
    t = tab.loc[f]
    ax.plot(t.index, t.half_mean, color=FAMILY_COLOR[f], lw=2, marker="o", ms=5, label=f)
ns = np.array([500, 4000])
ax.plot(ns, tab.loc["ER"].half_mean.iloc[0] * (ns / 500) ** (1 / 3), color="#888", ls="--", lw=1, label="~N^(1/3)")
ax.set(xscale="log", yscale="log", xlabel="nodes per layer N", ylabel="mean cascade length (half-steps)",
       title="Cascade length near $p_c$")
style(ax); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("results/figures/R1_length_vs_N.png", dpi=150)
print(f"total {time.time()-t0:.0f}s")
