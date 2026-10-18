"""E7: is the estimated critical region meaningful?  Three independent checks.
 (1) theory: ER p_c vs Buldyrev 2010 (1 - 2.4554/<k> = 0.386), and stability of p_c from N=1000 to N=2000
 (2) dynamics: cascade length should PEAK at p_c (Zhou et al. 2014: tau diverges at p_c) — independent of the label
 (3) label: p_c should barely move when the collapse threshold changes (first-order jump -> S_final is bimodal)
"""
import json, yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.contracts import ExperimentConfig, FAMILIES
from common import paths
from common.plotting import FAMILY_COLOR, style
from person_a_sim.threshold_analysis import sweep, estimate_pc

cfg = ExperimentConfig(**yaml.safe_load(open("configs/grid.yaml"))[0])
d = paths.artifact_dir(cfg.run_id)
out = pd.read_parquet(d / paths.OUTCOMES)
rows = []
for f in FAMILIES:
    o = out[out.family == f]
    small = sweep(f, n_graphs=20, grid=cfg.attack_grid, n_seeds=3, n=1000, base_seed=11)
    r = {"family": f, "pc_N1000": estimate_pc(small)["p_c"], "pc_N2000": estimate_pc(o)["p_c"]}
    L = o.groupby("p").n_half.mean()
    r["p_at_max_length"] = L.idxmax()
    for thr in [0.10, 0.25, 0.40]:
        r[f"pc_thr{thr:.2f}"] = estimate_pc(o.assign(collapsed=o.S_final < thr))["p_c"]
    r["S_final_in_0.10-0.40"] = ((o.S_final > 0.10) & (o.S_final < 0.40)).mean()
    rows.append(r)
t = pd.DataFrame(rows).round(3)
print(t.to_string(index=False))
print("ER theory (Buldyrev 2010):", round(1 - 2.4554 / cfg.avg_degree, 3))
t.to_csv("results/E7_threshold_validation.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for f in FAMILIES:
    o = out[out.family == f]
    L = o.groupby("p").n_half.mean()
    axes[0].plot(L.index, L, color=FAMILY_COLOR[f], lw=2, label=f)
    axes[0].axvline(t.set_index("family").loc[f, "pc_N2000"], color=FAMILY_COLOR[f], ls=":", lw=1.2)
    axes[1].hist(o.S_final, bins=40, range=(0, 1), histtype="step", lw=2, color=FAMILY_COLOR[f], label=f)
axes[0].set(xlabel="attack fraction p", ylabel="mean cascade length (half-steps)", title="Cascade length peaks at $p_c$ (dotted)")
axes[1].axvline(cfg.collapse_threshold, color="#888", ls="--", lw=1)
axes[1].set(xlabel="final surviving fraction S", ylabel="cascades", yscale="log", title="S is bimodal: threshold sits in the gap")
for a in axes: style(a); a.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("results/figures/E7_threshold_validation.png", dpi=150)
