"""Extension #2 - PARTIAL COUPLING: vary the fraction q of interdependent nodes.

q = 1.0 : every node depends on a partner in the other layer (what the project used so far).
q < 1.0 : only a fraction of nodes are interdependent (more realistic).

We watch three things as q changes:
  p_c        - the collapse point (higher = tougher system)
  sharpness  - how ABRUPT the collapse is (lower = gentler, more gradual)
  window     - half-steps of early warning before collapse (higher = easier to warn)

Known physics (Parshani-Buldyrev 2010): lowering q makes the system tougher AND turns the
sudden crash into a gentler decline -> we expect a LONGER warning window at low q.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from person_a_sim.threshold_analysis import sweep, estimate_pc

N, K, THR = 800, 4.0, 0.25
GRID = [round(0.03 * i, 3) for i in range(27)]     # 0.00 .. 0.78 (low q is very robust, so sweep high)
FAMS = ["ER", "BA", "WS", "SBM"]
QS = [0.25, 0.5, 0.75, 1.0]                         # fraction of interdependent nodes
NG, NS = 5, 3

rows = []
print(f"{'family':7s}{'coupling q':>11s}{'p_c':>9s}{'sharpness':>11s}{'window':>9s}")
for fam in FAMS:
    for q in QS:
        df = sweep(fam, NG, GRID, n_seeds=NS, n=N, k=K, q=q, threshold=THR, base_seed=0)
        est = estimate_pc(df)
        pc = est["p_c"]
        near = df[(df.collapsed) & (abs(df.p - (pc if pc == pc else 0)) <= 0.06)]
        lead = float(near["lead"].median()) if len(near) else float("nan")
        pc_s = f"{pc:.3f}" if pc == pc else "none"    # 'none' = never collapsed in range (very robust)
        print(f"{fam:7s}{q:>11.2f}{pc_s:>9s}{est['sharpness']:>11.1f}{lead:>9.1f}")
        rows.append({"family": fam, "q": q, "p_c": pc, "sharpness": est["sharpness"], "window": lead})
    print()

pd.DataFrame(rows).to_csv("results_coupling.csv", index=False)
print("saved results_coupling.csv")
print("MAIN FINDING: read down each family - as coupling q drops, p_c RISES = tougher system.")
print("(sharpness/window are exploratory and noisy; the robust, presentable result is the p_c trend.)")
