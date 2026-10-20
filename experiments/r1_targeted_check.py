"""Go/No-Go for targeted attacks: does an early-warning WINDOW survive when we hit hubs first?
Compares random vs degree attack: p_c (collapse point) and lead time (half-steps a doomed
cascade still 'looks alive'). GO if degree-attack median lead near p_c is roughly >= 3."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from person_a_sim.network_generation import make_graph
from person_a_sim.interdependency import couple
from person_a_sim.cascade_simulator import simulate, to_csr
from person_a_sim.attacks import attack_order
from person_a_sim.threshold_analysis import lead_time

N, K, THR = 1000, 4.0, 0.25
GRID = [round(0.01 * i, 3) for i in range(51)]      # 0.00 .. 0.50 (targeted p_c is small, so sweep low)
FAMILIES = ["ER", "BA", "WS", "SBM"]
ATTACKS = ["random", "degree"]
N_GRAPHS, N_SEEDS = 4, 3

rows = []
for fam in FAMILIES:
    for g in range(N_GRAPHS):
        s = 1000 * g
        GA, GB = make_graph(fam, N, K, s), make_graph(fam, N, K, s + 1)
        A, B = to_csr(GA), to_csr(GB)
        dep = couple(N, 1.0, seed=s + 2)
        orders = {a: attack_order(GA, a, seed=s + 5) for a in ATTACKS}
        for a in ATTACKS:
            for p in GRID:
                for r in range(N_SEEDS):
                    o = simulate(A, B, dep, p, seed=s + 3 + r, order=orders[a])
                    rows.append({"family": fam, "attack": a, "p": p, "S_final": o["S_final"],
                                 "collapsed": o["S_final"] < THR,
                                 "lead": lead_time(o["fail_t_A"], THR)})

df = pd.DataFrame(rows)
print("\n=== p_c (collapse point) and early-warning lead by family x attack ===")
print(f"{'family':6s}{'attack':10s}{'p_c':>8s}{'median lead near p_c':>24s}")
for fam in FAMILIES:
    for a in ATTACKS:
        d = df[(df.family == fam) & (df.attack == a)]
        pc = d.groupby("p")["collapsed"].mean()
        above = pc[pc >= 0.5]
        pc_val = float(above.index.min()) if len(above) else float("nan")
        if pc_val == pc_val:
            near = d[(d.collapsed) & (abs(d.p - pc_val) <= 0.04)]
        else:
            near = d[d.collapsed]
        lead_med = float(near["lead"].median()) if len(near) else float("nan")
        print(f"{fam:6s}{a:10s}{pc_val:8.3f}{lead_med:24.1f}")

df.to_csv("results_targeted_check.csv", index=False)
print("\nsaved results_targeted_check.csv")
print("GO if the degree rows show (a) much smaller p_c than random, and (b) median lead >= ~3.")
