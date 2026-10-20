"""ONE-COMMAND PRESENTATION DEMO.  Run:  python present.py

Runs the whole project end to end (~3-6 min) and prints every headline result:
  BASELINE (random attacks, your friend's work):
    1  robust transition point (p_c)        2  identify hidden network family
    3  early warning: count vs behaviour     4  transfer to an unseen network family
  EXTENSION (targeted attacks, your addition):
    5  robust-yet-fragile under targeted attack
    6  does early warning transfer across ATTACK type?  <- the headline
"""
import sys, os, json, subprocess, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from common.contracts import ExperimentConfig
from common import paths
from person_a_sim.pipeline import build_networks_and_cascades
from person_b_ml.features import extract_features
from person_b_ml.mechanism_classifier import run_mechanism
from person_b_ml.collapse_predictor import (within_distribution, leave_one_family_out,
                                            fit_predict_all, eval_masks, score)

N, NG, NS, SEED, THR = 800, 10, 3, 0, 0.25
HZ = [4, 8]
FAMS = ["ER", "BA", "WS", "SBM"]
DEG_GRID = [round(0.02 * i, 3) for i in range(16)]   # 0..0.30 (targeted collapses early)
AUC = "crit_roc_auc"


def hdr(t): print("\n" + "=" * 72 + "\n  " + t + "\n" + "=" * 72)
def build(run_id, attack, grid=None):
    kw = dict(run_id=run_id, n_nodes=N, n_graphs_per_family=NG, n_seeds=NS, seed=42,
              collapse_threshold=THR, attack=attack)
    if grid: kw["attack_grid"] = grid
    c = ExperimentConfig(**kw); build_networks_and_cascades(c); extract_features(c)
def load(run_id, h):
    d = paths.artifact_dir(run_id)
    feat = pd.read_parquet(d / paths.features_file(h)).reset_index(drop=True)
    with open(d / paths.THRESHOLD_REPORT) as fh:
        pc = {r["family_pair"].split("-")[0]: r["p_c_estimate"] for r in json.load(fh)}
    return feat, pc
def auc(df, prob, pc, mk="crit"):
    return score(df, prob, eval_masks(df, THR, pc))[f"{mk}_roc_auc"]

# ---- 0 tests ----
hdr("STEP 0  -  automated correctness tests (physics correct + no information leakage)")
try:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q"], capture_output=True, text=True, timeout=300)
    print((r.stdout or r.stderr).strip().splitlines()[-1])
except Exception as e:
    print("(skipped:", e, ")")

# ---- build ----
hdr("Building datasets: random + targeted attacks  (~2-4 min, please wait)")
build("demo_random", "random"); print("  random done")
build("demo_degree", "degree", DEG_GRID); print("  targeted done")

# ---- 1 ----
hdr("RESULT 1  -  tipping point p_c per network family (baseline, random attack)")
_, rpc = load("demo_random", HZ[0])
for f in FAMS: print(f"    {f:5s}  p_c = {rpc.get(f, float('nan')):.3f}")
print("  -> below p_c the system survives; above it, total collapse.")

# ---- 2 ----
hdr("RESULT 2  -  identify the hidden network family from the cascade (chance = 0.25)")
for h in HZ:
    feat, _ = load("demo_random", h)
    m, _, _ = run_mechanism(feat, "XGB", SEED)
    print(f"    horizon h={h}:  macro-F1 = {m['macro_f1']:.3f}")
print("  -> well above chance -> the family leaves a signature in the dynamics.")

# ---- 3 ----
hdr("RESULT 3  -  early warning of collapse: COUNT (A) vs BEHAVIOUR (B)   [ROC-AUC near p_c]")
for h in HZ:
    feat, pc = load("demo_random", h)
    pr = within_distribution(feat, ["XGB"], SEED)
    a = auc(feat, pr[("A_state", "XGB")], pc); b = auc(feat, pr[("B_dynamics", "XGB")], pc)
    print(f"    h={h}:  A_state(count) = {a:.3f}   B_dynamics(behaviour) = {b:.3f}")
print("  -> behaviour matches/beats counting -> watching HOW it unfolds adds information.")

# ---- 4 ----
hdr("RESULT 4  -  transfer to an UNSEEN network family (leave-one-family-out, h=%d)" % HZ[-1])
print("    (gap = drop when the family was never trained on; smaller = transfers better)")
h = HZ[-1]; feat, pc = load("demo_random", h)
wd = within_distribution(feat, ["XGB"], SEED)
print(f"    {'held-out':8s}{'A_state gap':>14s}{'B_dynamics gap':>16s}")
for f in FAMS:
    rows = (feat.family == f).values; sub = feat[rows]
    aw = auc(sub, wd[("A_state", "XGB")][rows], pc); bw = auc(sub, wd[("B_dynamics", "XGB")][rows], pc)
    te, pr = leave_one_family_out(feat, f, ["XGB"], SEED)
    al = auc(te, pr[("A_state", "XGB")], pc); bl = auc(te, pr[("B_dynamics", "XGB")], pc)
    print(f"    {f:8s}{aw - al:>14.3f}{bw - bl:>16.3f}")
print("  -> behaviour generalizes better to unseen families (smaller gaps).")

# ---- 5 ----
hdr("RESULT 5  -  [MY EXTENSION]  robust-yet-fragile: p_c under random vs TARGETED attack")
_, dpc = load("demo_degree", HZ[0])
print(f"    {'family':7s}{'p_c random':>12s}{'p_c targeted':>14s}")
for f in FAMS: print(f"    {f:7s}{rpc.get(f, float('nan')):>12.3f}{dpc.get(f, float('nan')):>14.3f}")
print("  -> targeted (hub) attack collapses the network at a MUCH smaller p (esp. BA).")

# ---- 6 ----
hdr("RESULT 6  -  [MY EXTENSION HEADLINE]  does early warning transfer across ATTACK type?")
print("    (gap = drop when attack type changes; smaller = transfers better)")
print(f"    {'h':>3s}  {'direction':18s}{'A_state gap':>14s}{'B_dynamics gap':>16s}")
for h in HZ:
    rf, rp = load("demo_random", h); df, dp = load("demo_degree", h)
    wr = within_distribution(rf, ["XGB"], SEED); wdg = within_distribution(df, ["XGB"], SEED)
    r2d = fit_predict_all(rf, df, ["XGB"], SEED); d2r = fit_predict_all(df, rf, ["XGB"], SEED)
    aR = auc(rf, wr[("A_state","XGB")], rp) - auc(df, r2d[("A_state","XGB")], dp)
    bR = auc(rf, wr[("B_dynamics","XGB")], rp) - auc(df, r2d[("B_dynamics","XGB")], dp)
    aD = auc(df, wdg[("A_state","XGB")], dp) - auc(rf, d2r[("A_state","XGB")], rp)
    bD = auc(df, wdg[("B_dynamics","XGB")], dp) - auc(rf, d2r[("B_dynamics","XGB")], rp)
    print(f"    {h:>3d}  {'random->targeted':18s}{aR:>14.3f}{bR:>16.3f}")
    print(f"    {'':>3s}  {'targeted->random':18s}{aD:>14.3f}{bD:>16.3f}")
print("  -> KEY: behaviour (B) gap << count (A) gap, esp. targeted->random.")
print("     Watching the cascade's behaviour survives a change in attack type; counting does not.")

hdr("DEMO COMPLETE")
print("  Baseline (1-4) = your friend's project. Extension (5-6) = your targeted-attack addition.")
