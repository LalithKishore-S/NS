"""HEADLINE of the extension: does early-warning learned on one attack TYPE transfer to another?

Trains the 3 collapse models on RANDOM attacks and tests on DEGREE (targeted) attacks, and the
reverse. Reports crit-case ROC-AUC within-attack (CV) vs transferred, and the transfer gap.
A small gap = the early-warning signal survives a change in HOW the system is attacked.
Compare A_state vs B_dynamics gaps: if dynamics transfer better, that's the story.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from common import paths
from person_b_ml.collapse_predictor import fit_predict_all, within_distribution, eval_masks, score

MODELS = ["XGB"]            # add "LR","RF" for more rows
THR = 0.25
HORIZONS = [2, 4, 6, 8]
RUN_RANDOM, RUN_DEGREE, SEED = "main_random", "main_degree", 0


def load(run_id, h):
    d = paths.artifact_dir(run_id)
    f = d / paths.features_file(h)
    if not f.exists():
        raise SystemExit(f"missing {f} - run: python experiments/build_attacks.py first")
    feat = pd.read_parquet(f).reset_index(drop=True)
    with open(d / paths.THRESHOLD_REPORT) as fh:
        pc = {r["family_pair"].split("-")[0]: r["p_c_estimate"] for r in json.load(fh)}
    return feat, pc


def crit_auc(test_feat, probs, pc):
    masks = eval_masks(test_feat, THR, pc)
    return {key: score(test_feat, p, masks).get("crit_roc_auc", float("nan")) for key, p in probs.items()}


rows = []
for h in HORIZONS:
    rf, rpc = load(RUN_RANDOM, h)
    df, dpc = load(RUN_DEGREE, h)
    w_rand = crit_auc(rf, within_distribution(rf, MODELS, SEED), rpc)   # train random / test random (CV)
    w_deg  = crit_auc(df, within_distribution(df, MODELS, SEED), dpc)   # train degree / test degree (CV)
    t_r2d  = crit_auc(df, fit_predict_all(rf, df, MODELS, SEED), dpc)   # train random -> test degree
    t_d2r  = crit_auc(rf, fit_predict_all(df, rf, MODELS, SEED), rpc)   # train degree -> test random
    for key in w_rand:
        mset, clf = key
        rows.append({"h": h, "model": mset,
                     "within_rand": round(w_rand[key], 3), "rand->deg": round(t_r2d[key], 3),
                     "gap_r2d": round(w_rand[key] - t_r2d[key], 3),
                     "within_deg": round(w_deg[key], 3), "deg->rand": round(t_d2r[key], 3),
                     "gap_d2r": round(w_deg[key] - t_d2r[key], 3)})

res = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(res.to_string(index=False))
res.to_csv("results_attack_transfer.csv", index=False)
print("\nsaved results_attack_transfer.csv")
print("Small gap = early warning TRANSFERS across attack type. If B_dynamics gap < A_state gap,")
print("the dynamics carry attack-invariant warning information - that is the headline finding.")
