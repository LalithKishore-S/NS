# Step 7 — ablations (E6), uncertainty, threshold validation (E7), reproducibility
Code: `person_b_ml/ablation.py`, `experiments/07_ablation_bootstrap.py`, `experiments/08_threshold_validation.py`,
`run_experiments.py`, `Makefile`, `paper/draft.md`

## Papers used here (already read in earlier steps)
- Zhou et al., PRE 90, 012803 (2014): cascade length diverges at p_c → E7 check 2 (length peaks at our p_c).
- Buldyrev et al., Nature 2010: ER theory p_c = 0.386 → E7 check 1.
- Scheffer 2009 / Dakos 2012 / Deb 2022: the EWS group whose transfer we test in E6.
- Ma et al. 2025: always compare with a simple baseline, report ROC-AUC with uncertainty → bootstrap design.

## E7 — the critical region is meaningful (three independent checks)
| family | p_c N=1000 | p_c N=2000 | p at max length | p_c thr 0.10 / 0.25 / 0.40 |
|---|---|---|---|---|
| ER | 0.402 | 0.394 | 0.40 | 0.394 / 0.394 / 0.374 |
| BA | 0.383 | 0.385 | 0.40 | 0.387 / 0.385 / 0.363 |
| WS | 0.229 | 0.225 | 0.24 | 0.225 / 0.225 / 0.225 |
| SBM | 0.394 | 0.392 | 0.40 | 0.392 / 0.392 / 0.373 |
- ER moves toward theory (0.386) as N grows. Length peaks within one grid step of p_c (label-free check).
- S_final is bimodal; threshold 0.10 and 0.25 give identical p_c. 0.40 cuts into the surviving branch → shifts p_c.
  This justifies 0.25 physically (plan §9: chosen by reasoning + sensitivity, not accuracy).

## E6 — what transfers
- Within distribution, feature groups are redundant (any of rates / timing / EWS ≈ full model).
- Held-out WS: timing (h=2) and rates (h≥4) carry the transferable signal; **EWS do not transfer** (they are a
  family signature themselves); cross-layer features slightly hurt at h=2.

## Uncertainty (graph-level bootstrap, 1,000 resamples of graphs)
- B − A within: +0.010 to +0.017 AUC, every CI excludes 0.
- B − A held-out WS: +0.11 to +0.34, every CI excludes 0.
- State-only model is unstable under shift across seeds (h=8: 0.69–0.88); dynamics model is not (0.99).

## Reproduce everything
`make test` then `make experiments` (≈35 min on 2 cores). `results/run_info.json` records git commit + config.
