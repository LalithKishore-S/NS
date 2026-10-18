# Step 6 reading — early collapse prediction and transfer (E3, E4, E5)
Code: `person_b_ml/collapse_predictor.py`, `experiments/05_collapse_prediction.py`, `experiments/06_collapse_report.py`

## Papers
1. ★ **Salehpour & Al-Anbagi, *RESP: A real-time early-stage failure prediction mechanism for smart grid systems*,
   IEEE Systems Journal (2024)** — [author's publication list](https://uregina.ca/~alanbagi/publications.html)
   - Full text / abstract was not accessible from here — **read it via your library (IEEE Xplore)** and fill the
     comparison-matrix row. Related by the same group: *Modeling cascading failures in coupled smart grid networks*,
     IEEE Access 10 (2022), and *An early-stage failure prediction mechanism in smart grid networks*, IET CPS 10(1) (2025).
   - Per the project plan: early-prediction features from partial failures + XGBoost, on a *fixed* coupled grid.
   - **Backs:** Model B's features and the XGBoost model. **We differ:** many synthetic topologies, a state-only
     baseline it must beat, and a test on a topology family never seen in training.
2. ★ **Liu et al., *Early predictor for the onset of critical transitions in networked dynamical systems*,
   Phys. Rev. X 14, 031009 (2024)** — [APS Physics summary](https://physics.aps.org/articles/v17/110)
   - Graph neural network (GIN) + GRU on short node-level time series; predicts **when** a networked system tips.
   - Tested on simulations and real data (e.g. vegetation in Central Africa); transfer across network sizes / dynamics
     is left as future work.
   - **Backs:** framing early prediction as a supervised task on networked dynamics, and the need to test transfer.
     **We differ:** no node-level data and no graph given to the model; binary outcome; we *do* test transfer (E5).
3. ★ (optional) **Ma, Zeng, Zhang & Bury, *Learning from the past: predicting critical transitions with machine
   learning trained on surrogates of historical data*** — [arXiv:2410.09707](https://arxiv.org/abs/2410.09707) (Comm. Phys. 2025 per plan)
   - Trains classifiers on surrogates of a system's own history; compares to variance and lag-1 AC with ROC-AUC.
   - **Backs:** our evaluation design — always report a simple baseline next to the ML model, using ROC-AUC.

## The three nested models (plan §11)
| model | features |
|---|---|
| A state-only | p, failed_A, failed_B — the obvious state; mandatory baseline |
| B dynamics-aware | A + rates, acceleration, EWS (var, lag-1 AC), cross-layer split, failure timing |
| C topology-informed | B + **predicted** family probabilities (mechanism classifier fitted on training graphs only, cross-fitted) |

## Evaluation subsets (D8, D9)
- **all**: every cascade — dominated by easy cases far from p_c (ceiling effect: state-only AUC ≈ 0.99).
- **und**: not yet visibly collapsed at the horizon.
- **crit (primary)**: und AND attack within ±0.04 of the family's p_c — the PRL-2025 plateau regime where ~50% collapse
  and the attack size alone is genuinely ambiguous. p_c only selects which cases are scored; it is never a model input.

## Results (N=2000, 10,800 cascades, 5-fold grouped CV × 3 seeds; ROC-AUC on crit cases)
Prevalence on crit: 0.52 (h=2), 0.48 (h=4), 0.36 (h=6), 0.24 (h=8); n = 2400 → 1528.

### E3 — do dynamics beat the state? **Yes, consistently.**
| h | A state (XGB) | B dynamics (XGB) | paired gain B−A (XGB / RF / LR) |
|---|---|---|---|
| 2 | 0.895 | 0.911 | +0.016 / +0.032 / +0.044 |
| 4 | 0.951 | 0.964 | +0.013 / +0.025 / +0.056 |
| 8 | 0.955 | 0.969 | +0.015 / +0.023 / +0.061 |
Every model, every horizon, every seed: B > A. Gains are modest in-distribution because the state is already informative.
On all cases both are ≈0.99 — that is why the crit subset is the honest test (D9).

### E4 — does inferred topology help? **Not beyond dynamics.**
C − B ≈ 0.000 for RF / XGB at every horizon. It helps only the linear model (LR: +0.05 at h=2), i.e. the family
probabilities supply a non-linear interaction LR cannot build itself. A tree model already extracts it from the dynamics.

### E5 — does it transfer to an unseen family? **Dynamics do; the state does not.** (headline)
XGB, crit cases, ROC-AUC (seen in training → held out):
| held-out | h | A state | B dynamics | C topology |
|---|---|---|---|---|
| WS | 2 | 0.920 → **0.510** | 0.918 → **0.821** | 0.918 → 0.770 |
| WS | 4 | 0.971 → **0.731** | 0.979 → **0.953** | 0.980 → 0.956 |
| WS | 8 | 0.980 → **0.694** | 0.989 → **0.990** | 0.989 → 0.990 |
| ER / BA / SBM | 2–8 | gap ≤ 0.01 | gap ≤ 0.01 | gap ≤ 0.01 |

Interpretation:
- ER, BA, SBM have nearly the same p_c (~0.39), so the state model's learned "collapse above p≈0.39" rule transfers.
- WS collapses at p≈0.23. A state model trained on the other three has never seen collapse at p≈0.23 → it is at
  chance (AUC 0.51 at h=2). The dynamics model reads *how the cascade is unfolding*, not *how hard it was hit*, and
  keeps AUC 0.82 at h=2 and 0.95 by h=4. **Generalization gap: state 0.24–0.41 vs dynamics 0.10 → 0.00.**
- Inferred topology does not survive the shift (C ≤ B at h=2 for held-out WS: the 3-class classifier forces WS into
  the wrong families). The plan (§11) lists this as a valid, publishable negative result.

### Caveats to state in the paper
- Seed CIs (± values in results/E34_auc_crit.csv) reflect fold / model randomness only; graph-level bootstrap CIs are added in Step 7.
- p_c used to define crit comes from the same dataset (it only selects which cases are scored).
- Only one family (WS) has a distinct p_c; the transfer result is driven by that family. ER/BA/SBM transfer is easy because their thresholds coincide.
