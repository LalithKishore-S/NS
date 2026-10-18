# Early Warning of Cascading Collapse in Interdependent Networks from Partial Observations: Identifiability, Prediction, and Transfer Across Hidden Topologies

*Draft v1 — 2026-10-18. Numbers come from `make experiments` (N = 2000 nodes per layer, 10,800 cascades). Every claim
is framed as a hypothesis tested in the experiments named next to it. Sections marked [A] / [B] / [A+B] show ownership (plan §6).*

---

## Abstract
Failures in coupled infrastructures — a power grid and the communication network that controls it — can cascade
between layers and collapse both abruptly. In practice the wiring of such systems is often unknown, but the sequence
of failures can be observed. We ask what an observer who sees only **aggregate failure counts per layer over time**
can learn. On two-layer interdependent networks generated from four families (Erdős–Rényi, Barabási–Albert,
Watts–Strogatz, stochastic block model) at matched size and mean degree, we find that (i) the hidden family leaves
a detectable signature after only four half-steps of a single cascade (macro-F1 0.65 vs. 0.25 chance), except that
ER and SBM are indistinguishable — an identifiability limit; (ii) near the critical attack size, cascade-dynamics
features consistently improve early collapse prediction over a state-only baseline; and (iii) when a family is held
out of training entirely, the state-only predictor falls to chance on Watts–Strogatz networks (ROC-AUC 0.51 at the
earliest horizon) while the dynamics-aware predictor retains 0.82, rising to 0.95 two half-steps later.
Adding the *inferred* topology does not improve prediction, and does not survive the topology shift.

## 1. Introduction [A+B]
- Motivation: interdependent infrastructures; the 2003 Italy blackout (Buldyrev et al. 2010) — failure in one network
  removes support for the other, producing abrupt, first-order collapse.
- The observability gap: operators see *which components fail and when*, rarely the full dependency wiring.
- Research questions:
  - **RQ1 (Problem A).** Does the hidden generating mechanism leave a signature in partial cascade dynamics?
  - **RQ2 (Problem B).** Do early cascade dynamics predict catastrophic collapse beyond the obvious state?
  - **RQ3 (Problem C).** Does such prediction transfer to network families unseen in training?
- Contributions (the five differentiators, plan §2):
  1. family identification rather than edge reconstruction;
  2. two-layer interdependent networks;
  3. cascade dynamics rather than generic node dynamics;
  4. early collapse warning as a second task;
  5. leave-one-family-out generalization.

## 2. Related work [A+B] (anchored by `literature/comparison_matrix.md`)
- **Interdependent cascades.** Buldyrev et al. (Nature 2010) — mutual giant component, p_c = 2.4554/⟨k⟩ for coupled
  ER; Parshani et al. (PRL 2010) — partial coupling turns the transition second-order; Zhou et al. (PRE 2014) — the
  cascade length diverges at p_c (~N^{1/3}) with a plateau; Dilmoney et al. (PRL 2025) — critical cascades as
  branching / birth–death processes, plateau length and collapse probability.
- **Network inference from dynamics.** Machado et al. (AAAI 2023) recover individual edges of one network from
  partially observed linear dynamics. We infer the *family*, from *cascade counts* with *no node identities*.
- **Early-warning signals.** Scheffer et al. (Nature 2009), Dakos et al. (PLoS ONE 2012): variance and lag-1
  autocorrelation rise before tipping points. Deb et al. (R. Soc. Open Sci. 2022, EWSNet), Ma et al. (2025),
  Liu et al. (PRX 2024): learned predictors of critical transitions; Liu et al. leave cross-network transfer open.
- **ML prediction of cascading failures.** Salehpour & Al-Anbagi (RESP, IEEE Syst. J. 2024): early-stage failure
  prediction with XGBoost on a fixed coupled smart grid. *[TODO: confirm details after reading full text.]*

## 3. Methodology
### 3.1 Network families [A]
ER (G(n,m), exactly ⟨k⟩ = 4), BA (m = 2), WS (k = 4, rewiring 0.1), SBM (4 blocks, 90% of edges within blocks,
expected ⟨k⟩ = 4). N = 2000 per layer; both layers drawn independently from the same family (D1). Matched size and
mean degree prevent a classifier from learning density instead of structure.

### 3.2 Interdependency and cascade [A]
One-to-one random bijection between layers (dependency fraction q = 1). Attack: remove a random fraction p of layer A.
Each round: (a) A-nodes outside A's giant component fail and their B-partners fail; (b) the same for B. Iterate to
stability (Buldyrev et al. 2010). Time is recorded in half-steps (a and b separately) (D2).
Validation (tests + `experiments/01`): no recovery, termination, final state = mutually connected giant component,
survival decreasing in p, and ER p_c matching theory.

### 3.3 Observations and features [B]
The observer receives, per half-step, only the fraction of each layer alive and newly failed (frozen schema, D7):
no node identities, no adjacency, no dependency map. At horizon h ∈ {2, 4, 6, 8} half-steps (fixed counts, not a
fraction of each cascade's length, which would leak the outcome; D3) we compute 15 features in four groups:
state (p, failed_A, failed_B), failure rates (rate, mean rate, acceleration, active steps, stopped),
early-warning signals (variance, lag-1 autocorrelation of the failure-rate series; flagged unreliable below 4 points),
cross-layer balance, and failure timing.

### 3.4 Models [B]
- **Mechanism classifier:** 4-class LR / RF / XGBoost on all non-constant features.
- **Collapse models** (label: S_final < 0.25):
  A = state only; B = A + dynamics; C = B + the mechanism classifier's *predicted* probabilities, fitted on training
  graphs only and cross-fitted on training rows (D10).

## 4. Experimental design [A+B]
- Dataset: 4 families × 50 graph pairs × 18 attack sizes p ∈ [0.16, 0.50] (one grid shared by all families, D5)
  × 3 attack seeds = 10,800 cascades.
- Splits: always by graph instance (StratifiedGroupKFold, 5 folds, 3 seeds); a test asserts no graph crosses a split.
- Mechanism ID: held-out-instance only (a classifier cannot predict an unseen class).
- Transfer: leave-one-family-out for the collapse models (train on three families, test on all graphs of the fourth).
- Collapse metrics on three subsets: all; undecided (not yet collapsed at h); **critical** (undecided and p within
  ±0.04 of the family's p_c — primary, D9). Prevalence on critical: 0.52 / 0.48 / 0.36 / 0.24 at h = 2 / 4 / 6 / 8.
- Uncertainty: mean ± 95% CI over seeds, plus graph-level bootstrap (1,000 resamples of graphs).

## 5. Results
### E1 / E7 — the transition is real, sharp, and has a warning window [A]
| family | p_c (N=1000) | p_c (N=2000) | p at max cascade length | p_c with threshold 0.10 / 0.25 / 0.40 |
|---|---|---|---|---|
| ER | 0.402 | 0.394 | 0.40 | 0.394 / 0.394 / 0.374 |
| BA | 0.383 | 0.385 | 0.40 | 0.387 / 0.385 / 0.363 |
| WS | 0.229 | 0.225 | 0.24 | 0.225 / 0.225 / 0.225 |
| SBM | 0.394 | 0.392 | 0.40 | 0.392 / 0.392 / 0.373 |

- ER approaches the theoretical 0.386 as N grows. BA collapses earlier and more gradually (hubs increase fragility
  under coupling, as in Buldyrev et al.); WS is by far the most fragile.
- Cascade length peaks within one grid step of p_c for every family — an estimate independent of the collapse label.
- S_final is bimodal (only 5–8% of cascades end between 0.10 and 0.40); thresholds 0.10 and 0.25 give identical p_c.
- R1: near p_c, collapsing cascades remain above the threshold for a median 6–8 half-steps (N = 1000), growing with N.
  *Figures: E1_survival_vs_p, R1_length_vs_N, E7_threshold_validation.*

### E2 — the family is identifiable, except ER vs. SBM [B]
Macro-F1 (chance 0.25): XGB 0.60 / 0.65 / 0.67 / 0.67 at h = 2 / 4 / 6 / 8; RF equal; LR 0.54–0.62.
Per-class recall at h = 4: WS 0.93, BA 0.77, ER 0.54, SBM 0.40, with ER and SBM confused with each other
(~0.4–0.5 each way). Merging ER and SBM, ~90% are correctly grouped. Identifiability is roughly flat in p.
*Figure: E2_mechanism_id.*

### E3 — dynamics beat the state [B]
ROC-AUC on critical cases (XGB; A → B): 0.895 → 0.911 (h=2), 0.951 → 0.964 (h=4), 0.955 → 0.969 (h=8).
B > A for every model, horizon and seed (paired gain +0.011 to +0.061). Graph-level bootstrap (XGB): B − A =
+0.017 [0.010, 0.024] (h=2), +0.014 [0.009, 0.018] (h=4), +0.010 [0.005, 0.015] (h=6), +0.015 [0.008, 0.024] (h=8); no
bootstrap resample had B ≤ A.

### E4 — inferred topology adds nothing beyond dynamics [B]
C − B ≈ 0.000 for RF and XGB at every horizon; C helps only the linear model (+0.05 at h=2), i.e. the family
probabilities supply a non-linearity a tree model already extracts from the dynamics.

### E5 — dynamics transfer to an unseen family; the state does not [B] (headline)
| held out | h | A state (seen → unseen) | B dynamics (seen → unseen) | C topology (seen → unseen) |
|---|---|---|---|---|
| WS | 2 | 0.920 → 0.510 | 0.918 → 0.821 | 0.918 → 0.770 |
| WS | 4 | 0.971 → 0.731 | 0.979 → 0.953 | 0.980 → 0.956 |
| WS | 8 | 0.980 → 0.694 | 0.989 → 0.990 | 0.989 → 0.990 |
| ER, BA, SBM | 2–8 | gap ≤ 0.01 | gap ≤ 0.01 | gap ≤ 0.01 |

Graph-bootstrap CIs for held-out WS (B − A): +0.31 [0.28, 0.34] (h=2), +0.17 [0.14, 0.21] (h=4), +0.34 [0.31, 0.37] (h=6),
+0.11 [0.08, 0.15] (h=8). The state-only model is also *unstable* under the shift (h=8 AUC 0.69–0.88 across seeds),
whereas the dynamics model is not (0.99 every seed). *Figure: E345_collapse.*

### E6 — which signals matter [B]
XGB, critical cases, ROC-AUC (mean of 3 seeds). Left: remove one group from B. Right: state + only that group.
| | within h=2 | within h=4 | held-out WS h=2 | held-out WS h=4 |
|---|---|---|---|---|
| A state | 0.895 | 0.951 | 0.510 | 0.731 |
| B full | 0.911 | 0.964 | 0.821 | 0.953 |
| B − rates | 0.910 | 0.964 | 0.832 | 0.946 |
| B − EWS | 0.911 | 0.964 | 0.823 | 0.954 |
| B − cross-layer | 0.909 | 0.964 | 0.868 | 0.948 |
| B − timing | 0.911 | 0.964 | **0.654** | 0.951 |
| state + rates | 0.910 | 0.964 | 0.718 | 0.952 |
| state + EWS | 0.910 | 0.960 | **0.517** | **0.693** |
| state + cross-layer | 0.896 | 0.952 | 0.604 | 0.903 |
| state + timing | 0.911 | 0.963 | **0.821** | 0.904 |

- Within distribution the groups are redundant: any one of rates, timing or EWS recovers almost all of B's gain,
  and removing any single group costs nothing.
- Under the topology shift they are not: **failure timing** carries the earliest transferable signal (h=2: removing it
  drops 0.82 → 0.65; timing alone gives 0.82), **failure rates** take over by h=4 (0.95 alone).
- **Classical EWS (variance, lag-1 AC) do not transfer** (0.52 at h=2, 0.69 at h=4 — no better than the state):
  their values are family-specific (WS lag-1 AC ≈ 0.49 vs. 0.88–0.96 for the others at h=4).
- Cross-layer features slightly *hurt* transfer at h=2 (removing them: 0.82 → 0.87).
*Figure: E6_ablation.*

## 6. Discussion [A+B]
- **Why the state fails to transfer.** ER, BA and SBM share p_c ≈ 0.39, so "collapse if p > 0.39" learned on any of
  them transfers to the others. WS collapses at p ≈ 0.23; a model trained without WS has never seen collapse at that
  attack size. Dynamics features describe *how the cascade is unfolding* (persistence of failures, rate, acceleration),
  which is closer to the branching-process picture of Dilmoney et al. (2025) and does not depend on where p_c lies.
- **Identifiability limit.** ER and SBM have the same degree distribution; with 10% inter-block edges the communities
  are too well connected to change percolation. Predicted: the limit should shift as communities strengthen (future work).
- **Inferred topology is not needed.** The family is partly identifiable, but knowing it does not help a flexible
  predictor and hurts under shift: a 3-class classifier must misassign an unseen family.
- **Practical reading.** Watching the first few rounds of failures is more robust than reasoning from the size of the
  initial hit when the system's structure is uncertain.

## 7. Limitations and conclusion [A+B]
- Synthetic networks; same family in both layers; full one-to-one coupling; random attacks only.
- The transfer result rests on the one family with a distinct p_c (WS). Families with other p_c values (denser
  networks, partial coupling q < 1, different SBM strengths) are needed to show it generalizes.
- p_c used to define the critical subset is estimated from the same data (it selects cases; it is never an input).
- The early-warning indicators are used as empirical features; critical slowing down theory assumes smooth dynamics.
- Path to real data: apply to coupled power–communication models (e.g. RESP's setting) with observed failure logs.

## References
- Barabási, A.-L. & Albert, R. Emergence of scaling in random networks. Science 286, 509 (1999).
- Buldyrev, S. V., Parshani, R., Paul, G., Stanley, H. E. & Havlin, S. Catastrophic cascade of failures in interdependent networks. Nature 464, 1025 (2010).
- Dakos, V. et al. Methods for detecting early warnings of critical transitions in time series illustrated using simulated ecological data. PLoS ONE 7, e41010 (2012).
- Deb, S., Sidheekh, S., Clements, C. F., Krishnan, N. C. & Dutta, P. S. Machine learning methods trained on simple models can predict critical transitions in complex natural systems. R. Soc. Open Sci. 9, 211475 (2022).
- Dilmoney, D., Gross, B., Havlin, S. & Shnerb, N. M. Dynamics of critical cascades in interdependent networks. Phys. Rev. Lett. 134, 187401 (2025).
- Erdős, P. & Rényi, A. On random graphs I. Publ. Math. Debrecen 6, 290 (1959).
- Holland, P. W., Laskey, K. B. & Leinhardt, S. Stochastic blockmodels: first steps. Social Networks 5, 109 (1983).
- Liu, Z. et al. Early predictor for the onset of critical transitions in networked dynamical systems. Phys. Rev. X 14, 031009 (2024).
- Ma, Z., Zeng, C., Zhang, Y.-C. & Bury, T. M. Learning from the past: predicting critical transitions with machine learning trained on surrogates of historical data. arXiv:2410.09707 (Comm. Phys. 2025).
- Machado, S. et al. Recovering the graph underlying networked dynamical systems under partial observability: a deep learning approach. AAAI 2023.
- Parshani, R., Buldyrev, S. V. & Havlin, S. Interdependent networks: reducing the coupling strength leads to a change from a first to second order percolation transition. Phys. Rev. Lett. 105, 048701 (2010).
- Salehpour, A. & Al-Anbagi, I. RESP: A real-time early-stage failure prediction mechanism for smart grid systems. IEEE Systems Journal (2024).
- Scheffer, M. et al. Early-warning signals for critical transitions. Nature 461, 53 (2009).
- Watts, D. J. & Strogatz, S. H. Collective dynamics of 'small-world' networks. Nature 393, 440 (1998).
- Zhou, D. et al. Simultaneous first- and second-order percolation transitions in interdependent networks. Phys. Rev. E 90, 012803 (2014).
