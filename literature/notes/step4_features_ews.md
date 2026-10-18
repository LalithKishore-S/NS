# Step 4 reading — early-warning features
Code: `common/contracts.py` (frozen trajectory schema), `person_a_sim/pipeline.py`, `common/dummy.py`,
`person_b_ml/feature_extraction.py`, `person_b_ml/features.py`, `tests/test_no_leakage.py`, `experiments/03_dataset_and_features.py`

## Papers
1. ★ **Deb, Sidheekh, Clements, Krishnan & Dutta, *Machine learning methods trained on simple models can predict critical
   transitions in complex natural systems*, R. Soc. Open Sci. 9(2), 211475 (2022)** — [doi:10.1098/rsos.211475](https://doi.org/10.1098/rsos.211475)
   - EWSNet: a deep net trained on simple simulated models, tested on real climate / ecological collapses.
   - Needs no knowledge of the system's structure; can also tell catastrophic from non-catastrophic transitions.
   - Compares against the classical generic indicators (variance, lag-1 autocorrelation).
   - **For us:** we take those classical indicators as *baseline features* (`ews_var`, `ews_ac1`) and use classical
     ML, not a deep net (plan §3: CPU, no neural nets). Their "catastrophic vs non-catastrophic" = our collapse label.
2. **Scheffer, Bascompte, Brock et al., *Early-warning signals for critical transitions*, Nature 461, 53–59 (2009)**
   - Origin of the indicators: near a tipping point, recovery from small shocks slows down ("critical slowing down")
     → the time series gets more variable and more correlated with its own past (variance ↑, lag-1 AC ↑).
3. **Dakos et al., *Methods for detecting early warnings of critical transitions in time series illustrated using
   simulated ecological data*, PLoS ONE 7(7), e41010 (2012)**
   - How to compute the indicators properly, and why short windows make them unreliable → our `ews_reliable` flag.

## Honest caveat (put this in the paper)
Critical slowing down is derived for smooth systems near a bifurcation. Our cascade is a *branching process* (PRL 2025),
not a smooth system. Our reading of the indicators: lag-1 AC of the failure-rate series measures how *persistent*
the failure activity is — when each failure triggers ~1 more (branching ratio ≈ 1, the plateau), the rate stays
roughly constant → high AC. So the indicators are used as empirical features, not as a theory claim.

## Features (all from aggregate counts up to horizon h only)
| group | features | backed by |
|---|---|---|
| state | p, failed_A, failed_B | plan's mandatory state-only baseline (Model A) |
| temporal | rate_last, rate_mean, accel, n_active, stopped | plan §9 / RESP-style early-stage features |
| EWS | ews_var, ews_ac1, ews_reliable | Scheffer 2009, Dakos 2012, Deb 2022 |
| cross-layer | cross_ratio_last, cross_share_B | interdependence (Buldyrev 2010) |
| timing | t_mean, t_std, conc_last | plan §9 "distribution of node failure times" (aggregated — node ids are not exposed) |

Dropped: `failed_B − failed_A` — identically 0 at even h when every node has a partner (the two layers mirror after each full round).

## Dataset (configs/grid.yaml: N=2000, 50 graphs × 4 families × 18 p × 3 seeds = 10,800 cascades, 82 s)
- p_c at N=2000: ER 0.394, BA 0.385, WS 0.225, SBM 0.392 (consistent with N=1000).
- Collapse prevalence: ER 0.32, BA 0.35, SBM 0.33, **WS 0.79** → prevalence shifts by family (matters for leave-one-family-out).

## Important finding for Step 6
| h | already visibly collapsed (S<0.25) | still undecided but will collapse |
|---|---|---|
| 2 | 15% | 30% |
| 4 | 33% | 12% |
| 6 | 39% | 6% |
| 8 | 41% | 4% |
Far from p_c, cascades are fast, so by h=4 most collapses are *already visible*. Predicting those is trivial.
→ Step 6 must report scores on **all cases** AND on the **undecided subset** (not yet collapsed at the horizon) —
the undecided subset is the honest early-warning test.
