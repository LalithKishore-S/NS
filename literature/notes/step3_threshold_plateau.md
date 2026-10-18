# Step 3 reading — critical threshold and the early-warning window
Code: `person_a_sim/threshold_analysis.py`, `experiments/02_threshold_and_R1.py`
Outputs: `results/figures/E1_survival_vs_p.png`, `results/figures/R1_length_vs_N.png`, `results/threshold_report.json`, `results/R1_summary.csv`

## Papers
1. ★ **Dilmoney, Gross, Havlin & Shnerb, *Dynamics of critical cascades in interdependent networks*, PRL 134, 187401 (2025)** — [arXiv:2504.06862](https://arxiv.org/abs/2504.06862)
   - At criticality the cascade is a *critical branching process*: each failure causes ~1 more on average.
   - Failures make the network weaker, so the branching factor creeps up. If the cascade dies out during this
     "plateau" the system survives; otherwise it collapses abruptly.
   - They study (i) collapse probability, (ii) avalanche duration, (iii) **plateau length**, vs. system size and trigger size.
   - **For us:** the plateau is the time window in which an early-warning model can see the system "wobbling" before it falls.
2. **Zhou, Bashan, Cohen, Berezin, Shnerb & Havlin, *Simultaneous first- and second-order percolation transitions in interdependent networks*, Phys. Rev. E 90, 012803 (2014)** — [arXiv:1211.2330](https://arxiv.org/abs/1211.2330)
   - Number of cascade iterations τ diverges at p_c; at p_c, ⟨τ⟩ ~ N^{1/3} (mean-field estimate N^{1/4}).
   - Near p_c the giant component of A decays slowly in a plateau whose length grows with N.
   - **For us:** the reason R1 is solvable — the window gets longer as networks get bigger and as we move closer to p_c.

## Our results (N=1000, 20 graphs × 31 p-values × 3 seeds per family = 7,440 cascades, 16 s)
| family | p_c (P(collapse)=0.5) | 95% CI | 10→90% width | reading |
|---|---|---|---|---|
| ER  | 0.398 | 0.393–0.403 | 0.056 | theory 0.386 (Buldyrev) — small finite-size shift ✅ |
| BA  | 0.383 | 0.370–0.393 | 0.116 | collapses earlier and more gradually: hubs make coupled networks *more* fragile, as Buldyrev reports |
| WS  | 0.228 | 0.224–0.231 | 0.042 | by far the most fragile: lattice-like, so removed nodes cut long paths |
| SBM | 0.393 | 0.384–0.402 | 0.067 | almost identical to ER (same Poisson degrees) → hardest pair to separate (R2) |

## R1 — go/no-go
Cascades started within ±0.02 of each family's p_c:
- mean length 7–11 half-steps at N=1000, growing to 11–15 at N=4000 (slope ≈ 0.23 on log-log, between the N^{1/4} and N^{1/3} predictions; we average over a window, not exactly at p_c).
- **collapsing** cascades stay above S=0.25 for a median **6–8 half-steps** at N=1000 (8–10 at N=4000).
- **Verdict: GO.** Horizons h = 2, 4, 6 half-steps all fall before the typical collapse. Use N=2000 for the final dataset for extra room (DECISIONS.md D4, D6).

## Already a family signature
p_c and transition width differ by family (WS ≪ others; BA wider). This is the first evidence for Problem A — but it is visible only
across *many* attacks; Step 5 asks whether a *single partial cascade* reveals it.
