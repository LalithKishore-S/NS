# Step 5 reading — identifying the hidden family (E2)
Code: `person_b_ml/evaluation.py` (graph-grouped splits), `person_b_ml/models.py`, `person_b_ml/mechanism_classifier.py`,
`experiments/04_mechanism_id.py` → `results/E2_table.csv`, `results/figures/E2_mechanism_id.png`

## Paper
★ **Machado, Sridhar, Gil, Henriques, Moura & Santos, *Recovering the graph underlying networked dynamical systems under
partial observability: a deep learning approach*, AAAI 2023** — [arXiv:2208.04405](https://arxiv.org/abs/2208.04405)
- Setting: linear stochastic dynamics on a network; only a *subset* of nodes is observed.
- Method: a feature vector computed from the observed time series (shown to be linearly separable for linked vs.
  unlinked pairs), fed to a CNN that decides **each edge** present/absent.
- Transfers from synthetic random graphs to real networks; works for dense and sparse graphs.
- **How we differ (say this in one breath):** they recover *individual edges* of *one* network from *generic linear
  dynamics*; we infer the *generating family* of a *two-layer interdependent* network from *cascade failure counts* —
  a coarser target, but one that needs far less data and that we then use for collapse prediction.
- Why their edge-level approach does not apply here: we only see aggregate failure fractions per half-step (no node
  ids, by design), so edge recovery is impossible — family-level inference is the most one can ask for.

## Set-up (plan §10 rule)
- 4-class, **held-out instance**: 5-fold StratifiedGroupKFold on graph_id (all p and seeds of a graph in one fold),
  3 seeds, LR / RF / XGB, one cascade at a time. No leave-one-family-out here — a classifier can't predict an unseen class.

## Results (N=2000, 10,800 cascades; chance = 0.25)
| h | LR | RF | XGB |
|---|---|---|---|
| 2 | 0.543 ± 0.008 | 0.586 ± 0.002 | 0.595 ± 0.005 |
| 4 | 0.597 ± 0.007 | 0.652 ± 0.002 | 0.653 ± 0.005 |
| 6 | 0.617 ± 0.002 | 0.668 ± 0.002 | 0.669 ± 0.004 |
| 8 | 0.606 ± 0.004 | 0.671 ± 0.003 | 0.669 ± 0.004 |
(macro-F1, mean ± 95% CI over 3 seeds)

Confusion at h=4 (XGB; rows = true):
| | ER | BA | WS | SBM |
|---|---|---|---|---|
| ER | 0.54 | 0.08 | 0.00 | 0.38 |
| BA | 0.09 | 0.77 | 0.04 | 0.10 |
| WS | 0.00 | 0.07 | **0.93** | 0.00 |
| SBM | 0.49 | 0.11 | 0.00 | 0.40 |

## Reading the result
1. **Topology leaves a signature:** 2.6× chance from a single cascade after only 4 half-steps; most of the gain arrives by h=4.
2. **R2 confirmed exactly where predicted (Step 1):** ER vs SBM is a coin flip. Both have Poisson degrees; with 4 blocks
   of 500 and 10% of edges between blocks, the communities are too well connected to change how failures spread.
   Merge them and the problem is ~90% solvable (ER∪SBM / BA / WS). This is an **identifiability-limit result**, not a bug.
3. Identifiability is roughly flat in p (right panel) — no single attack size is needed.
4. Trees beat LR → the signature is non-linear.

## Next for the R2 boundary (optional, adds a strong figure)
Sweep SBM's community strength (share of edges inside blocks: 0.90 → 0.99) and report where SBM becomes separable from ER.
