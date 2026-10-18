# Step 2 reading — Buldyrev et al., Nature 464, 1025 (2010)
Code: `person_a_sim/interdependency.py`, `person_a_sim/cascade_simulator.py`, `experiments/01_buldyrev_check.py`

## What to read
- Fig. 1 (the Italy 2003 blackout: power stations ↔ internet nodes) — the motivation.
- Fig. 2 (the cascade drawn step by step) — **this is exactly `simulate()`**.
- The result for coupled ER networks: collapse when the remaining fraction drops below 2.4554/<k>.
- The observation that broader degree distributions (scale-free) are *more* fragile when coupled — the reverse of single networks.

## The cascade in their words → our code
| Paper | Code |
|---|---|
| remove fraction 1−p of nodes in A | `hit = rng.choice(n, p*n)` (our p = fraction *removed*) |
| only the giant component of A stays functional | `alive_A &= giant_mask(adj_A, alive_A)` |
| B nodes depending on failed A nodes fail | `alive_B[dep[dead_A]] = False` |
| same for B, then back to A, until no change | `for t ...: if no change: break` |
| surviving set = mutually connected giant component | `S_final` |

## Reproduced (experiments/01_buldyrev_check.py, ER–ER, N=1000, <k>=4, 10 graphs)
Theory: p_c = 1 − 2.4554/4 = **0.386**. Simulation: survival 0.41 at p=0.37 → 0.04 at p=0.41, collapse probability crosses 50% at p≈0.40.
Small finite-size shift, abrupt (first-order) jump — matches the paper.

## Partial coupling — Parshani, Buldyrev & Havlin, PRL 105, 048701 (2010)
Only a fraction q of nodes are dependent (`dependency_fraction`). Lowering q turns the abrupt jump into a smooth (second-order) transition.
We keep q=1 for now (plan §3); q is a knob for later.

## What this means for our project (preview of R1)
Cascades last only ~3–6 iterations, up to ~12 near p_c. That is very short for "early warning" → Step 3 checks this properly (PRL 2025 plateau).
