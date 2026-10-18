# Decision log (plan §3 says: no change without a team vote — record each one here)

| # | Date | Decision | Why |
|---|---|---|---|
| D1 | 2026-10-18 | Both layers come from the **same family** (ER–ER, BA–BA, WS–WS, SBM–SBM). | Keeps the mechanism label unambiguous; leave-one-family-out stays clean. Mixed pairs = possible later extension. |
| D2 | 2026-10-18 | Time is measured in **half-steps** (A-pruning, B-pruning = 2 per Buldyrev round). | Doubles time resolution of a short cascade (R1). |
| D3 | 2026-10-18 | Observation horizons are **fixed half-step counts** h ∈ {2, 4, 6, 8}, NOT % of each cascade. | "% of cascade" needs the cascade's final length = future information, and length correlates with collapse → label leakage. |
| D4 | 2026-10-18 | **R1 = GO** with the Buldyrev cascade, working in the near-critical window. | Near p_c, collapsing cascades stay above the threshold for a median 6–8 half-steps at N=1000, growing with N (results/R1_summary.csv). |
| D5 | 2026-10-18 | Dataset attack grid is **one common grid for all families** (0.16–0.50), never centred per family. | Family-specific grids would let the classifier read the family off p (WS p_c=0.23 vs ~0.39 for the rest). |
| D6 | 2026-10-18 | Scale to N=2000 for the final dataset once the pipeline works end to end. | Longer plateau → more observable early window (R1 table). |
| D7 | 2026-10-18 | Trajectory schema frozen: one row per (graph_id, seed, p, h) with alive_A/B and new_fail_A/B fractions only. | Aggregates only → no node ids, wiring or dependency map can leak (rule 1). |
| D8 | 2026-10-18 | Collapse scores reported on ALL cases and on the UNDECIDED subset (S(h) ≥ threshold at horizon h). | By h=4, 33% of cascades are already visibly collapsed; predicting those is trivial. |
