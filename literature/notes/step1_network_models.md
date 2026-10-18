# Step 1 reading — the four hidden generating families

| Family | Paper | Mechanism | Signature we expect |
|---|---|---|---|
| ER | Erdős & Rényi, 1959 | every pair linked with equal probability | Poisson degrees, no hubs, ~0 clustering |
| BA | Barabási & Albert, Science 1999 | growth + preferential attachment ("rich get richer") | heavy-tailed degrees, a few big hubs |
| WS | Watts & Strogatz, Nature 1998 | ring lattice, small fraction of edges rewired | high clustering + short paths, near-uniform degree |
| SBM | Holland, Laskey & Leinhardt, 1983 | nodes in blocks; dense inside, sparse between | communities; locally ER-like |

Why degree matching: if BA had <k>=6 and ER <k>=4, a classifier would separate them by density alone.
Every family here is built at N=1000, <k>≈4, so any difference the model finds must come from *structure*.

Observation from results/network_stats.csv: ER and SBM have almost identical degree statistics
(both Poisson-like). They differ only in community structure → likely the hardest pair (risk R2).
