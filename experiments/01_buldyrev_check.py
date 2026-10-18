"""Step 2 validation against Buldyrev 2010: ER-ER, <k>=4, q=1.
Theory: collapse when remaining fraction 1-p < 2.4554/<k>  ->  p_c(removal) = 1 - 2.4554/4 = 0.386."""
import time
import numpy as np
import pandas as pd
from person_a_sim.network_generation import make_graph
from person_a_sim.interdependency import couple
from person_a_sim.cascade_simulator import simulate, to_csr

N, K = 1000, 4.0
rows = []
t0 = time.time()
for g in range(10):
    A, B = to_csr(make_graph("ER", N, K, 2 * g)), to_csr(make_graph("ER", N, K, 2 * g + 1))
    dep = couple(N, 1.0, seed=g)
    for p in np.round(np.arange(0.30, 0.47, 0.01), 2):
        r = simulate(A, B, dep, p, seed=g)
        rows.append({"p": p, "S": r["S_final"], "iters": r["n_iter"]})
df = pd.DataFrame(rows).groupby("p").agg(S_mean=("S", "mean"), P_collapse=("S", lambda s: (s < 0.25).mean()),
                                         iters_mean=("iters", "mean"), iters_max=("iters", "max"))
print(df.round(3))
print(f"theory p_c = {1 - 2.4554 / K:.3f};  {len(rows)} cascades in {time.time() - t0:.1f}s")
