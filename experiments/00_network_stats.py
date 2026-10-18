"""Step 1 check: are the four families matched on N and <k>, yet structurally different?"""
import pandas as pd
from person_a_sim.network_generation import make_graph, graph_stats

rows = []
for fam in ["ER", "BA", "WS", "SBM"]:
    for s in range(5):
        rows.append({"family": fam, **graph_stats(make_graph(fam, 1000, 4.0, seed=s))})
df = pd.DataFrame(rows).groupby("family").mean().round(3)
print(df)
df.to_csv("results/network_stats.csv")
