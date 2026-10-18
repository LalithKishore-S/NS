"""make_dummy_artifacts(cfg): random data in exactly the contract shapes, so Person B can build
features/models before the real simulator exists (plan §8 'Day-0 parallel start')."""
import numpy as np
import pandas as pd

from common import paths
from common.contracts import ExperimentConfig, validate_trajectories, validate_outcomes


def make_dummy_artifacts(cfg: ExperimentConfig) -> None:
    rng = np.random.default_rng(cfg.seed)
    d = paths.artifact_dir(cfg.run_id)
    trajs, outs, metas = [], [], []
    for fam in cfg.families:
        for g in range(cfg.n_graphs_per_family):
            gid = f"{fam}_{g:03d}"
            metas.append({"graph_id": gid, "family_A": fam, "family_B": fam, "n_nodes": cfg.n_nodes,
                          "realized_avg_degree_A": cfg.avg_degree, "realized_avg_degree_B": cfg.avg_degree,
                          "clustering_A": 0.0, "coupling": cfg.coupling,
                          "dependency_fraction": cfg.dependency_fraction, "seed": g})
            for p in cfg.attack_grid:
                for r in range(cfg.n_seeds):
                    H = int(rng.integers(2, 15))
                    newA = rng.dirichlet(np.ones(H + 1)) * rng.uniform(p, 1)
                    newB = rng.dirichlet(np.ones(H + 1)) * rng.uniform(0, 1)
                    newA[0], newB[0] = p, 0
                    newA, newB = newA * min(1, 1 / newA.sum()), newB * min(1, 1 / max(newB.sum(), 1e-9))
                    aA, aB = 1 - np.cumsum(newA), 1 - np.cumsum(newB)
                    trajs.append(pd.DataFrame({"graph_id": gid, "seed": r, "p": p, "h": np.arange(H + 1),
                                               "alive_A": aA.clip(0, 1), "alive_B": aB.clip(0, 1),
                                               "new_fail_A": newA, "new_fail_B": newB}))
                    outs.append({"graph_id": gid, "family": fam, "seed": r, "p": p, "S_final": aA[-1].clip(0, 1),
                                 "collapsed": aA[-1] < cfg.collapse_threshold, "n_half": H})
    traj, outc = pd.concat(trajs, ignore_index=True), pd.DataFrame(outs)
    validate_trajectories(traj); validate_outcomes(outc)
    pd.DataFrame(metas).to_parquet(d / paths.GRAPH_META)
    traj.to_parquet(d / paths.TRAJECTORIES)
    outc.to_parquet(d / paths.OUTCOMES)
