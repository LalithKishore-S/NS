"""Person A entry point: build networks, run cascades, write the contract files.

Writes to artifacts/{run_id}/: graph_meta.parquet, trajectories.parquet, outcomes.parquet, threshold_report.json.
Adjacency and dependency maps never leave this function (leakage rule 1).
"""
import json
import numpy as np
import pandas as pd
import networkx as nx

from common.contracts import (ExperimentConfig, GraphMeta, ThresholdReport,
                              validate_trajectories, validate_outcomes)
from common import paths
from person_a_sim.network_generation import make_graph, avg_degree
from person_a_sim.interdependency import couple
from person_a_sim.cascade_simulator import simulate, to_csr
from person_a_sim.threshold_analysis import estimate_pc
from person_a_sim.attacks import attack_order


def _traj_rows(gid, seed, p, out, n):
    fa, fb = out["fail_t_A"], out["fail_t_B"]
    H = max(int(fa.max()), int(fb.max()), 0)
    h = np.arange(H + 1)
    newA = np.array([(fa == t).sum() for t in h]) / n
    newB = np.array([(fb == t).sum() for t in h]) / n
    return pd.DataFrame({"graph_id": gid, "seed": seed, "p": p, "h": h,
                         "alive_A": 1 - np.cumsum(newA), "alive_B": 1 - np.cumsum(newB),
                         "new_fail_A": newA, "new_fail_B": newB})


def build_networks_and_cascades(cfg: ExperimentConfig) -> list[ThresholdReport]:
    out_dir = paths.artifact_dir(cfg.run_id)
    metas, trajs, outs = [], [], []
    for fi, fam in enumerate(cfg.families):
        for g in range(cfg.n_graphs_per_family):
            s = cfg.seed + 100_000 * fi + 1000 * g
            GA, GB = make_graph(fam, cfg.n_nodes, cfg.avg_degree, s), make_graph(fam, cfg.n_nodes, cfg.avg_degree, s + 1)
            order_A = attack_order(GA, cfg.attack, seed=s + 5)
            A, B = to_csr(GA), to_csr(GB)
            dep = couple(cfg.n_nodes, cfg.dependency_fraction, seed=s + 2)
            gid = f"{fam}_{g:03d}"
            metas.append(GraphMeta(graph_id=gid, family_A=fam, family_B=fam, n_nodes=cfg.n_nodes,
                                   realized_avg_degree_A=avg_degree(GA), realized_avg_degree_B=avg_degree(GB),
                                   clustering_A=nx.average_clustering(GA), coupling=cfg.coupling,
                                   dependency_fraction=cfg.dependency_fraction, seed=s).model_dump())
            for p in cfg.attack_grid:
                for r in range(cfg.n_seeds):
                    o = simulate(A, B, dep, p, seed=s + 3 + r, order=order_A)
                    trajs.append(_traj_rows(gid, r, p, o, cfg.n_nodes))
                    outs.append({"graph_id": gid, "family": fam, "seed": r, "p": p, "S_final": o["S_final"],
                                 "collapsed": o["S_final"] < cfg.collapse_threshold, "n_half": o["n_half"], "attack": cfg.attack})
    traj, outc = pd.concat(trajs, ignore_index=True), pd.DataFrame(outs)
    validate_trajectories(traj)
    validate_outcomes(outc)
    pd.DataFrame(metas).to_parquet(out_dir / paths.GRAPH_META)
    traj.to_parquet(out_dir / paths.TRAJECTORIES)
    outc.to_parquet(out_dir / paths.OUTCOMES)

    reports = []
    for fam in cfg.families:
        est = estimate_pc(outc[outc.family == fam])
        reports.append(ThresholdReport(family_pair=f"{fam}-{fam}", p_c_estimate=est["p_c"],
                                       transition_sharpness=est["sharpness"], seed=cfg.seed))
    json.dump([r.model_dump() for r in reports], open(out_dir / paths.THRESHOLD_REPORT, "w"), indent=2)
    return reports
