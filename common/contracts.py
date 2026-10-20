"""Frozen Day-0 contracts shared by Person A and Person B."""
from pydantic import BaseModel, field_validator

FAMILIES = ("ER", "BA", "WS", "SBM")


class ExperimentConfig(BaseModel):
    run_id: str
    families: list[str] = list(FAMILIES)   # families to generate (same family in both layers)
    n_graphs_per_family: int = 30
    n_nodes: int = 1000
    avg_degree: float = 4.0
    coupling: str = "one_to_one"
    dependency_fraction: float = 1.0
    attack_grid: list[float] = [round(0.16 + 0.02 * i, 2) for i in range(18)]  # 0.16..0.50, common to all families (D5)
    attack: str = "random"          # random | degree | betweenness | kcore
    horizons: list[int] = [2, 4, 6, 8]     # observation horizons in half-steps (D2, D3)
    n_seeds: int = 3                       # attack seeds per (graph, p)
    seed: int = 42
    collapse_threshold: float = 0.25

    @field_validator("families")
    @classmethod
    def _known(cls, v):
        bad = set(v) - set(FAMILIES)
        if bad:
            raise ValueError(f"unknown families {bad}")
        return v


class GraphMeta(BaseModel):
    graph_id: str
    family_A: str
    family_B: str
    n_nodes: int
    realized_avg_degree_A: float
    realized_avg_degree_B: float
    clustering_A: float
    coupling: str
    dependency_fraction: float
    seed: int


class ThresholdReport(BaseModel):
    family_pair: str
    p_c_estimate: float
    transition_sharpness: float
    seed: int


class MetricsRow(BaseModel):
    experiment: str
    model: str
    observation_fraction: float
    held_out_family: str | None = None
    macro_f1: float | None = None
    roc_auc: float | None = None
    pr_auc: float | None = None
    balanced_acc: float | None = None
    seed: int


# ---- trajectories.parquet schema (frozen) -------------------------------------------------
# one row per (graph_id, seed, p, h), h = half-step 0..end of cascade
#   alive_A, alive_B   : fraction of layer A / B still functional after half-step h
#   new_fail_A/B       : fraction of layer A / B that failed AT half-step h
# Only aggregate counts are stored: no node ids, no adjacency, no dependency map (leakage rule 1).
TRAJ_KEYS = ["graph_id", "seed", "p", "h"]
TRAJ_COLS = TRAJ_KEYS + ["alive_A", "alive_B", "new_fail_A", "new_fail_B"]
OUTCOME_COLS = ["graph_id", "family", "seed", "p", "S_final", "collapsed", "n_half"]


def validate_trajectories(df) -> None:
    missing = set(TRAJ_COLS) - set(df.columns)
    assert not missing, f"trajectories missing {missing}"
    assert ((df[["alive_A", "alive_B"]] >= -1e-12) & (df[["alive_A", "alive_B"]] <= 1 + 1e-12)).all().all()
    g = df.sort_values(TRAJ_KEYS).groupby(["graph_id", "seed", "p"])
    assert (g.alive_A.diff().fillna(0) <= 1e-12).all(), "alive_A must be non-increasing (no recovery)"


def validate_outcomes(df) -> None:
    missing = set(OUTCOME_COLS) - set(df.columns)
    assert not missing, f"outcomes missing {missing}"
    assert set(df.family) <= set(FAMILIES)
