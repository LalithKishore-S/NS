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
    attack_grid: list[float] = [round(0.02 * i, 2) for i in range(31)]  # 0.00..0.60
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
