from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GRAPH_META = "graph_meta.parquet"
TRAJECTORIES = "trajectories.parquet"
OUTCOMES = "outcomes.parquet"
THRESHOLD_REPORT = "threshold_report.json"
METRICS = "metrics.json"


def artifact_dir(run_id: str) -> Path:
    d = ROOT / "artifacts" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def features_file(h: int) -> str:
    """Features at a fixed horizon of h half-steps (D3)."""
    return f"features_h{h}.parquet"
