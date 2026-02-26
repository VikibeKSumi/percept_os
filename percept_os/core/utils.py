from dataclasses import dataclass
from pathlib import Path
import json
import time
import uuid


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    artifacts_dir: Path
    logs_file: Path
    run_json: Path
    metrics_json: Path


def new_run_paths(base_dir: str = "runs") -> RunPaths:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    run_dir = base / run_id
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        run_dir=run_dir,
        artifacts_dir=artifacts_dir,
        logs_file=run_dir / "logs.txt",
        run_json=run_dir / "run.json",
        metrics_json=run_dir / "metrics.json",
    )


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def detect_pipeline_type(input_config: dict) -> str:
    """Auto-detect: webcam/video = realtime, image/folder = aerial"""
    # Get source or path, correctly handling number 0 for webcam
    source = input_config.get("source")
    path = input_config.get("path", "")

    if source is not None:
        target = str(source).lower().strip()
    else:
        target = str(path).lower().strip()

    # Realtime / edge_road cases
    if any(x in target for x in ["rtsp://", "http://", "0", ".mp4", ".avi", ".mov", ".mkv"]):
        return "edge_road"

    # Everything else = aerial (images)
    return "aerial_space"

class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000.0