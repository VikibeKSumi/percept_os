from __future__ import annotations
import json
import sys
from pathlib import Path

from .core.logger import RunLogger
from .core.utils import new_run_paths, write_json, Timer, detect_pipeline_type
from .pipelines.aerial import run as aerial_run
from .pipelines.realtime import run as realtime_run


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python -m percept_os.run jobs/aerial_demo.json")
        print("       or jobs/edge_demo.json")
        return 2

    job_path = argv[1]
    data = json.loads(Path(job_path).read_text(encoding="utf-8"))

    # Auto detect task if "auto"
    if data.get("task") in (None, "auto", ""):
        data["task"] = detect_pipeline_type(data.get("input", {}))

    job = data  # simple dict now

    paths = new_run_paths("runs")
    logger = RunLogger(paths.logs_file)
    timer = Timer()

    ctx = {"paths": paths, "logger": logger, "timer": timer}

    logger.info(f"Run dir: {paths.run_dir}")
    logger.info(f"Task: {job.get('task')}")
    logger.info(f"Input: {job.get('input')}")

    # Save job copy
    write_json(paths.run_json, {"job": job, "argv": argv})

    try:
        if job.get("task") == "aerial_space":
            result = aerial_run(job, ctx)
        elif job.get("task") == "edge_road":
            result = realtime_run(job, ctx)
        else:
            raise ValueError(f"Unknown task: {job.get('task')}")

        metrics = {
            "task": job.get("task"),
            "elapsed_ms": round(timer.ms(), 3),
            "status": "success"
        }
        write_json(paths.metrics_json, {"metrics": metrics, "result": result or {}})

        logger.ok(f"✅ Done in {metrics['elapsed_ms']} ms")
        logger.info(f"Artifacts saved in → {paths.artifacts_dir}")
        return 0

    except Exception as e:
        logger.err(f"💥 Run failed: {e}")
        raise


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))