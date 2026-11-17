"""JSON experiment result recorder.

Saves experiment results to experiments/results/ as structured JSON files.
No external dependencies — standard library only.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_DIR = _PROJECT_ROOT / "experiments" / "results"


def record_experiment(
    algorithm_name: str,
    config: dict,  # type: ignore[type-arg]
    dataset_info: dict,  # type: ignore[type-arg]
    evaluation_results: dict,  # type: ignore[type-arg]
    results_dir: Path | None = None,
) -> Path:
    """Write experiment results to a JSON file.

    Args:
        algorithm_name:     Name of the recommender algorithm.
        config:             Experiment configuration dict.
        dataset_info:       Dataset and split metadata.
        evaluation_results: Output from Evaluator.evaluate().
        results_dir:        Output directory. Defaults to experiments/results/.

    Returns:
        Path to the written JSON file.
    """
    out_dir = results_dir or _RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    experiment_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now(UTC).isoformat()

    record = {
        "experiment_id": experiment_id,
        "timestamp": timestamp,
        "phase": "3",
        "algorithm": algorithm_name,
        "config": config,
        "dataset": dataset_info,
        "results": evaluation_results,
    }

    filename = f"phase3_{algorithm_name.replace(' ', '_')}_{experiment_id}.json"
    out_path = out_dir / filename

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)

    return out_path
