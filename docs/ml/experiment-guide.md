# Experiment Guide

**Phase:** 3 — Baseline Recommendation Systems  
**ADR:** ADR-P3-06  
**Last updated:** Phase 3

---

## Overview

This guide explains how to run, record, and interpret recommendation experiments in the
Phase 3 baseline framework. Experiments are tracked as JSON files in `experiments/results/`.

---

## Prerequisites

Ensure the following data files exist before running experiments:

```
data/processed/kuairec/small_matrix.parquet
data/processed/kuairec/item_categories.parquet
data/processed/kuairec/splits/train.parquet
data/processed/kuairec/splits/val.parquet
data/processed/kuairec/splits/test.parquet
data/processed/kuairec/splits/split_manifest.json
```

If the split files are missing, regenerate them:

```powershell
# From project root
backend\.venv\Scripts\python.exe scripts/ml/prepare_splits.py
```

---

## Running All Baselines

The main experiment script fits all 5 baseline algorithms on the training split,
evaluates them on the test split, and writes one JSON result file per algorithm:

```powershell
# From project root
backend\.venv\Scripts\python.exe scripts/ml/run_baselines.py
```

Expected console output:

```
[1/5] RandomRecommender ...
  NDCG@10=0.3236  P@10=0.3233  R@10=0.0095  Coverage@10=0.6674
  Saved → experiments/results/phase3_random_<id>.json

[2/5] PopularityRecommender(weighted_score) ...
  NDCG@10=0.4461  P@10=0.3703  R@10=0.0109  Coverage@10=0.1265
  Saved → experiments/results/phase3_popularity_weighted_score_<id>.json
...

Sanity check: Popularity NDCG@10 (0.4461) >= Random NDCG@10 (0.3236) → PASS

All 5 experiments completed.
```

If the sanity check fails (Popularity NDCG < Random NDCG), there is a bug in the
evaluation pipeline — do not interpret results until the cause is identified.

---

## Result File Format

Each algorithm produces one JSON file:

```json
{
  "experiment_id": "a1b2c3d4",
  "timestamp": "2024-01-15T10:30:00+00:00",
  "phase": "3",
  "algorithm": "PopularityRecommender",
  "config": {
    "metric": "weighted_score"
  },
  "dataset": {
    "name": "kuairec_small_matrix",
    "train_rows": 3328196,
    "test_rows": 674188,
    "n_users": 1411,
    "n_items": 3327
  },
  "results": {
    "k5":  { "ndcg": 0.4812, "precision": 0.4150, "recall": 0.0062, "map": 0.3914 },
    "k10": { "ndcg": 0.4461, "precision": 0.3703, "recall": 0.0109, "map": 0.3568 },
    "k20": { "ndcg": 0.4214, "precision": 0.3304, "recall": 0.0196, "map": 0.3341 },
    "coverage_k10": 0.1265,
    "diversity_k10": null,
    "novelty_k10": 8.7432,
    "evaluated_users": 1411,
    "skipped_users": 0,
    "evaluation_time_seconds": 42.1
  }
}
```

**Note:** `diversity_k10` is `null` when no item→category map is loaded (Random and
Popularity baselines). Category-aware baselines populate this field.

---

## Git and Data Policy

Result JSON files are **excluded from Git** via `.gitignore`:

```
experiments/results/*.json
```

Only `experiments/results/.gitkeep` is committed — this preserves the directory in Git
without committing data files.

**Why?** Result files are large, change frequently during experimentation, and are
reproducible by re-running the scripts. Committing them would pollute the Git history.

If you want to preserve a specific result permanently (e.g., official Phase 3 numbers),
copy the file to `docs/ml/` and commit it there with a descriptive name:

```powershell
Copy-Item experiments/results/phase3_popularity_a1b2c3d4.json `
          docs/ml/phase3-official-results.json
git add docs/ml/phase3-official-results.json
git commit -m "docs: add official Phase 3 baseline results"
```

---

## Running a Single Algorithm

You can import and run any baseline in a Python script or notebook directly:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import polars as pl
from ml.data.loader import load_small_matrix
from ml.data.splits import create_temporal_split
from ml.baselines.popularity_recommender import PopularityRecommender
from ml.evaluation.evaluator import Evaluator
from ml.experiments.recorder import record_experiment

# Load data
df = load_small_matrix()
splits = create_temporal_split(df)

# Fit
rec = PopularityRecommender(metric="weighted_score")
rec.fit(splits.train)

# Evaluate
evaluator = Evaluator(k_values=[5, 10, 20])
results = evaluator.evaluate(rec, train_df=splits.train, test_df=splits.test)

# Record
record_experiment(
    algorithm_name=rec.name,
    config={"metric": "weighted_score"},
    dataset_info={"name": "kuairec_small_matrix"},
    evaluation_results=results,
)
```

---

## Comparing Experiments

To compare two result files:

```python
import json
from pathlib import Path

results_dir = Path("experiments/results")
files = sorted(results_dir.glob("phase3_*.json"))

rows = []
for f in files:
    r = json.loads(f.read_text())
    rows.append({
        "algorithm": r["algorithm"],
        "ndcg@10":   r["results"]["k10"]["ndcg"],
        "p@10":      r["results"]["k10"]["precision"],
        "coverage":  r["results"]["coverage_k10"],
    })

rows.sort(key=lambda x: x["ndcg@10"], reverse=True)
for row in rows:
    print(f"{row['algorithm']:40s}  NDCG@10={row['ndcg@10']:.4f}  Coverage={row['coverage']:.4f}")
```

---

## Adding a New Experiment

1. Implement your algorithm in `ml/baselines/` or `ml/` (Phase 4+).
2. Register it in `scripts/ml/run_baselines.py` (or create a new script).
3. Use `record_experiment()` from `ml/experiments/recorder.py` to persist results.
4. Run all tests: `backend\.venv\Scripts\pytest.exe backend/tests/ -v`.
5. Run ruff: `backend\.venv\Scripts\ruff.exe check ml/ scripts/ml/ backend/tests/ml/`.
6. Commit code, not result JSON files.

---

## Module Reference

| Module | Purpose |
|---|---|
| [`ml/experiments/recorder.py`](../../ml/experiments/recorder.py) | `record_experiment()` — writes JSON |
| [`ml/evaluation/evaluator.py`](../../ml/evaluation/evaluator.py) | `Evaluator` class |
| [`ml/evaluation/metrics.py`](../../ml/evaluation/metrics.py) | 7 metric functions |
| [`ml/data/loader.py`](../../ml/data/loader.py) | Parquet file loaders |
| [`ml/data/splits.py`](../../ml/data/splits.py) | Temporal split logic |
| [`ml/config.py`](../../ml/config.py) | `ExperimentConfig` dataclass |
| [`scripts/ml/run_baselines.py`](../../scripts/ml/run_baselines.py) | Main experiment runner |
| [`scripts/ml/prepare_splits.py`](../../scripts/ml/prepare_splits.py) | Split regeneration script |
