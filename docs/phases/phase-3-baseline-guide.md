# Phase 3 Setup Guide — Baseline Recommendation Systems

**Phase:** 3  
**Status:** Complete  
**Prerequisites:** Phase 1 (backend skeleton) and Phase 2 (data engineering) must be complete.

---

## What Was Built in Phase 3

Phase 3 implements a complete offline evaluation framework and five non-ML baseline
recommendation algorithms on the KuaiRec 2.0 small_matrix dataset.

### New modules

| Module | Purpose |
|---|---|
| `ml/` | Standalone ML package (no FastAPI/DB imports) |
| `ml/baselines/` | 5 baseline recommenders + abstract base |
| `ml/evaluation/` | 7 offline metrics + Evaluator class |
| `ml/features/` | Item category map + user affinity features |
| `ml/data/` | Parquet data loaders + temporal split logic |
| `ml/experiments/` | JSON experiment result recorder |

### New scripts

| Script | Purpose |
|---|---|
| `scripts/ml/run_eda.py` | EDA → `eda_report.json` + `eda_report.md` |
| `scripts/ml/prepare_splits.py` | Temporal train/val/test split |
| `scripts/ml/run_baselines.py` | Fit + evaluate all 5 baselines |

### New tests

72 new Pytest tests covering splits, all 7 metrics, and all 5 baseline algorithms.

---

## Prerequisites

Verify Phase 2 data files exist:

```powershell
# From project root — these must all exist
Test-Path data\processed\kuairec\small_matrix.parquet
Test-Path data\processed\kuairec\item_categories.parquet
Test-Path data\processed\kuairec\user_features.parquet
```

If any file is missing, re-run the Phase 2 pipeline:

```powershell
backend\.venv\Scripts\python.exe scripts/data/convert_to_parquet.py
backend\.venv\Scripts\python.exe scripts/data/clean_dataset.py
```

---

## Environment Setup

No new Python packages are required for Phase 3. The Phase 2 environment is sufficient:

```powershell
# Verify virtual environment is active
backend\.venv\Scripts\python.exe --version
# Expected: Python 3.14.x

# Verify Polars is available
backend\.venv\Scripts\python.exe -c "import polars; print(polars.__version__)"
# Expected: 2.x.x
```

---

## Step 1 — Generate Splits

Run the temporal split script to create train/val/test Parquet files:

```powershell
# From project root
backend\.venv\Scripts\python.exe scripts/ml/prepare_splits.py
```

Expected output:

```
Creating temporal split...
  Total rows:          4,676,570
  Null-timestamp rows:   181,990  (assigned to train)
  Timed rows:          4,494,580

Split sizes:
  train: 3,328,196 rows (71.11%)
  val:     674,186 rows (14.41%)
  test:    674,188 rows (14.41%)

Temporal integrity check:
  train max ts: 1595980769.0
  val   min ts: 1595980782.0  [OK — no leakage]
  val   max ts: 1596662994.0
  test  min ts: 1596663028.0  [OK — no leakage]

Saved:
  data\processed\kuairec\splits\train.parquet
  data\processed\kuairec\splits\val.parquet
  data\processed\kuairec\splits\test.parquet
  data\processed\kuairec\splits\split_manifest.json
```

---

## Step 2 — Run All Baselines

```powershell
# From project root
backend\.venv\Scripts\python.exe scripts/ml/run_baselines.py
```

This will fit all 5 baseline algorithms on the training split, evaluate them on the test
split, and write JSON result files to `experiments/results/`.

Expected final output:

```
Sanity check: Popularity NDCG@10 (0.4461) >= Random NDCG@10 (0.3236) → PASS

All 5 experiments completed.
```

If the sanity check fails, the evaluation pipeline has a bug — do not proceed.

---

## Step 3 — Run Tests

```powershell
# From project root
backend\.venv\Scripts\pytest.exe backend/tests/ -v
```

Expected: **134 tests pass, 0 failures**.

The 72 Phase 3 tests cover:
- `backend/tests/ml/test_splits.py` — 11 split correctness tests
- `backend/tests/ml/test_metrics.py` — 29 metric unit tests
- `backend/tests/ml/test_baselines_basic.py` — 15 tests (Random + Popularity)
- `backend/tests/ml/test_baselines_advanced.py` — 17 tests (Category + Trending + UserPref)

---

## Step 4 — Run Quality Gates

```powershell
# From project root
backend\.venv\Scripts\ruff.exe check ml/ scripts/ml/ backend/tests/ml/
```

Expected: `All checks passed!`

---

## Baseline Results Reference

Phase 3 official results on KuaiRec 2.0 small_matrix test split (K=10):

| Algorithm | NDCG@10 | P@10 | R@10 | Coverage@10 |
|---|---|---|---|---|
| RandomRecommender | 0.3236 | 0.3233 | 0.0095 | 0.6674 |
| PopularityRecommender | **0.4461** | **0.3703** | **0.0109** | 0.1265 |
| CategoryPopularityRecommender | 0.4096 | 0.3402 | 0.0100 | 0.1532 |
| TrendingRecommender | 0.3096 | 0.2694 | 0.0080 | 0.1151 |
| UserPreferenceRecommender | 0.3396 | 0.3516 | 0.0103 | **0.4824** |

**Sanity check:** Popularity NDCG@10 (0.4461) ≥ Random NDCG@10 (0.3236) ✅

> **Note on dense matrix:** The 99.6% matrix density inflates P@10 and R@10 values
> compared to realistic sparse datasets. Phase 4 big_matrix evaluation will show more
> typical values.

---

## EDA Report (Optional)

To regenerate the exploratory data analysis report:

```powershell
backend\.venv\Scripts\python.exe scripts/ml/run_eda.py
```

Output files (excluded from Git):
- `data/processed/kuairec/reports/eda_report.json`
- `data/processed/kuairec/reports/eda_report.md`

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'ml'`

The `ml/` directory is not on the Python path. All scripts must be run from the
**project root**, not from inside `ml/` or `scripts/`.

```powershell
# Wrong — run from project root instead
cd scripts/ml
python run_baselines.py  # ❌

# Correct
cd C:\Users\Edunet Foundation\Desktop\IBM Bob Projects\Recommendation System
backend\.venv\Scripts\python.exe scripts/ml/run_baselines.py  # ✅
```

### `FileNotFoundError: data/processed/kuairec/splits/train.parquet`

Run `scripts/ml/prepare_splits.py` first (Step 1 above).

### Sanity check fails

If `Popularity NDCG@10 < Random NDCG@10`:
1. Verify split files are not empty: `polars.read_parquet("data/.../train.parquet").height`
2. Verify `watch_ratio` column exists in train split.
3. Re-run `prepare_splits.py` and then `run_baselines.py`.

---

## Documentation Reference

| Document | Location |
|---|---|
| Evaluation methodology | `docs/ml/evaluation-methodology.md` |
| Baseline algorithm details | `docs/ml/baseline-algorithms.md` |
| Experiment guide | `docs/ml/experiment-guide.md` |
| Architecture decisions | `docs/decisions/architecture-decisions.md` (ADR-P3-01 to ADR-P3-07) |
| Phase 2 data pipeline | `docs/phases/phase-2-data-engineering-guide.md` |
