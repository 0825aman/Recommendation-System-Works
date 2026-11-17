# Phase 2 Data Pipeline

This document describes the Phase 2 data engineering pipeline: its stages, design
decisions, tools, and conventions.

---

## Pipeline Overview

```
Zenodo (remote)
     │
     ▼
download_dataset.py     — Download + checksum verify + extract
     │
     ▼
inspect_dataset.py      — Scan schemas, row counts, dtypes
     │
     ▼
validate_dataset.py     — 20 integrity checks → validation_report.json
     │
     ▼
convert_to_parquet.py   — clean_dataset.py + write Parquet (zstd)
     │
     ▼
generate_report.py      — DuckDB analytics → data_quality_report.json/.md
```

All scripts are in `scripts/data/`. Run from the project root with the backend venv.

---

## Tools

| Tool | Role |
|---|---|
| **Polars** | DataFrame operations — cleaning, type casting, schema inspection |
| **DuckDB** | SQL analytics on Parquet files — statistics, distributions |
| **PyArrow** | Parquet I/O backend for Polars |
| **tqdm** | Download progress bars |
| **requests** | HTTP download from Zenodo |
| **rich** | Formatted terminal output |

---

## Script Responsibilities

### `scripts/data/common.py`

Shared constants and utilities:
- `ALL_RAW_FILES` — dict mapping file keys to `Path` objects in `data/raw/`
- `PARQUET_FILES` — dict mapping file keys to `Path` objects in `data/processed/`
- `get_logger(name)` — returns a configured `logging.Logger`
- `format_bytes(n)` — human-readable byte size
- `ensure_directories()` — creates all required `data/` subdirectories

### `scripts/data/download_dataset.py`

Downloads and verifies the KuaiRec 2.0 dataset from Zenodo.
- Skips files that already exist and pass checksum
- Writes `data/manifests/dataset_manifest.json` with URLs, checksums, timestamps

### `scripts/data/inspect_dataset.py`

Uses Polars lazy scanning (`scan_csv`) to report schemas without loading large files.
- Outputs `data/interim/kuairec/inspected/inspection_report.json`

### `scripts/data/validate_dataset.py`

Runs 20 explicit integrity checks. Each check returns a dict with `status` in
`{pass, warning, fail, skip}`. Checks include:

- Duplicate key detection
- Non-negative column constraints
- Null rate thresholds
- Watch ratio range analysis (values > 1.0 are valid)
- Referential integrity (small_matrix → user_features, item_categories)
- Cross-file user count consistency
- video_id=1225 absence check (v2.0 marker)

Outputs `data/interim/kuairec/validated/validation_report.json`.

### `scripts/data/clean_dataset.py`

Applies documented, non-destructive transforms to 5 Phase 2 files.
Every transform logs rows affected. **Zero rows are dropped.**

| File | Rules applied |
|---|---|
| `small_matrix` | Parse `time` to Datetime, cast `date` Float64→Int64, confirm Int64 durations |
| `user_features` | Cast 10 binary (0/1) columns to Boolean |
| `user_features_raw` | Cast 9 binary columns to Boolean, cast `os_version` to String |
| `item_categories` | Add `feat_parsed` column (parse `"[27, 9]"` strings to `List[Int64]`) |
| `kuairec_caption_category` | No transforms — schema already clean, nulls documented |

Outputs `data/interim/kuairec/intermediate/cleaning_log.json`.

### `scripts/data/convert_to_parquet.py`

Calls `clean_all()` from `clean_dataset.py`, then writes each DataFrame to Parquet
with zstd compression. Verifies row counts after writing.

Outputs 5 `.parquet` files and `data/processed/kuairec/parquet/parquet_manifest.json`.

### `scripts/data/generate_report.py`

Uses DuckDB to run SQL analytics on the Parquet files:
- Row/column counts
- Null rate per column
- watch_ratio distribution statistics
- Category distribution (top categories)
- User activity distribution

Outputs `data/processed/kuairec/reports/data_quality_report.{json,md}`.

---

## Conventions

### No raw file modification

Raw CSV files in `data/raw/` are never modified. All transforms produce new files in
`data/interim/` or `data/processed/`. This is enforced by convention, not by code.

### Non-destructive cleaning

No rows are dropped during Phase 2 cleaning. All decisions about filtering, capping,
or imputation are deferred to Phase 3 (feature engineering), where they can be
validated against model performance metrics.

### Every anomaly is documented, not silently fixed

If a value is unexpected (3.89% null timestamps, watch_ratio = 571.52), it is logged
in the validation report with a clear status and note. It is never silently coerced.

### Parquet as the analytical format

Parquet provides:
- ~3–4× compression vs CSV (101 MB vs 387 MB for small_matrix)
- Native type preservation (Datetime, Boolean, List[Int64])
- Columnar format suitable for Polars and DuckDB
- Fast reads for specific columns in Phase 3 feature engineering

---

## Adding a New Pipeline Stage

1. Create `scripts/data/<stage_name>.py`
2. Import shared utilities from `scripts/data/common.py`
3. Add `sys.path.insert(0, str(Path(__file__).resolve().parents[2]))` at the top
4. Write tests in `backend/tests/data/test_<stage_name>.py`
5. Document any anomalies discovered in the validation report
6. Update this document and `docs/data/dataset-overview.md`
