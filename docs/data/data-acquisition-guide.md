# KuaiRec 2.0 — Data Acquisition Guide

This guide explains how to download and verify the KuaiRec 2.0 dataset from scratch.
Run these steps from the project root on Windows PowerShell.

---

## Prerequisites

- Backend virtual environment is set up: `backend/.venv`
- Polars, tqdm, and requests are installed in the venv (added in Phase 2)
- Docker is not required for this step

Verify the venv has the required packages:

```powershell
backend\.venv\Scripts\python.exe -c "import polars, tqdm, requests; print('OK')"
```

---

## Download the Dataset

The download script fetches KuaiRec 2.0 from Zenodo (DOI: `10.5281/zenodo.18164998`),
verifies SHA-256 checksums, and extracts the zip archive.

```powershell
# Run from project root
backend\.venv\Scripts\python.exe scripts/data/download_dataset.py
```

**What happens:**
1. Downloads `KuaiRec.zip` (~370 MB) from Zenodo and extracts 7 CSV files
2. Downloads 3 additional CSV files separately from Zenodo
3. Verifies SHA-256 checksums of all 9 files
4. Writes `data/manifests/dataset_manifest.json` with provenance and timestamps

**Expected output files in `data/raw/kuairec/original_downloads/`:**
```
small_matrix.csv              (387 MB)
big_matrix.csv              (1,033 MB)
user_features.csv
user_features_raw.csv
item_categories.csv
video_raw_categories_multi.csv
kuairec_caption_category.csv
item_daily_features.csv
social_network.csv
```

---

## Inspect the Dataset

After downloading, inspect schemas and row counts:

```powershell
backend\.venv\Scripts\python.exe scripts/data/inspect_dataset.py
```

Output: `data/interim/kuairec/inspected/inspection_report.json`

---

## Validate the Dataset

Run integrity checks (duplicates, null rates, referential integrity, watch_ratio range):

```powershell
backend\.venv\Scripts\python.exe scripts/data/validate_dataset.py
```

Output: `data/interim/kuairec/validated/validation_report.json`

Expected result: **16 pass, 3 warning, 1 fail**
- Warnings: 3.89% null timestamps, 12.63% null captions — both expected and documented
- Fail: video_id=1225 present in item_categories — benign, not in interaction matrices

---

## Convert to Parquet

Apply cleaning transforms and convert to Parquet format (zstd compression):

```powershell
backend\.venv\Scripts\python.exe scripts/data/convert_to_parquet.py
```

Output in `data/processed/kuairec/parquet/`:
```
small_matrix.parquet          (101 MB — from 387 MB CSV)
user_features.parquet         (106 KB)
user_features_raw.parquet     (147 KB)
item_categories.parquet        (40 KB)
kuairec_caption_category.parquet (638 KB)
```

---

## Generate Data Quality Report

Produce a comprehensive quality report using DuckDB analytics:

```powershell
backend\.venv\Scripts\python.exe scripts/data/generate_report.py
```

Outputs:
- `data/processed/kuairec/reports/data_quality_report.json`
- `data/processed/kuairec/reports/data_quality_report.md`

---

## Full Pipeline (in order)

```powershell
backend\.venv\Scripts\python.exe scripts/data/download_dataset.py
backend\.venv\Scripts\python.exe scripts/data/inspect_dataset.py
backend\.venv\Scripts\python.exe scripts/data/validate_dataset.py
backend\.venv\Scripts\python.exe scripts/data/convert_to_parquet.py
backend\.venv\Scripts\python.exe scripts/data/generate_report.py
```

---

## Git Policy

**Data files are never committed to Git.** The `.gitignore` excludes:
- `data/raw/`
- `data/interim/`
- `data/processed/`
- `data/manifests/`

The scripts themselves (`scripts/data/*.py`) are committed and versioned.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Download fails partway through | Re-run the download script — it skips files that already exist |
| Checksum mismatch | Delete the affected file and re-run the download script |
| `ModuleNotFoundError: polars` | Run `backend\.venv\Scripts\pip.exe install polars` |
| Out of disk space | Ensure at least 2 GB free; `big_matrix.csv` is 1 GB |
| PowerShell execution policy blocks `.ps1` | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
