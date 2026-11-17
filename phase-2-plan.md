# Phase 2 — Dataset Acquisition & Data Engineering
## Complete Implementation Plan

---

## Top-Level Overview

**Goal:** Establish a reproducible, well-documented data engineering foundation using the KuaiRec 2.0 dataset — from acquisition through validation, cleaning, standardization, Parquet conversion, and quality reporting.

**Scope:** Data acquisition and engineering only. No recommendation models, no ML training, no changes to the existing FastAPI application or PostgreSQL schema.

**Approach:** Modular pipeline scripts in `scripts/data/`, each independently executable and testable. One work package at a time, verified before proceeding.

---

## Environment Assessment (Verified)

| Item | Confirmed State |
|---|---|
| Python | **3.14.5** in `backend/.venv` (pip3.14 binary confirmed) |
| FastAPI | 0.163.1 |
| SQLAlchemy | 2.1.9 |
| Pydantic | 2.14.3 |
| Pytest | 9.2.0 |
| Ruff | installed |
| Uvicorn | 0.54.0 |
| Polars | **NOT installed** — Phase 2 addition |
| DuckDB | **NOT installed** — Phase 2 addition |
| Existing tests | 18/18 passing — must remain passing throughout |
| `data/` structure | `raw/`, `processed/`, `interim/` with `.gitkeep` |
| `scripts/` | Empty directory |
| `backend/app/models/` | Empty directory |
| `backend/app/schemas/` | Empty directory |
| `backend/app/services/` | Empty directory |

**Important correction:** The virtual environment runs Python **3.14**, not 3.13 as originally specified. This is because the installed Python on the machine is `Python313` by folder name but `3.14.5` by actual version. All Phase 2 libraries are compatible with 3.14.

---

## KuaiRec 2.0 Dataset Assessment (Verified from Official Sources)

### Source
- **Official site:** https://kuairec.com/
- **Download host:** Zenodo (DOI: `10.5281/zenodo.18164998`)
- **License:** CC BY 4.0 (Creative Commons Attribution 4.0 International)
  - Free to use for research and commercial purposes
  - Attribution required when publishing
  - **No restrictions on local use or academic projects**
- **Version:** 2.0 (updated 2022-05-16, raw features added 2026-01-03)
- **Paper:** CIKM 2022, Gao et al.

### Verified File Manifest (from Zenodo API)

| File | Size | MD5 Checksum | Notes |
|---|---|---|---|
| `KuaiRec.zip` | **411.8 MB** | `261550d472c48eff4990fb13c0e5bcf7` | Contains 6 core files |
| `kuairec_caption_category.csv` | **1.9 MB** | `31bc38cdccdf75a71df137779035f8cb` | LLM-friendly, also in zip |
| `video_raw_categories_multi.csv` | **1.6 MB** | `d05eea147135d2cdf7759fba5c0d70d4` | Added Jan 2026 |
| `user_features_raw.csv` | **1.5 MB** | `3969b8120035e7ced36d56926a7cbd24` | Added Jan 2026 |
| **Total** | **~417 MB** | | |

### Files Inside `KuaiRec.zip`

```
KuaiRec/data/
├── big_matrix.csv            # 12,530,806 interactions, 7,176 users × 10,728 items, density 16.3%
├── small_matrix.csv          # 4,676,570 interactions, 1,411 users × 3,327 items, density 99.6%
├── social_network.csv        # User social graph (friend lists)
├── user_features.csv         # 7,176 users, 31 columns (anonymized features)
├── item_categories.csv       # Video tags (list format per video)
└── item_daily_features.csv   # Per-video daily statistics (53 columns)
```

### Dataset Statistics

| Matrix | Users | Items | Interactions | Density |
|---|---|---|---|---|
| Small matrix | 1,411 | 3,327 | 4,676,570 | **99.6%** |
| Big matrix | 7,176 | 10,728 | 12,530,806 | 16.3% |

### Critical Data Interpretation Notes

1. **Small matrix ≠ small data.** 4.7 million interactions is substantial. At 8 columns × float64, the small matrix CSV is ~300–400 MB uncompressed.
2. **Big matrix ≠ fully observed.** The big matrix has 16.3% density. Unobserved entries are NOT negative signals — they were never shown to users by the recommendation system.
3. **Small matrix IS nearly fully observed (99.6%).** This makes it unique: missing entries are explicitly blocked videos, not unseen content. This distinction is fundamental for training unbiased models.
4. **watch_ratio is the continuous feedback signal.** There is no binary "like" label. If binary labels are needed in future phases, they must be derived with a documented threshold (e.g., `watch_ratio > 2.0`) with justification.
5. **item_daily_features is the largest side-information file.** It has 53 columns with daily aggregated engagement statistics. This file may be large and should be processed with Polars, not loaded into memory naively.
6. **user_features has 18 encrypted `onehot_feat` columns.** These are anonymized demographic/behavioral features with known ranges but unknown semantic meaning.
7. **user_features_raw.csv has rich demographics** (gender, age_range, city, phone, ISP) — more useful for profiling than the anonymized version. This file is NOT in the zip and must be downloaded separately.

### RAM Estimates for Processing

| File | Estimated CSV size | Estimated in-memory (Polars) |
|---|---|---|
| small_matrix.csv | ~350 MB | ~200 MB |
| big_matrix.csv | ~850 MB | ~500 MB |
| item_daily_features.csv | ~200 MB | ~120 MB |
| user_features.csv | ~5 MB | ~3 MB |
| item_categories.csv | ~2 MB | ~1 MB |
| kuairec_caption_category.csv | ~2 MB | ~1 MB |
| **Total peak (all loaded)** | | **~825 MB** |

**Assessment:** Processing all files simultaneously would use ~825 MB RAM. With Python overhead and Polars working memory, peak could reach 2–3 GB. Well within the 16 GB limit, but the pipeline should not load all files simultaneously to stay efficient.

---

## Architecture Decisions Requiring Approval

### D-P2-01 — Download Strategy: Full Zip vs Selective Files

**Options:**
- **Option A (Recommended):** Download `KuaiRec.zip` (412 MB) + the 3 separate files outside the zip. Total ~417 MB. This gets everything in one coherent download.
- **Option B:** Download only specific files needed for Phase 2. Would skip `social_network.csv` and `item_daily_features.csv` initially.

**Recommendation:** Option A. The dataset is small enough to download in full, and having all files available prevents repeated downloads. The zip contains the most complete snapshot.

---

### D-P2-02 — Primary Processing Tool: Polars vs DuckDB vs Both

**Situation:** Both Polars and DuckDB are listed as Phase 2 technologies. They have overlapping capabilities but different strengths.

| Tool | Best For |
|---|---|
| Polars | Row-level transformations, type casting, streaming CSV reads |
| DuckDB | SQL-style aggregations, joins across files, analytical queries |

**Recommendation:** Use **both** — Polars for cleaning and standardization pipelines, DuckDB for cross-file analytical queries and the quality report generation. This establishes both tools in the project without over-engineering.

---

### D-P2-03 — Where Does Data Engineering Code Live?

**Options:**
- **Option A (Recommended):** `scripts/data/` — standalone Python scripts, executable directly with `python scripts/data/inspect_dataset.py`. Not inside the FastAPI app. This keeps the data engineering layer separate from the web API layer.
- **Option B:** `backend/app/data/` — inside the FastAPI app. Unnecessarily couples data engineering to the web server.

**Recommendation:** Option A. Data engineering scripts are offline batch processes, not web request handlers. They belong in `scripts/`, not `app/`. A thin `backend/app/data/` module can be added later if the API needs to serve processed data.

---

### D-P2-04 — Which Files to Process to Parquet (Phase 2 scope)

**All 9 files could be converted.** However, Phase 2 should be disciplined:

**Recommended Phase 2 Parquet targets:**
1. `small_matrix.csv` → `small_matrix.parquet` (the primary training/evaluation dataset)
2. `user_features.csv` → `user_features.parquet`
3. `user_features_raw.csv` → `user_features_raw.parquet`
4. `item_categories.csv` → `item_categories.parquet`
5. `kuairec_caption_category.csv` → `kuairec_caption_category.parquet`

**Deferred to Phase 3:**
- `big_matrix.csv` — large, only needed for full-scale training (Phase 3)
- `item_daily_features.csv` — 53 columns, 200+ MB; needs careful schema design
- `social_network.csv` — list-format data needs special handling; deferred to Phase 3
- `video_raw_categories_multi.csv` — multi-row-per-video format; deferred

---

### D-P2-05 — PostgreSQL Integration Boundary

**Decision:** Do NOT load any KuaiRec data into PostgreSQL in Phase 2.

**Rationale:** 
- The interaction files (small_matrix: 4.7M rows, big_matrix: 12.5M rows) would create large PostgreSQL tables without a clear query pattern yet.
- Phase 2 is analytical/exploratory. Parquet + DuckDB is the right tool for this stage.
- Phase 3 will design the application data model with proper indexing strategy.
- Exception: lightweight metadata (file manifest, quality report summary) can optionally go into PostgreSQL as a future enhancement — but not required in Phase 2.

---

## Proposed Directory Structure

```
data/
├── raw/
│   └── kuairec/
│       └── original_downloads/     # Immutable — never modified after download
│           ├── KuaiRec.zip
│           ├── big_matrix.csv      # Extracted from zip
│           ├── small_matrix.csv    # Extracted from zip
│           ├── social_network.csv  # Extracted from zip
│           ├── user_features.csv   # Extracted from zip
│           ├── item_categories.csv # Extracted from zip
│           ├── item_daily_features.csv # Extracted from zip
│           ├── kuairec_caption_category.csv  # Downloaded separately
│           ├── video_raw_categories_multi.csv # Downloaded separately
│           └── user_features_raw.csv          # Downloaded separately
│
├── interim/
│   └── kuairec/
│       ├── inspected/              # Inspection reports (JSON)
│       └── validated/              # Validation reports (JSON)
│
├── processed/
│   └── kuairec/
│       ├── parquet/                # Final Parquet outputs
│       └── reports/                # Data quality reports (JSON + Markdown)
│
├── manifests/
│   └── dataset_manifest.json       # Download provenance, checksums, dates
│
└── metadata/
    └── schema/                     # Field-level schema documentation (JSON)
```

```
scripts/
└── data/
    ├── __init__.py
    ├── common.py                   # Shared utilities, paths, logging
    ├── download_dataset.py         # WP1: Download with verification
    ├── inspect_dataset.py          # WP2: File inspection and schema discovery
    ├── validate_dataset.py         # WP3: Validation and quality analysis
    ├── clean_dataset.py            # WP4: Cleaning and standardization
    ├── convert_to_parquet.py       # WP5: CSV → Parquet conversion
    └── generate_report.py          # WP6: Data quality report generation
```

```
backend/
└── app/
    └── (unchanged — no data engineering code in the web app layer)

docs/
└── data/                           # New documentation directory
    ├── dataset-overview.md
    ├── data-dictionary.md
    ├── data-acquisition-guide.md
    ├── data-pipeline.md
    └── data-quality-report.md

docs/decisions/
    └── data-engineering-decisions.md  # Appended to existing decisions file
```

```
backend/
└── tests/
    └── data/                       # New test module for data pipeline
        ├── __init__.py
        ├── conftest.py             # Fixtures: tiny synthetic CSV samples
        ├── test_inspect.py
        ├── test_validate.py
        ├── test_clean.py
        └── test_parquet.py
```

---

## New Dependencies (Phase 2)

To be added to `backend/pyproject.toml` under a new `[project.optional-dependencies]` group called `data`:

```toml
[project.optional-dependencies]
data = [
    "polars>=1.0.0",
    "duckdb>=1.0.0",
    "pyarrow>=17.0.0",   # required by Polars for Parquet I/O
    "rich>=13.0.0",      # for readable console output in scripts
    "tqdm>=4.0.0",       # download progress bars
]
```

**Why not add to main dependencies?**
- These are data engineering tools, not web API dependencies.
- The FastAPI app does not need Polars/DuckDB at runtime.
- Keeping them in a separate optional group ensures the production web app stays lightweight.

**Install command:**
```powershell
cd backend
.venv\Scripts\Activate.ps1
pip install -e ".[data]"
```

---

## Work Packages

---

### WP1 — Environment Setup & Dependency Installation

**Status:** `[ ] pending`

**Intent:** Install Phase 2 data engineering dependencies (Polars, DuckDB, PyArrow, rich, tqdm) into the existing virtual environment without breaking Phase 1.

**Expected Outcomes:**
- `polars`, `duckdb`, `pyarrow`, `rich`, `tqdm` importable in `backend/.venv`
- `import polars; print(polars.__version__)` succeeds
- `pytest tests/ -v` still shows 18/18 passing after installation

**Files to Modify:**
- `backend/pyproject.toml` — add `[project.optional-dependencies.data]` section
- `AGENTS.md` — update approved stack table (Polars, DuckDB now active)

**Commands:**
```powershell
cd backend
.venv\Scripts\Activate.ps1
pip install -e ".[data]"
pytest tests/ -v    # Must still pass 18/18
```

**Acceptance Criteria:**
- All 5 new libraries importable
- Zero new test failures
- `ruff check app/` still passes

---

### WP2 — Data Storage Structure & Scripts Foundation

**Status:** `[ ] pending`

**Intent:** Create the directory structure under `data/` and the `scripts/data/` foundation. Set up `common.py` with shared path constants and logging.

**Expected Outcomes:**
- Full `data/` hierarchy created with `.gitkeep` placeholders
- `scripts/data/` directory with `__init__.py` and `common.py`
- `data/manifests/` directory created
- `data/metadata/schema/` directory created
- All new data directories added to `.gitignore`

**Files to Create:**
- `scripts/data/__init__.py`
- `scripts/data/common.py` — `DATA_ROOT`, `RAW_DIR`, `INTERIM_DIR`, `PROCESSED_DIR`, `MANIFEST_PATH` constants + logging setup
- Directory placeholders under `data/`
- `.gitignore` additions for `data/raw/kuairec/`, `data/processed/kuairec/`, etc.

**Acceptance Criteria:**
- `python scripts/data/common.py` runs without errors
- `git status` shows no data files tracked
- Directory structure matches the plan

---

### WP3 — Dataset Download & Verification

**Status:** `[ ] pending`

**Intent:** Download all KuaiRec 2.0 files from Zenodo, verify checksums, extract the zip, and write the dataset manifest.

**Expected Outcomes:**
- `KuaiRec.zip` downloaded to `data/raw/kuairec/original_downloads/`
- MD5 checksum verified against `261550d472c48eff4990fb13c0e5bcf7`
- Zip extracted: 6 core CSV files present
- 3 additional files downloaded separately (caption_category, video_raw_categories_multi, user_features_raw)
- `data/manifests/dataset_manifest.json` written with: source URL, download date/time, file sizes, MD5 checksums, file paths
- Download script is re-runnable (skips already-downloaded files)

**Files to Create:**
- `scripts/data/download_dataset.py`
- `data/manifests/dataset_manifest.json` (generated)

**Download URLs (verified):**
```
KuaiRec.zip:                    https://zenodo.org/records/18164998/files/KuaiRec.zip
kuairec_caption_category.csv:   https://zenodo.org/records/18164998/files/kuairec_caption_category.csv
video_raw_categories_multi.csv: https://zenodo.org/records/18164998/files/video_raw_categories_multi.csv
user_features_raw.csv:          https://zenodo.org/records/18164998/files/user_features_raw.csv
```

**Expected checksums:**
```
KuaiRec.zip:                    md5:261550d472c48eff4990fb13c0e5bcf7
kuairec_caption_category.csv:   md5:31bc38cdccdf75a71df137779035f8cb
video_raw_categories_multi.csv: md5:d05eea147135d2cdf7759fba5c0d70d4
user_features_raw.csv:          md5:3969b8120035e7ced36d56926a7cbd24
```

**Commands:**
```powershell
cd "C:\Users\Edunet Foundation\Desktop\IBM Bob Projects\Recommendation System"
python scripts/data/download_dataset.py
```

**Risk:** Network interruption during 412 MB zip download.
**Mitigation:** Stream download in chunks with progress bar; support resume on retry.

**Acceptance Criteria:**
- All 9 CSV files present in `data/raw/kuairec/original_downloads/`
- All checksums verified
- Manifest JSON written and parseable
- No data files appear in `git status`

---

### WP4 — Dataset Inspection & Schema Discovery

**Status:** `[ ] pending`

**Intent:** Inspect each raw CSV file using Polars with lazy evaluation (not loading full files into memory). Record schema, row counts, column counts, data types, and sample values.

**Expected Outcomes:**
- Inspection report JSON written to `data/interim/kuairec/inspected/`
- Actual row/column counts confirmed (not just documentation claims)
- Data types verified against documentation
- Null counts per column per file
- Sample of first/last rows without loading entire file
- Identified any discrepancies between documentation and actual data

**Files to Create:**
- `scripts/data/inspect_dataset.py`
- `data/interim/kuairec/inspected/inspection_report.json` (generated)

**Key inspection questions to answer:**
1. Do row counts match the documented statistics (1,411 users × 3,327 items for small matrix)?
2. What are the actual null rates in `watch_ratio`?
3. What is the range of `watch_ratio` values? (Expected: mostly 0–5, but outliers exist)
4. Is `item_categories.csv` truly in list format (e.g., `"[27,9]"`) or is it already parsed?
5. What encoding is used for Chinese text fields?
6. What is the actual date range of interactions?

**Commands:**
```powershell
python scripts/data/inspect_dataset.py
```

**Acceptance Criteria:**
- `inspection_report.json` exists and is valid JSON
- All 9 files inspected
- No file loaded fully into RAM at once (use Polars lazy API)
- Schema discrepancies flagged clearly

---

### WP5 — Validation & Data Quality Analysis

**Status:** `[ ] pending`

**Intent:** Validate referential integrity across files, detect duplicates, check required columns, verify numerical ranges, and document all findings.

**Expected Outcomes:**
- Validation report written to `data/interim/kuairec/validated/`
- Duplicate interaction detection in both matrices
- Referential integrity: every `video_id` in matrices exists in `item_categories.csv`
- Referential integrity: every `user_id` in matrices exists in `user_features.csv`
- Watch ratio outlier analysis (> 5.0 or < 0.0)
- Zero-duration play detection
- Categorization of issues: fixable, preserve, investigate

**Files to Create:**
- `scripts/data/validate_dataset.py`
- `data/interim/kuairec/validated/validation_report.json` (generated)

**Key validation checks:**
1. No duplicate `(user_id, video_id)` pairs within each matrix
2. All `user_id` values in matrices have entries in `user_features.csv`
3. All `video_id` values in matrices have entries in `item_categories.csv`
4. `watch_ratio` is non-negative
5. `play_duration` ≤ reasonable upper bound (not 100x `video_duration`)
6. `timestamp` is within expected date range (2020)
7. `video_id = 1225` does not appear (was removed in v2.0)

**Acceptance Criteria:**
- Validation report exists and is machine-readable
- Every check is explicitly pass/fail/warning with counts
- No silent failures

---

### WP6 — Cleaning & Standardization

**Status:** `[ ] pending`

**Intent:** Apply documented cleaning rules to produce standardized versions of the datasets. Every transformation is logged with before/after counts.

**Expected Outcomes:**
- Cleaned files written to `data/interim/kuairec/validated/` (not overwriting raw)
- Cleaning log embedded in report
- Column names standardized to `snake_case`
- Data types cast to appropriate Polars types
- `time` field parsed to proper datetime
- `feat` field in `item_categories.csv` parsed from string `"[27,9]"` to list of integers
- `friend_list` in `social_network.csv` similarly parsed
- Chinese text fields preserved as UTF-8 strings

**Cleaning Rules (documented, not silent):**

| Rule | Why | Affects |
|---|---|---|
| Cast `timestamp` to Float64 | Consistency | Both matrices |
| Parse `time` as Datetime | Enable time-based filtering | Both matrices |
| Parse `feat` list strings | Enable categorical use | item_categories |
| Parse `friend_list` strings | Enable graph use | social_network |
| Cast `int64` columns to Int32 where range allows | Memory efficiency | user_features |
| Standardize boolean-like int columns (0/1) | Clarity | user_features |

**What is NOT cleaned:**
- `watch_ratio > 1.0` values are **preserved** — a user watching a video more than once results in ratios > 1.0 and this is legitimate behavior
- Null values are documented but not imputed in Phase 2 — imputation strategy belongs to Phase 3 feature engineering
- Chinese text is preserved as-is — no transliteration

**Files to Create:**
- `scripts/data/clean_dataset.py`

**Acceptance Criteria:**
- Cleaning log documents every row affected
- Zero rows silently dropped
- Raw files in `data/raw/` are unchanged

---

### WP7 — Parquet Conversion

**Status:** `[ ] pending`

**Intent:** Convert the Phase 2 scope files to Parquet using Polars, with appropriate compression and schema metadata.

**Expected Outputs (5 files):**

| Source CSV | Output Parquet | Compression |
|---|---|---|
| `small_matrix.csv` | `small_matrix.parquet` | zstd |
| `user_features.csv` | `user_features.parquet` | zstd |
| `user_features_raw.csv` | `user_features_raw.parquet` | zstd |
| `item_categories.csv` | `item_categories.parquet` | zstd |
| `kuairec_caption_category.csv` | `kuairec_caption_category.parquet` | zstd |

**Why zstd?** Better compression ratio than snappy, still fast to decompress, supported by Polars and DuckDB.

**Files to Create:**
- `scripts/data/convert_to_parquet.py`
- 5 `.parquet` files in `data/processed/kuairec/parquet/`

**Acceptance Criteria:**
- Each Parquet file readable with `polars.read_parquet()`
- Row counts match source CSVs
- Schema preserved (column names, types)
- No data files committed to Git

---

### WP8 — Data Quality Report Generation

**Status:** `[ ] pending`

**Intent:** Produce a machine-readable JSON report and human-readable Markdown summary capturing all findings from inspection, validation, and cleaning.

**Report Contents:**
- Dataset name, version, source, DOI, license
- Download date and file checksums
- Per-file statistics (rows, columns, nulls, duplicates)
- Validation results summary
- Cleaning operations log
- Parquet output summary
- Issues discovered / corrected / preserved / requiring investigation
- Processing timestamps

**Files to Create:**
- `scripts/data/generate_report.py`
- `data/processed/kuairec/reports/data_quality_report.json` (generated)
- `data/processed/kuairec/reports/data_quality_report.md` (generated)
- `docs/data/data-quality-report.md` (copy of Markdown report, committed to Git)

**Acceptance Criteria:**
- Report JSON is valid and schema-consistent
- Markdown report is human-readable without tooling
- All documented issues have a status (corrected / preserved / investigate)

---

### WP9 — Tests for Data Pipeline

**Status:** `[ ] pending`

**Intent:** Write Pytest tests for all pipeline stages using small synthetic fixtures — no real dataset files required.

**Test Strategy:**
- Create `backend/tests/data/fixtures/` with small synthetic CSVs matching KuaiRec schema
- Tests import pipeline functions from `scripts/data/` and run against fixtures
- Tests do NOT download or require real dataset files

**Test Coverage:**
| Test File | What It Tests |
|---|---|
| `test_inspect.py` | Schema discovery, type detection, null counting on fixture |
| `test_validate.py` | Duplicate detection, referential integrity on fixture |
| `test_clean.py` | List parsing, type casting, cleaning log format |
| `test_parquet.py` | Round-trip CSV → Parquet → read, row count match, schema match |

**Acceptance Criteria:**
- All new tests pass
- Original 18 tests still pass (run full `pytest tests/` from `backend/`)
- No test requires internet access or real dataset files

---

### WP10 — Documentation

**Status:** `[ ] pending`

**Intent:** Write all 5 data documentation files and update AGENTS.md.

**Files to Create:**
- `docs/data/dataset-overview.md` — KuaiRec 2.0 overview, statistics, interpretation notes
- `docs/data/data-dictionary.md` — Field-level definitions for all 9 files
- `docs/data/data-acquisition-guide.md` — Step-by-step download guide for Windows
- `docs/data/data-pipeline.md` — Pipeline architecture, stages, commands
- `docs/data/data-quality-report.md` — Template updated after WP8 generates the actual report
- `docs/decisions/data-engineering-decisions.md` — ADRs D-P2-01 through D-P2-05

**Files to Modify:**
- `AGENTS.md` — Update phase description, add Polars/DuckDB as active technologies, note Python 3.14 correction
- `README.md` — Add Phase 2 status, data setup section

**Acceptance Criteria:**
- All 6 doc files written
- AGENTS.md updated
- No documentation claims things that haven't been verified

---

## Dependency Order

```
WP1: Install dependencies
  ↓
WP2: Create directory structure + scripts foundation
  ↓
WP3: Download dataset + verify checksums
  ↓
WP4: Inspect files (lazy, no full memory loads)
  ↓
WP5: Validate integrity
  ↓
WP6: Clean and standardize
  ↓
WP7: Convert to Parquet
  ↓
WP8: Generate quality report
  ↓
WP9: Tests (can partially begin after WP6, needs WP7 for Parquet tests)
  ↓
WP10: Documentation (can begin after WP4 for dictionary, finalized after WP8)
```

---

## Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Zenodo download interrupted (412 MB) | Medium | Stream in 8 KB chunks with resume support; verify checksum after |
| `item_daily_features.csv` larger than estimated | Low | Only in Phase 3 scope; not loaded in Phase 2 |
| `watch_ratio` has unexpected large outliers | Medium | Document in validation report; do not silently drop |
| Chinese text encoding issues on Windows | Medium | Explicitly specify UTF-8 when reading CSVs with Polars |
| Polars/DuckDB incompatibility with Python 3.14 | Low | Will verify immediately in WP1; fallback to latest versions if needed |
| `item_categories.csv` list format causes Parquet schema issues | Medium | Parse list columns to native Polars List type before writing |
| `video_id = 1225` present (should be removed in v2.0) | Low | Explicitly validate in WP5; flag if found |
| Phase 1 tests break after new dependency install | Low | Run `pytest tests/ -v` immediately after WP1 install |

---

## Phase 2 Acceptance Criteria

| # | Criterion | Verified By |
|---|---|---|
| 1 | KuaiRec 2.0 source verified (Zenodo DOI documented) | Manifest JSON |
| 2 | License documented (CC BY 4.0) | docs/data/dataset-overview.md |
| 3 | All 9 raw files downloaded and checksums verified | Manifest JSON |
| 4 | Raw files immutable (never modified after download) | No write operations on `data/raw/` |
| 5 | Dataset schemas documented | docs/data/data-dictionary.md |
| 6 | Validation implemented and report generated | validation_report.json |
| 7 | Data quality report generated (JSON + Markdown) | data_quality_report.json |
| 8 | Cleaning reproducible from raw files | `python scripts/data/clean_dataset.py` |
| 9 | 5 Parquet files in `data/processed/kuairec/parquet/` | File existence + schema check |
| 10 | Dataset manifest with provenance recorded | dataset_manifest.json |
| 11 | Pipeline tests pass (new + existing 18) | `pytest tests/ -v` |
| 12 | Phase 1 tests unbroken | 18/18 still passing |
| 13 | Documentation complete (5 data docs) | File review |
| 14 | No data files in Git | `git status` clean of data files |
| 15 | Pipeline reproducible from setup guide | Walkthrough from clean `data/` state |
