# KuaiRec 2.0 — Dataset Overview

## Purpose

KuaiRec 2.0 is the primary interaction dataset for the Hybrid Recommendation Intelligence
Platform. It provides real-world user–video interaction data from the Kuaishou short-video
platform and is used to train and evaluate recommendation algorithms in Phase 3 and beyond.

---

## Dataset Identity

| Property | Value |
|---|---|
| Name | KuaiRec 2.0 |
| Source | Kuaishou Technology (short-video platform) |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| DOI | `10.5281/zenodo.18164998` |
| Version | 2.0 (video_id=1225 removed vs v1.0) |
| Acquisition phase | Phase 2 |

---

## Files

### Phase 2 Scope (fully processed)

| File | Rows | Columns | Parquet size | Description |
|---|---|---|---|---|
| `small_matrix.csv` | 4,676,570 | 8 | 101 MB | Dense user–video interactions (99.6% density) |
| `user_features.csv` | 7,176 | 31 | 106 KB | User demographic and engagement features |
| `user_features_raw.csv` | 7,176 | — | 147 KB | Raw user features with original encoding |
| `item_categories.csv` | 3,897 | 2 | 40 KB | Video category labels (integer lists) |
| `kuairec_caption_category.csv` | 3,897 | 10 | 638 KB | Video captions and three-level category hierarchy |

### Phase 3 Scope (deferred — not yet processed)

| File | Size | Reason deferred |
|---|---|---|
| `big_matrix.csv` | 1,033 MB | Requires streaming processing; too large for in-memory approach |
| `item_daily_features.csv` | — | Daily aggregates; not needed until model training |
| `social_network.csv` | — | Graph features; deferred to social-aware filtering phase |
| `video_raw_categories_multi.csv` | — | Redundant with `item_categories`; deferred |

---

## small_matrix Key Properties

The `small_matrix` is the primary interaction file used in Phase 2 and Phase 3.

| Property | Value |
|---|---|
| Total interactions | 4,676,570 |
| Unique users | 1,411 |
| Unique videos | 3,327 |
| Interaction density | 99.6% — essentially fully dense |
| Null timestamps | 181,992 rows (3.89%) — expected, source data gap |
| watch_ratio > 1.0 | 1,514,717 rows — valid replays, not data errors |
| watch_ratio max | 571.52 — extreme outlier, documented for Phase 3 investigation |
| watch_ratio nulls | 0 — always present |

**Important:** `watch_ratio > 1.0` is intentional and valid in this dataset. A ratio of 1.5
means the user watched the video 1.5× its full duration (i.e., replayed part of it). Do not
cap or drop these values without Phase 3 analysis.

---

## Documented Anomalies

| Issue | Severity | Decision |
|---|---|---|
| 3.89% null timestamps in small_matrix | Warning | Preserved — source data gap, not a cleaning error |
| 12.63% null captions in kuairec_caption_category | Warning | Preserved — imputation deferred to Phase 3 |
| video_id=1225 present in item_categories (1 row) | Fail | Documented — not present in interaction matrices; dataset is v2.0 |
| watch_ratio max = 571.52 | Warning | Preserved — extreme replay; flagged for Phase 3 investigation |

---

## Storage Location

All dataset files are excluded from Git. See `.gitignore`.

```
data/
├── raw/kuairec/
│   └── original_downloads/        ← 9 original CSV files + KuaiRec.zip
├── interim/kuairec/
│   ├── inspected/inspection_report.json
│   ├── validated/validation_report.json
│   └── intermediate/cleaning_log.json
├── processed/kuairec/
│   ├── parquet/                   ← 5 Parquet files (zstd compression)
│   │   ├── small_matrix.parquet
│   │   ├── user_features.parquet
│   │   ├── user_features_raw.parquet
│   │   ├── item_categories.parquet
│   │   └── kuairec_caption_category.parquet
│   └── reports/
│       ├── data_quality_report.json
│       └── data_quality_report.md
└── manifests/dataset_manifest.json
```

---

## Re-acquiring the Dataset

If the data directory is lost, re-run:

```powershell
# From project root, with backend venv activated
backend\.venv\Scripts\python.exe scripts/data/download_dataset.py
```

This downloads KuaiRec 2.0 from Zenodo, verifies checksums, and extracts all files.
See [`docs/data/data-acquisition-guide.md`](data-acquisition-guide.md) for full instructions.

---

## References

- Original paper: Gao et al., "KuaiRec: A Fully-Observed Dataset and Insights for Evaluating
  Recommender Systems", CIKM 2022.
- Zenodo record: https://zenodo.org/records/18164998
- License: https://creativecommons.org/licenses/by/4.0/
