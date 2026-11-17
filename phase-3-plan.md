# Phase 3 — Baseline Recommendation Systems
## Technical Implementation Blueprint

**Status:** Awaiting approval  
**Preceded by:** Phase 2 — Data Engineering ✅ Complete  
**Followed by:** Phase 4 — Content-Based and Collaborative Filtering (redefined)

---

## Phase Redefinition

Per owner decision (2026-11-17), Phase 3 is redefined from the original "Recommendation Engine
(content-based, collaborative, hybrid)" to:

> **Phase 3 — Baseline Recommendation Systems**
>
> Deliverables: EDA on KuaiRec 2.0, target label definition, train/val/test split,
> evaluation framework, and five non-ML baseline recommenders with measurable benchmarks.

Content-based filtering and collaborative filtering move to Phase 4.
ML ranking (LightGBM/XGBoost) moves to Phase 5.

---

## 0. Verified Ground Truth (From Phase 2)

### Available Parquet Files

| File | Rows | Columns | Size | Key columns |
|---|---|---|---|---|
| `small_matrix.parquet` | 4,676,570 | 8 | 101 MB | user_id, video_id, play_duration, video_duration, time (Datetime), date (Int64), timestamp (Float64), watch_ratio (Float64) |
| `user_features.parquet` | 7,176 | 31 | 106 KB | user_id, user_active_degree, is_* (Boolean), follow/fan/friend counts, register_days, onehot_feat0–17 |
| `user_features_raw.parquet` | 7,176 | 37 | 147 KB | user_id + demographic details: gender, age_range, phone_brand, city, platform, app installs |
| `item_categories.parquet` | 10,728 | 3 | 40 KB | video_id, feat (String), feat_parsed (List[Int64]) |
| `kuairec_caption_category.parquet` | 10,728 | 10 | 638 KB | video_id, caption (12.63% null), first/second/third_level_category_name, topic_tag |

### Interaction Statistics

- **Total interactions:** 4,676,570
- **Unique users:** 1,411 (all in user_features, dense coverage)
- **Unique items:** 3,327 (subset of 10,728 items with features)
- **Matrix density:** 99.6% — nearly fully observed
- **watch_ratio:** continuous [0.0, 571.52], never null; 32.4% > 1.0 (replays)
- **Null timestamps:** 3.89% of rows (181,992) — preserved

### Critical Property: Near-Full Density

The small_matrix has 99.6% density (1,411 × 3,327 = 4,694,397 possible pairs; 4,676,570
observed). This means:
- Standard leave-N-out or random negative sampling does not apply
- Explicit ratings exist for almost all user-item pairs
- "Unobserved" does NOT mean "disliked" — most pairs simply haven't happened yet
- Evaluation must use temporal hold-out to simulate realistic conditions

### Deferred Data (Phase 4+)

- `big_matrix.csv`: 12.5M interactions, 7,176 × 10,728, 16.3% density — realistic sparsity
- `item_daily_features.csv`: 53-column daily feature aggregates
- `social_network.csv`: user–user follow graph

---

## 1. Scope and Non-Scope

### In Scope (Phase 3)

1. EDA on small_matrix, user_features, item_categories, kuairec_caption_category
2. Target label definition and justification
3. Temporal train/validation/test split
4. `ml/` directory scaffold
5. Evaluation framework (Precision@K, Recall@K, NDCG@K, MAP@K, Coverage, Diversity, Novelty)
6. Five baseline recommenders:
   - Random recommendation
   - Global popularity (by interaction count or mean watch_ratio)
   - Category-based popularity
   - Recent/trending popularity (requires temporal data — 96.11% of rows have timestamps)
   - User-preference heuristic (category affinity from interaction history)
7. Experiment result recording (JSON, no MLflow yet)
8. Full test suite for all algorithms and metrics
9. Phase 3 documentation

### Out of Scope (Phase 3)

- Content-based filtering (TF-IDF, embeddings) → Phase 4
- Collaborative filtering (matrix factorization, ALS, NCF) → Phase 4
- Hybrid combining logic → Phase 4
- LightGBM / XGBoost ranking → Phase 5
- big_matrix processing → Phase 4
- FastAPI recommendation endpoint → Phase 4
- MLflow experiment tracking → Phase 4

---

## 2. Architecture Design

### Directory Structure

```
ml/
├── __init__.py
├── config.py                  # Experiment configuration dataclass
├── data/
│   ├── __init__.py
│   ├── loader.py              # Parquet loading + lazy access
│   └── splits.py              # Temporal train/val/test split logic
├── features/
│   ├── __init__.py
│   ├── user_features.py       # User feature extraction / aggregation
│   └── item_features.py       # Item category feature extraction
├── baselines/
│   ├── __init__.py
│   ├── base.py                # Abstract recommender interface
│   ├── random_recommender.py
│   ├── popularity_recommender.py
│   ├── category_popularity_recommender.py
│   ├── trending_recommender.py
│   └── user_preference_recommender.py
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py             # Precision@K, Recall@K, NDCG@K, MAP@K, Coverage, Diversity, Novelty
│   └── evaluator.py           # Evaluation runner
└── experiments/
    ├── __init__.py
    └── recorder.py            # JSON experiment recorder

scripts/
└── ml/                        # New subdirectory for Phase 3+
    ├── __init__.py
    ├── run_eda.py             # EDA script
    ├── prepare_splits.py      # Create and save train/val/test splits
    └── run_baselines.py       # Run all baselines + evaluation

data/
└── processed/
    └── kuairec/
        └── splits/            # Created by prepare_splits.py (not committed to Git)
            ├── train.parquet
            ├── val.parquet
            └── test.parquet

experiments/                   # New top-level directory
    └── results/               # JSON experiment result files (not committed to Git)
        └── .gitkeep

docs/
└── phases/
    └── phase-3-baseline-guide.md
docs/
└── ml/                        # New docs directory
    ├── evaluation-methodology.md
    ├── baseline-algorithms.md
    └── experiment-guide.md

backend/tests/
└── ml/                        # New test directory
    ├── __init__.py
    ├── conftest.py
    ├── test_splits.py
    ├── test_metrics.py
    ├── test_random_recommender.py
    ├── test_popularity_recommender.py
    ├── test_category_recommender.py
    ├── test_trending_recommender.py
    └── test_user_preference_recommender.py
```

### Design Principles

1. **`ml/` is standalone** — no FastAPI imports, no database imports, no backend dependencies
2. **All recommenders implement one interface** — `BaseRecommender` with `fit(train_df)` and `recommend(user_id, n, exclude_seen)` methods
3. **Polars throughout** — no Pandas; lazy scanning for large frames
4. **No dense matrices** — do not build a full user × item matrix in memory
5. **Deterministic** — all recommenders accept a `random_seed` parameter
6. **Scripts are runners** — `scripts/ml/*.py` orchestrate `ml/` modules; they do not contain business logic

---

## 3. Target Label Definition

### Decision

**Use `watch_ratio` as the relevance signal for ranking evaluation.**

Rationale:
- It is the only explicit engagement signal in small_matrix
- It is never null — no imputation required
- It captures engagement quality beyond binary click (0.0 = no play, > 1.0 = replay)
- Continuous signal supports both ranking and binary threshold experiments

### Binary Threshold (for ranking metrics)

For Precision@K, Recall@K, NDCG@K, and MAP@K — which require a binary relevant/not-relevant
label — use:

> **relevant = watch_ratio ≥ 0.5**

Rationale:
- Watching at least half the video is a meaningful engagement signal
- Avoids treating brief accidental plays as positive
- Excludes the 30,312 zero-ratio rows as confirmed non-engagements

**This threshold is a Phase 3 decision and is recorded as an ADR. It can be revisited in Phase 4.**

### Zero-ratio rows

30,312 rows have watch_ratio = 0.0. These indicate the video was initiated but not played.
They are **preserved in training data** (they are still observed interactions) but
**classified as non-relevant** under the binary threshold.

### Unobserved pairs

With 99.6% density, only 0.4% of user-item pairs are unobserved. These are treated as
unknown, NOT as confirmed negatives. Evaluation only uses observed test interactions.

---

## 4. Train / Validation / Test Split

### Strategy: Temporal Hold-Out

**Method:** Sort interactions by timestamp (or `date` where timestamp is null).
- Train: interactions before cutoff_1
- Validation: interactions between cutoff_1 and cutoff_2
- Test: interactions after cutoff_2

**Rationale:**
- Simulates real deployment — the system only knows past interactions at serving time
- Prevents data leakage from future interactions influencing popularity statistics
- Appropriate for sequential recommendation scenarios

**Null timestamp handling:** 181,992 rows (3.89%) have null timestamp/time/date. These are
assigned to the **training split** (conservative assumption: they are older interactions).
This is documented as an assumption and recorded in the split metadata.

**Split proportions (approximate by time):**
- Train: ~70% of the time range
- Validation: ~15%
- Test: ~15%

**Minimum interactions per user in test:** Users with < 1 relevant test interaction are
**excluded from evaluation** (they cannot contribute to recall/precision calculations),
but they may still receive recommendations.

**Output:** Three Parquet files saved to `data/processed/kuairec/splits/` (excluded from Git).
A JSON split manifest records exact cutoff timestamps and row counts.

### Leakage Prevention

- Popularity statistics are computed **only on training data**
- Category affinity is computed **only on training data**
- User history used for seen-item filtering comes **only from training data** at evaluation time
- Validation split is used for K-selection only; test split is used for final reporting

---

## 5. Baseline Algorithm Specifications

### 5.1 Random Recommender

**Logic:** For each user, recommend N items uniformly at random from the item catalog,
excluding items already seen by that user.

**Input:** Item catalog (list of video_ids)

**Features required:** None

**Parameters:** `random_seed: int`, `n_recommendations: int`

**Cold-start:** Handles new users naturally — no history needed

**Complexity:** O(N) per user (item catalog sample)

**Expected performance:** Near-zero precision; sets absolute lower bound

**Purpose:** Verifies evaluation framework is working; establishes absolute floor

---

### 5.2 Global Popularity Recommender

**Logic:** Rank all items by a global popularity score computed on the training set.
Recommend top-N globally popular items to every user, excluding seen items.

**Popularity signal options** (choose one — decision recorded in ADR):
- Option A: **Interaction count** — number of times each item appears in training data
- Option B: **Weighted score** — mean(watch_ratio) × log(1 + interaction_count)

**Recommended:** Option B (weighted score) — accounts for both reach and engagement quality.
Items with many interactions but low watch_ratio (skipped) rank lower than items with
moderate reach and high engagement.

**Input:** `small_matrix` training split

**Features required:** `video_id`, `watch_ratio`, `user_id` (for count)

**Cold-start:** Handles new users — no history needed

**Complexity:** O(N_interactions) to compute scores; O(K) to retrieve top-K per user

**Memory:** ~40 MB for item score table (3,327 items)

**Leakage risk:** Popularity must be computed on **training split only**

---

### 5.3 Category-Based Popularity Recommender

**Logic:** For each item category (first_level_category_name), compute a popularity ranking
of items within that category using training data. For each user, determine the user's
top-k preferred categories from their training history, then recommend the most popular
items from those categories that the user has not seen.

**Input:** `small_matrix` (train), `item_categories`, `kuairec_caption_category`

**Features required:** `video_id`, `first_level_category_name` (or `feat_parsed`),
`user_id`, `watch_ratio`

**Cold-start:** New user → fall back to global popularity; new item → no category history

**Complexity:** O(N_interactions) for training; O(K × C) per user at serving (C = number of categories)

**Memory:** ~1 MB for per-category popularity tables

---

### 5.4 Recent/Trending Popularity Recommender

**Logic:** Compute popularity scores using only recent training interactions (within a
configurable recency window, e.g., last 7 or 14 days of training data). Items trending
recently outrank items that were popular earlier.

**Availability check:** 96.11% of rows have valid timestamps. The recency window is
computed using the `time` or `timestamp` column. The 3.89% null-timestamp rows
are excluded from recency scoring (they are still included in the training split but
treated as historical, not recent).

**Input:** `small_matrix` training split (timestamp-filtered subset)

**Parameters:** `recency_window_days: int` (default: 14)

**Cold-start:** New user → fall back to trending list (no history needed)

**Complexity:** O(N_recent) where N_recent << N_interactions

**Leakage risk:** "Recent" means recent in **training** — no test timestamps used

---

### 5.5 User-Preference Heuristic Recommender

**Logic:** Use each user's category interaction history from the training set to build
a soft category preference vector. Recommend items from categories the user has
historically engaged with most (highest mean watch_ratio in that category), ranked
by global popularity within those preferred categories.

This is a heuristic, not a learned model — it uses hand-crafted aggregation rules,
not any ML training.

**Input:** `small_matrix` (train), `item_categories`, `kuairec_caption_category`

**Features required:** `user_id`, `video_id`, `watch_ratio`, `first_level_category_name`

**Cold-start:** New user with no history → fall back to global popularity

**Complexity:** O(N_user_interactions) per user to build preference vector at training time;
pre-computed and cached after `fit()`

**Memory:** ~100 KB for per-user category preference vectors (1,411 users × ~20 categories)

---

## 6. Evaluation Framework

### Metrics

For a given user u, ground-truth relevant items G(u) = items with watch_ratio ≥ 0.5 in test,
and recommended items R(u, K) = top-K recommendations excluding seen items:

| Metric | Formula | What it measures |
|---|---|---|
| Precision@K | \|R(u,K) ∩ G(u)\| / K | Fraction of recommended items that are relevant |
| Recall@K | \|R(u,K) ∩ G(u)\| / \|G(u)\| | Fraction of relevant items retrieved |
| NDCG@K | DCG@K / IDCG@K | Ranking quality — rewards relevant items ranked higher |
| MAP@K | Mean of AP@K across users | Average ranking quality over all users |
| Catalog Coverage | \|{items recommended to any user}\| / \|catalog\| | Breadth of recommendations |
| Intra-List Diversity | Avg pairwise category distance within R(u,K) | Category variety per recommendation list |
| Novelty | Avg(-log2(p(item))) where p = item popularity | How non-obvious the recommendations are |

### K Values

Evaluate at K ∈ {5, 10, 20}.

Rationale:
- K=5: short recommendation lists (strict)
- K=10: standard benchmark for recommendation literature
- K=20: longer lists, appropriate given 3,327 available items

### Evaluation Protocol

1. `fit(train_df)` — build all indexes/statistics on training split only
2. For each user in the test set who has ≥ 1 relevant test interaction:
   a. `exclude_seen` = items seen in train (not test)
   b. `recommend(user_id, n=K_max, exclude_seen=exclude_seen)` → top-20 candidates
   c. Compute Precision@K, Recall@K, NDCG@K, MAP@K for K ∈ {5, 10, 20}
3. Compute Catalog Coverage across all users for K=10
4. Compute Diversity and Novelty per user, then average
5. Report per-user distributions (mean, median, std, 10th/90th percentile) for each metric

### Handling Edge Cases

| Case | Handling |
|---|---|
| User has no relevant test interactions | Excluded from precision/recall/NDCG/MAP; included in coverage |
| Item not in training catalog | Cannot be recommended; excluded from seen-item filtering |
| User seen all catalog items in training | Recommend from global popularity; R(u,K) may be empty → metrics = 0 |
| New user (not in training) | Degraded to global popularity or random |
| watch_ratio = 0.0 (non-relevant) | Classified as non-relevant under binary threshold; still counts as "seen" |

---

## 7. Experiment Recording

### Format: JSON (no MLflow in Phase 3)

MLflow is approved but deferred to Phase 4 (requires scikit-learn ecosystem integration).
In Phase 3, each run produces a JSON file:

```json
{
  "experiment_id": "uuid",
  "timestamp": "ISO-8601",
  "phase": "3",
  "algorithm": "GlobalPopularityRecommender",
  "config": {
    "popularity_metric": "weighted_score",
    "k_values": [5, 10, 20],
    "relevance_threshold": 0.5,
    "random_seed": 42
  },
  "dataset": {
    "name": "KuaiRec 2.0 small_matrix",
    "split": "temporal",
    "train_rows": 0,
    "val_rows": 0,
    "test_rows": 0,
    "split_manifest_path": "data/processed/kuairec/splits/split_manifest.json"
  },
  "metrics": {
    "precision_at_5": 0.0,
    "recall_at_5": 0.0,
    "ndcg_at_5": 0.0,
    "precision_at_10": 0.0,
    "recall_at_10": 0.0,
    "ndcg_at_10": 0.0,
    "precision_at_20": 0.0,
    "recall_at_20": 0.0,
    "ndcg_at_20": 0.0,
    "map_at_10": 0.0,
    "catalog_coverage_at_10": 0.0,
    "diversity_at_10": 0.0,
    "novelty_at_10": 0.0
  },
  "runtime_seconds": 0.0,
  "peak_memory_mb": 0.0
}
```

Stored in `experiments/results/` (top-level directory, excluded from Git via `.gitignore`).
A `.gitkeep` file ensures the directory structure is committed.

---

## 8. Dependency Decisions

### Packages to Add

| Package | Version constraint | Justification |
|---|---|---|
| `numpy` | `>=1.26.0` | Required for NDCG/DCG computation; approved for Phase 3+ |
| `scipy` | `>=1.13.0` | Optional — may be needed for statistical tests in evaluation report |
| `scikit-learn` | None in Phase 3 | NOT added — no ML models in Phase 3 baselines |

**NumPy is the only new dependency for Phase 3.** It is added to a new `[ml]` optional
dependency group in `backend/pyproject.toml`.

**SciPy is NOT added yet** — it can be added in Phase 4 when needed.

**MLflow is NOT added yet** — deferred to Phase 4.

### Packages Already Available (Reused)

- `polars` — all data loading, filtering, aggregation
- `duckdb` — EDA analytics, cross-file joins
- `pyarrow` — Parquet I/O

---

## 9. Hardware and Performance Constraints

### Development Machine

- AMD Ryzen 7 7735HS, 16 GB RAM (~13.3 GB usable), CPU-only

### Memory Budget

| Component | Estimated Memory |
|---|---|
| small_matrix training split (~70%) | ~70 MB in Polars |
| Item popularity table | ~1 MB |
| Per-user category preferences (1,411 users) | ~100 KB |
| Evaluation result buffers | ~10 MB |
| **Total peak** | **~90–100 MB** |

This is well within the 13.3 GB usable budget. No chunking or streaming required for
small_matrix baseline evaluation.

### Runtime Estimates

| Step | Estimated Time |
|---|---|
| Load training split from Parquet | < 2 seconds |
| Compute global popularity | < 1 second |
| Compute per-category popularity | < 1 second |
| Compute user-preference vectors | < 2 seconds |
| Evaluate 1 algorithm at K=10 | < 10 seconds |
| Full 5-algorithm evaluation suite | < 2 minutes |

---

## 10. Testing Strategy

All tests use **synthetic datasets** — no real KuaiRec data in tests.

### Test Fixtures

Synthetic fixtures in `backend/tests/ml/conftest.py`:
- 5-user × 10-item interaction DataFrame with known watch_ratios
- Pre-defined train/test split with known relevant items
- Item catalog with known categories
- User preference vectors with known category affinities

### Test Coverage

| Test File | What it tests |
|---|---|
| `test_splits.py` | Temporal split correctness, cutoff timestamps, null handling, no data leakage |
| `test_metrics.py` | Precision@K, Recall@K, NDCG@K, MAP@K correctness on known inputs; edge cases (empty ground truth, full recall) |
| `test_random_recommender.py` | Output is N items, no duplicates, no seen items, deterministic with seed |
| `test_popularity_recommender.py` | Rankings match expected order, seen-item exclusion, cold-start behavior |
| `test_category_recommender.py` | Category preference correctly computed, fallback to global for cold users |
| `test_trending_recommender.py` | Recency window correctly filters, null timestamps handled |
| `test_user_preference_recommender.py` | Category affinity matches training history, cold-start fallback |

### Existing Test Preservation

All 62 existing tests (Phase 1 health/system + Phase 2 data pipeline) must continue to pass.

---

## 11. Work Packages

---

### WP1 — EDA on KuaiRec 2.0

**Objective:** Produce a thorough exploratory data analysis on the interaction and feature
data to surface insights needed for target label selection, split strategy, and baseline design.

**Technical Tasks:**
1. Create `scripts/ml/` directory and `scripts/ml/__init__.py`
2. Write `scripts/ml/run_eda.py` using Polars + DuckDB
3. Analyze: watch_ratio distribution, user activity distribution, item popularity distribution,
   category distribution, temporal distribution of interactions, interaction count per user,
   interaction count per item, user-item overlap between feature files and interaction matrix
4. Write findings to `data/processed/kuairec/reports/eda_report.json` and `eda_report.md`

**Files created/modified:**
- `scripts/ml/__init__.py` (new)
- `scripts/ml/run_eda.py` (new)
- `data/processed/kuairec/reports/eda_report.json` (generated, not committed)
- `data/processed/kuairec/reports/eda_report.md` (generated, not committed)

**Dependencies:** Phase 2 Parquet files

**Expected Outputs:** EDA report with distributions, statistics, identified outliers

**Tests:** None (EDA is exploratory output; correctness verified by human review)

**Acceptance Criteria:**
- Script runs without error from project root
- Report covers all 5 Parquet files
- Ruff clean

---

### WP2 — Target Label Definition and ADR

**Objective:** Formally define the relevance signal, binary threshold, and document the
decision as an Architecture Decision Record.

**Technical Tasks:**
1. Record ADR-P3-01: Target label = watch_ratio, binary threshold = 0.5
2. Record ADR-P3-02: Phase redefinition decision
3. Record ADR-P3-03: Small matrix vs big matrix decision (use small_matrix for Phase 3)
4. Update `docs/decisions/architecture-decisions.md`

**Files created/modified:**
- `docs/decisions/architecture-decisions.md` (modified — append ADRs)

**Dependencies:** WP1 (EDA confirms signal distribution)

**Expected Outputs:** ADR-P3-01 through ADR-P3-03 documented

**Acceptance Criteria:**
- Three ADRs written with context, decision, rationale, consequences
- Threshold value explicitly stated with justification

---

### WP3 — ML Directory Scaffold

**Objective:** Create the `ml/` module directory structure and abstract base class.

**Technical Tasks:**
1. Create `ml/` directory with all subdirectories (see Section 2 above)
2. Write `ml/__init__.py`, `ml/data/__init__.py`, `ml/features/__init__.py`,
   `ml/baselines/__init__.py`, `ml/evaluation/__init__.py`, `ml/experiments/__init__.py`
3. Write `ml/baselines/base.py` — abstract `BaseRecommender` with protocol:
   - `fit(train_df: pl.DataFrame) -> None`
   - `recommend(user_id: int, n: int, exclude_seen: set[int]) -> list[int]`
   - `name: str` property
4. Write `ml/config.py` — `ExperimentConfig` dataclass with all configurable parameters
5. Create `experiments/results/.gitkeep` and add rule to `.gitignore`

**Files created/modified:**
- `ml/` directory tree (all new)
- `experiments/results/.gitkeep` (new)
- `.gitignore` (append `experiments/results/*.json`)

**Dependencies:** None

**Expected Outputs:** Importable `ml` module; abstract base class passing mypy-equivalent type checks

**Acceptance Criteria:**
- `from ml.baselines.base import BaseRecommender` works from project root
- Ruff clean

---

### WP4 — Data Loading and Train/Validation/Test Split

**Objective:** Implement temporal splitting logic and persist the three split files.

**Technical Tasks:**
1. Write `ml/data/loader.py` — lazy Parquet loading with caching
2. Write `ml/data/splits.py` — `create_temporal_split(df, train_ratio, val_ratio)` function
   - Sort by timestamp, fall back to date, then assign null-timestamp rows to train
   - Write output to `data/processed/kuairec/splits/`
   - Write `split_manifest.json` with cutoffs, row counts, null-row count in train
3. Write `scripts/ml/prepare_splits.py` — runner that calls `splits.py` and saves output
4. Write `backend/tests/ml/conftest.py` — synthetic fixtures
5. Write `backend/tests/ml/test_splits.py` — correctness tests

**Files created/modified:**
- `ml/data/loader.py` (new)
- `ml/data/splits.py` (new)
- `scripts/ml/prepare_splits.py` (new)
- `backend/tests/ml/__init__.py` (new)
- `backend/tests/ml/conftest.py` (new)
- `backend/tests/ml/test_splits.py` (new)

**Dependencies:** WP3

**Expected Outputs:**
- `data/processed/kuairec/splits/{train,val,test}.parquet` (generated)
- `data/processed/kuairec/splits/split_manifest.json` (generated)
- All split tests passing

**Acceptance Criteria:**
- Train + val + test row counts sum to total rows
- No test-set timestamps appear in training data
- Null-timestamp rows assigned to train
- Tests pass; Ruff clean

---

### WP5 — Evaluation Framework

**Objective:** Implement all evaluation metrics and the evaluation runner.

**Technical Tasks:**
1. Write `ml/evaluation/metrics.py`:
   - `precision_at_k(recommended, relevant, k)`
   - `recall_at_k(recommended, relevant, k)`
   - `ndcg_at_k(recommended, relevant_scores, k)` — using numpy for DCG computation
   - `map_at_k(recommended, relevant, k)`
   - `catalog_coverage(all_recommendations, catalog_size)`
   - `intra_list_diversity(recommendations, item_categories)`
   - `novelty(recommendations, item_popularity)`
2. Write `ml/evaluation/evaluator.py` — `Evaluator` class:
   - Takes a fitted recommender, test split, seen items per user
   - Runs evaluation for K ∈ {5, 10, 20}
   - Returns per-user metrics and aggregate statistics
3. Write `backend/tests/ml/test_metrics.py` — verify each metric on known inputs

**Files created/modified:**
- `ml/evaluation/metrics.py` (new)
- `ml/evaluation/evaluator.py` (new)
- `backend/tests/ml/test_metrics.py` (new)

**Dependencies:** WP3; NumPy added to `[ml]` optional deps in `pyproject.toml`

**Expected Outputs:** All metrics produce correct values on synthetic test cases

**Acceptance Criteria:**
- NDCG@K = 1.0 when perfect ranking on known test case
- Precision@K = 0.0 for random baseline on known test case
- Tests pass; Ruff clean

---

### WP6 — Random and Global Popularity Baselines

**Objective:** Implement the two simplest baselines.

**Technical Tasks:**
1. Write `ml/baselines/random_recommender.py`
2. Write `ml/baselines/popularity_recommender.py`
3. Add numpy dependency to `backend/pyproject.toml` under `[project.optional-dependencies.ml]`
4. Write `backend/tests/ml/test_random_recommender.py`
5. Write `backend/tests/ml/test_popularity_recommender.py`

**Files created/modified:**
- `ml/baselines/random_recommender.py` (new)
- `ml/baselines/popularity_recommender.py` (new)
- `backend/pyproject.toml` (modified)
- `backend/tests/ml/test_random_recommender.py` (new)
- `backend/tests/ml/test_popularity_recommender.py` (new)

**Dependencies:** WP3, WP4, WP5

**Acceptance Criteria:**
- Random recommender: N unique items, no seen items, deterministic with seed
- Popularity recommender: items ranked by weighted score (popularity metric defined in ADR)
- Both integrate with Evaluator; tests pass; Ruff clean

---

### WP7 — Category-Based and Trending Baselines

**Objective:** Implement the two popularity variants that use category or temporal features.

**Technical Tasks:**
1. Write `ml/features/item_features.py` — category lookups from item Parquet files
2. Write `ml/baselines/category_popularity_recommender.py`
3. Write `ml/baselines/trending_recommender.py`
4. Write `backend/tests/ml/test_category_recommender.py`
5. Write `backend/tests/ml/test_trending_recommender.py`

**Files created/modified:**
- `ml/features/item_features.py` (new)
- `ml/baselines/category_popularity_recommender.py` (new)
- `ml/baselines/trending_recommender.py` (new)
- `backend/tests/ml/test_category_recommender.py` (new)
- `backend/tests/ml/test_trending_recommender.py` (new)

**Dependencies:** WP3, WP4, WP5, WP6

**Acceptance Criteria:**
- Category recommender uses only training data for popularity
- Trending recommender correctly filters by recency window; falls back gracefully
- Tests pass; Ruff clean

---

### WP8 — User-Preference Heuristic Baseline

**Objective:** Implement the most sophisticated heuristic baseline.

**Technical Tasks:**
1. Write `ml/features/user_features.py` — user category affinity computation
2. Write `ml/baselines/user_preference_recommender.py`
3. Write `backend/tests/ml/test_user_preference_recommender.py`

**Files created/modified:**
- `ml/features/user_features.py` (new)
- `ml/baselines/user_preference_recommender.py` (new)
- `backend/tests/ml/test_user_preference_recommender.py` (new)

**Dependencies:** WP7

**Acceptance Criteria:**
- Category affinity vector computed from training watch_ratio
- Cold-start users receive global popularity fallback
- Tests pass; Ruff clean

---

### WP9 — Experiment Runner and Results

**Objective:** Run all five baselines on the test split, record results, and produce a
comparison table.

**Technical Tasks:**
1. Write `ml/experiments/recorder.py` — JSON result writer
2. Write `scripts/ml/run_baselines.py` — orchestrates: load splits → fit all 5 baselines
   → evaluate at K ∈ {5, 10, 20} → write JSON results
3. Run the experiment; inspect results
4. Write `data/processed/kuairec/reports/baseline_results.md` (generated)

**Files created/modified:**
- `ml/experiments/recorder.py` (new)
- `scripts/ml/run_baselines.py` (new)
- `experiments/results/<run_id>.json` (generated, not committed)
- `data/processed/kuairec/reports/baseline_results.md` (generated, not committed)

**Dependencies:** WP3–WP8

**Acceptance Criteria:**
- All 5 baselines run without error
- Results JSON contains all required fields
- Random < Popularity (by any metric) — sanity check passes
- Ruff clean

---

### WP10 — Documentation

**Objective:** Write all Phase 3 documentation.

**Technical Tasks:**
1. `docs/ml/evaluation-methodology.md` — metric definitions, K values, protocol, leakage prevention
2. `docs/ml/baseline-algorithms.md` — description of all 5 baselines, inputs, limitations
3. `docs/ml/experiment-guide.md` — how to run baselines, configure experiments, read results
4. `docs/phases/phase-3-baseline-guide.md` — setup + execution guide
5. Update `AGENTS.md` — Phase 3 status to Complete; Phase 4 scope updated
6. Update `README.md` — Phase 3 complete, new doc links
7. Update `docs/decisions/architecture-decisions.md` — ADR-P3-01 through ADR-P3-03

**Files created/modified:**
- `docs/ml/evaluation-methodology.md` (new)
- `docs/ml/baseline-algorithms.md` (new)
- `docs/ml/experiment-guide.md` (new)
- `docs/phases/phase-3-baseline-guide.md` (new)
- `AGENTS.md` (modified)
- `README.md` (modified)
- `docs/decisions/architecture-decisions.md` (modified)

**Dependencies:** WP1–WP9

**Acceptance Criteria:**
- All documentation files created
- AGENTS.md Phase 3 marked Complete, Phase 4 scope updated
- Ruff clean on `ml/` and `scripts/ml/`

---

### WP11 — Final Quality Gates and Commit

**Objective:** Run all quality checks, verify no data files are committed, and push to GitHub.

**Technical Tasks:**
1. `backend\.venv\Scripts\pytest.exe backend/tests/ -v` — all tests pass (target: 62 + ~40 new = ~100 total)
2. `ruff check ml/ scripts/ml/ backend/tests/ml/` — clean
3. `ruff format --check ml/ scripts/ml/ backend/tests/ml/` — clean
4. `git status` — confirm no data files, parquet files, or JSON result files staged
5. Commit with conventional message: `feat(phase3): add baseline recommender systems`
6. Push to GitHub

**Acceptance Criteria:**
- All tests pass
- Ruff clean
- No data files committed
- Commit pushed successfully

---

## 12. Architecture Decision Records Planned

| ADR ID | Title | Decision |
|---|---|---|
| ADR-P3-01 | Phase 3 Redefinition | Phase 3 = Baseline Recommendation Systems; content-based/collaborative → Phase 4 |
| ADR-P3-02 | Relevance Signal | watch_ratio continuous; binary threshold = 0.5 for ranking metrics |
| ADR-P3-03 | Dataset Scope | Use small_matrix only; big_matrix deferred to Phase 4 |
| ADR-P3-04 | Temporal Split | Sort by timestamp; null-timestamp rows assigned to train |
| ADR-P3-05 | Popularity Metric | Weighted score = mean(watch_ratio) × log(1 + count) |
| ADR-P3-06 | Experiment Tracking | JSON files in Phase 3; MLflow deferred to Phase 4 |
| ADR-P3-07 | ml/ as standalone module | No FastAPI/DB imports in ml/; separate from backend/app/ |

---

## 13. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| 99.6% density makes evaluation trivial — all recommenders perform similarly | Medium | Medium | Use temporal split to introduce meaningful train/test difference; also evaluate on a held-out user subset |
| Null timestamps in 3.89% of rows affect split quality | Low | Low | Assign to train; document in split manifest |
| watch_ratio threshold = 0.5 is arbitrary | Medium | Medium | Record as ADR; run sensitivity analysis (also test at 0.3 and 0.7) in WP9 |
| NumPy import conflicts with existing venv | Low | Low | Pin to `>=1.26.0`; test after install |
| EDA finds dataset issues that invalidate Phase 2 assumptions | Low | High | WP2 ADRs are written after WP1 EDA; if issues found, present for approval before proceeding |

---

## 14. Phase 3 Acceptance Criteria

Phase 3 is complete only when ALL of the following are true:

1. EDA report generated and reviewed
2. ADR-P3-01 through ADR-P3-07 documented
3. `ml/` module created and importable
4. Temporal train/val/test split created and verified
5. `evaluation/metrics.py` verified on known synthetic inputs
6. All 5 baseline recommenders implemented and tested
7. Experiment runner produces valid JSON results
8. Random baseline performs below popularity baseline (sanity check)
9. All existing 62 tests continue to pass
10. New test count ≥ 40 (target ~100 total tests)
11. Ruff clean on all new code
12. No data files committed to Git
13. All 10 documentation files written
14. AGENTS.md and README.md updated
15. Commit pushed to GitHub

---

## 15. Implementation Sequence

```
WP1 (EDA)
  ↓
WP2 (Target Label + ADRs)
  ↓
WP3 (ml/ scaffold + base.py)
  ↓
WP4 (data loading + splits)
  ↓
WP5 (evaluation metrics)
  ↓
WP6 (random + popularity baselines)
  ↓
WP7 (category + trending baselines)
  ↓
WP8 (user-preference heuristic)
  ↓
WP9 (experiment runner + results)
  ↓
WP10 (documentation)
  ↓
WP11 (quality gates + commit)
```

Each WP is independently verifiable. WP6–WP8 can be partially parallelized but
sequential is safer for first pass.
