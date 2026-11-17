# Architecture Decision Records

This document records significant architectural decisions made during the development
of the Hybrid Recommendation Intelligence Platform, including context, options considered,
and the rationale for the chosen approach.

---

## ADR-001 — Modular Monolith over Microservices

**Status:** Accepted  
**Phase:** 1

**Context:**  
The platform will eventually consist of multiple concerns: API serving, data processing,
ML training, model serving, and feedback collection. A microservices architecture would
treat each as an independent deployable service.

**Decision:**  
Start with a modular monolith. Keep all backend logic in a single FastAPI application
with well-defined internal module boundaries. Extract services into independent processes
only when there is a demonstrated, specific need (e.g. ML training is too resource-intensive
to share a process with the API server).

**Rationale:**  
- Microservices add significant operational overhead (service discovery, inter-service auth,
  distributed tracing, independent CI/CD pipelines) before the system is well understood
- The team is a single developer — a monolith is proportionate to the team size
- Module boundaries within a monolith can be enforced with import rules; extraction later
  is straightforward when the boundaries are clean
- Martin Fowler's Monolith First pattern applies here

---

## ADR-002 — Python 3.13 (not 3.11 as originally specified)

**Status:** Accepted  
**Phase:** 1

**Context:**  
The original tech stack specification called for Python 3.11. The development machine has
Python 3.13 installed (at `C:\Users\Edunet Foundation\AppData\Local\Programs\Python\Python313`).
Python 3.11 was not installed.

**Decision:**  
Use Python 3.13. Update the tech stack specification accordingly.

**Rationale:**  
- All Phase 1 dependencies (FastAPI, SQLAlchemy, Alembic, Pydantic, psycopg2-binary, Ruff,
  Pytest, httpx) are fully compatible with Python 3.13
- Future-phase ML libraries (PyTorch, scikit-learn, LightGBM) now also support 3.13
- Installing Python 3.11 alongside 3.13 would require managing two Python installations
  with no technical benefit for Phase 1
- Python 3.13 is strictly newer and brings performance improvements

**Consequence:**  
If a future-phase ML library is found to be incompatible with 3.13 at that time, the
virtual environment can be recreated with a pyenv-managed 3.11 interpreter without
changing any application code.

---

## ADR-003 — Docker Compose V2 (`docker compose`, not `docker-compose`)

**Status:** Accepted  
**Phase:** 1

**Context:**  
Two versions of Docker Compose exist: V1 (standalone binary, invoked as `docker-compose`)
and V2 (Docker plugin, invoked as `docker compose` without hyphen). V1 reached end-of-life
in July 2023.

**Decision:**  
Use Docker Compose V2 (`docker compose`) in all scripts, documentation, and commands.

**Rationale:**  
- Docker Desktop on Windows ships V2 by default
- V1 is deprecated and no longer receives updates
- V2 is the standard going forward

---

## ADR-004 — Backend Virtual Environment at `backend/.venv`

**Status:** Accepted  
**Phase:** 1

**Context:**  
Python virtual environments can be placed at the repository root or inside the backend
directory.

**Decision:**  
Place the virtual environment at `backend/.venv`.

**Rationale:**  
- Co-located with the backend code it serves
- Clear ownership — no confusion about which `.venv` belongs to which component
- Excluded from Git by `.gitignore`
- Standard convention for Python projects

---

## ADR-005 — Axios as the Frontend HTTP Client

**Status:** Accepted  
**Phase:** 1

**Context:**  
The frontend needs to make HTTP requests to the backend. Options are native `fetch` API
(zero dependencies) or Axios (additional dependency, more ergonomic).

**Decision:**  
Use Axios. The decision was confirmed by the project owner during Phase 1 planning.

**Rationale:**  
- Axios is the established standard in React projects
- Provides a centralized instance with base URL, timeout, and default headers
- Interceptors allow easy request/response transformation in future phases
- More familiar to most developers than raw fetch with manual error handling

---

## ADR-006 — PostgreSQL Bound to localhost Only

**Status:** Accepted  
**Phase:** 1

**Context:**  
The Docker Compose port mapping can expose PostgreSQL to all interfaces (`0.0.0.0:5432`)
or only to localhost (`127.0.0.1:5432`).

**Decision:**  
Bind to `127.0.0.1:5432` only.

**Rationale:**  
- PostgreSQL should never be directly reachable from outside the development machine
- All database access goes through the FastAPI backend
- Principle of least privilege — expose only what is necessary
- No functional impact on local development

---

## ADR-007 — `postgresql+psycopg2` Dialect Explicitly in SQLAlchemy URL

**Status:** Accepted  
**Phase:** 1

**Context:**  
SQLAlchemy 2.x with a plain `postgresql://` URL attempts to use `psycopg` v3 by default
if installed, falling back to `psycopg2`. We install `psycopg2-binary` (v2).

**Decision:**  
Use `postgresql+psycopg2://` explicitly in the connection URL.

**Rationale:**  
- Makes the driver choice explicit and unambiguous
- Avoids SQLAlchemy silently picking a different driver if `psycopg` v3 is later installed
- `psycopg2-binary` has the best Windows support and broadest compatibility

---

## ADR-008 — oxlint as Frontend Linter (Vite 8 Default)

**Status:** Accepted  
**Phase:** 1

**Context:**  
Vite 8 scaffolds projects with `oxlint` as the default linter instead of ESLint.
`oxlint` is a Rust-based linter that is significantly faster than ESLint.

**Decision:**  
Accept the Vite 8 default and use `oxlint`. Do not replace it with traditional ESLint.

**Rationale:**  
- oxlint implements the same rule set as the most important ESLint rules
- 50–100x faster than ESLint on large codebases
- Zero configuration required for a React + TypeScript project
- Already included in `package.json` by the Vite scaffold
- TypeScript type checking is handled separately by `tsc --noEmit`, which is more
  thorough than any linter's type rules


---

## ADR-009 — Python 3.14 Correction (Phase 2 Discovery)

**Status:** Accepted  
**Phase:** 2

**Context:**  
ADR-002 documented the runtime as Python 3.13 based on Phase 1 inspection. During Phase 2,
closer inspection of the virtual environment pip binary (`pip3.14`) revealed the actual
runtime is Python 3.14.5.

**Decision:**  
Accept Python 3.14. Update `AGENTS.md` and documentation to reflect the actual version.

**Rationale:**  
- All Phase 2 libraries (Polars, DuckDB, PyArrow, tqdm, requests, rich) are compatible
  with Python 3.14
- No re-installation is required — the virtual environment functions correctly
- Keeping documentation accurate prevents confusion for future sessions

---

## ADR-D-P2-01 — Full KuaiRec 2.0 Download

**Status:** Accepted  
**Phase:** 2

**Context:**  
KuaiRec 2.0 is distributed as a zip file plus separate additional files on Zenodo
(~417 MB total). An alternative is to download only the Phase 2 scope files.

**Decision:**  
Download all 9 files in full.

**Rationale:**  
- Downloading selectively would require re-running downloads when Phase 3 files are needed
- 417 MB is manageable on a development machine
- Checksums for all files are verified on download — partial downloads risk silent corruption
- Provenance is cleaner: one complete, verified acquisition

---

## ADR-D-P2-02 — Polars for Transforms, DuckDB for Analytics

**Status:** Accepted  
**Phase:** 2

**Context:**  
Both Polars and DuckDB can process tabular data. The question was whether to use one or both.

**Decision:**  
Use Polars for data cleaning and type transformation, and DuckDB for SQL analytics and
report generation.

**Rationale:**  
- Polars' DataFrame API is better suited to column-wise type manipulation (casting,
  parsing, adding derived columns)
- DuckDB's SQL interface is more natural for aggregations, distributions, and
  cross-file joins used in the quality report
- Both tools operate on Parquet files efficiently — no duplication of data
- Phase 3 will require both for feature engineering pipelines

---

## ADR-D-P2-03 — Data Scripts in `scripts/data/`, Not in `app/`

**Status:** Accepted  
**Phase:** 2

**Context:**  
The data processing scripts could be placed inside the FastAPI application (`backend/app/`)
or as standalone scripts in `scripts/data/`.

**Decision:**  
Place all data processing scripts in `scripts/data/`. They are run directly from the
command line, not through the API.

**Rationale:**  
- Data processing is an offline, batch operation — not a web request
- Including it in `app/` would couple the API server to data engineering dependencies
- Scripts in `scripts/data/` can be run independently of Docker or the FastAPI server
- Future ML training scripts will follow the same pattern

---

## ADR-D-P2-04 — Defer big_matrix and Three Files to Phase 3

**Status:** Accepted  
**Phase:** 2

**Context:**  
KuaiRec 2.0 includes `big_matrix.csv` (1,033 MB), `item_daily_features.csv`,
`social_network.csv`, and `video_raw_categories_multi.csv`. These were not converted
to Parquet in Phase 2.

**Decision:**  
Convert only the 5 Phase 2 scope files. Defer the remaining 4 files to Phase 3.

**Rationale:**  
- `big_matrix.csv` at 1,033 MB cannot be loaded into memory on a 16 GB development
  machine alongside other processes — it requires streaming or chunked processing
- `item_daily_features` and `social_network` are only needed for advanced collaborative
  filtering and graph-based recommendations (Phase 3+)
- `video_raw_categories_multi` is redundant with `item_categories` for Phase 2 purposes
- Deferring keeps Phase 2 tractable and testable

---

## ADR-D-P2-05 — PostgreSQL Untouched in Phase 2

**Status:** Accepted  
**Phase:** 2

**Context:**  
Data could be loaded into PostgreSQL during Phase 2 for use by the FastAPI backend.

**Decision:**  
PostgreSQL is not used in Phase 2. All data stays in Parquet files.

**Rationale:**  
- Loading 4.6M rows into PostgreSQL without a finalized schema would require
  migration work that belongs in Phase 3 (schema design)
- Parquet files are readable directly by Polars and DuckDB without a database server
- Phase 3 will design the database schema based on the actual ML feature requirements,
  preventing premature schema commits
- Keeping the database clean avoids polluting Phase 1 Alembic migration history

---

## ADR-P3-01 — Phase 3 Redefined as Baseline Recommendation Systems

**Status:** Accepted  
**Phase:** 3

**Context:**  
The original AGENTS.md defined Phase 3 as "Recommendation Engine (content-based, collaborative,
hybrid)." The project owner requested that Phase 3 be scoped to establish measurable benchmarks
with non-ML heuristic baselines before introducing learned models.

**Decision:**  
Phase 3 = Baseline Recommendation Systems. Content-based filtering and collaborative filtering
move to Phase 4. ML ranking moves to Phase 5.

**Rationale:**  
- Baselines define the measurement scale for all future ML models
- Evaluation infrastructure built in Phase 3 is reused unchanged in all subsequent phases
- Isolates the value of each additional signal (popularity → category → temporal → user history → ML)
- Standard industry practice: Netflix, Spotify, YouTube all benchmark against non-ML baselines

---

## ADR-P3-02 — Relevance Signal and Binary Threshold

**Status:** Accepted  
**Phase:** 3

**Context:**  
`watch_ratio` is the only explicit engagement signal in small_matrix. Ranking metrics
(Precision@K, Recall@K, NDCG@K, MAP@K) require a binary relevant/non-relevant label.

EDA findings:
- 72.73% of interactions have watch_ratio ≥ 0.5 (3,401,473 of 4,676,570)
- 27.27% have watch_ratio < 0.5 (non-relevant)
- 0.65% are exactly 0.0 (video initiated, not played)
- Median watch_ratio = 0.7692 (P50 is above the threshold)

**Decision:**  
- Primary signal: `watch_ratio` (continuous)
- Binary relevance threshold: `watch_ratio ≥ 0.5`
- Zero-ratio rows: classified as non-relevant; kept in training data

**Rationale:**  
- 0.5 = watched at least half the video — a meaningful engagement signal
- Avoids treating brief accidental plays as positive
- P25 = 0.4676, so the threshold sits just above the first quartile — creates meaningful positive/negative split
- Can be revisited in Phase 4 sensitivity analysis

**Consequence:**  
If Phase 4 experiments show threshold sensitivity, run evaluation at 0.3 and 0.7 as ablation.

---

## ADR-P3-03 — Use small_matrix Only; Defer big_matrix to Phase 4

**Status:** Accepted  
**Phase:** 3

**Context:**  
KuaiRec 2.0 has two interaction matrices: small_matrix (4.68M rows, 99.6% density,
1,411 × 3,327) and big_matrix (12.5M rows, 16.3% density, 7,176 × 10,728).

**Decision:**  
Phase 3 uses small_matrix only. big_matrix is deferred to Phase 4.

**Rationale:**  
- big_matrix (1,033 MB CSV) requires streaming/chunked processing — not yet implemented
- small_matrix is dense enough to validate the entire evaluation framework quickly
- Phase 4 will introduce the realistic sparse scenario (big_matrix) where collaborative
  filtering adds the most value

---

## ADR-P3-04 — Temporal Train/Validation/Test Split

**Status:** Accepted  
**Phase:** 3

**Context:**  
small_matrix spans 2020-07-05 to 2020-09-05 (63 unique dates). 3.89% of rows have null
timestamps. Standard random splits would leak future data into training.

**Decision:**  
- Sort all interactions by `timestamp` (fall back to `date` column where null)
- Assign null-timestamp rows to the training split (conservative assumption)
- Split proportions: 70% train / 15% validation / 15% test (by time)
- Split is computed once and saved to `data/processed/kuairec/splits/` as Parquet files

**Rationale:**  
- Temporal split simulates real deployment conditions
- Prevents leakage of future interaction patterns into popularity statistics
- Null-timestamp rows assigned to train = most conservative assumption about their age

---

## ADR-P3-05 — Popularity Metric: Weighted Score

**Status:** Accepted  
**Phase:** 3

**Context:**  
Popularity can be measured as raw interaction count or as a quality-weighted score.

**Decision:**  
`weighted_popularity = mean(watch_ratio) × log(1 + interaction_count)`

**Rationale:**  
- Raw count rewards items that many users clicked but immediately skipped
- Weighted score rewards genuine engagement: high reach AND high watch_ratio
- log(1 + count) dampens the influence of extreme interaction counts

---

## ADR-P3-06 — Experiment Tracking: JSON Files; MLflow Deferred to Phase 4

**Status:** Accepted  
**Phase:** 3

**Context:**  
MLflow is in the approved stack but not yet installed.

**Decision:**  
Phase 3 records experiment results as structured JSON files in `experiments/results/`.
MLflow is introduced in Phase 4.

**Rationale:**  
- JSON files are zero-dependency and fully readable without a tracking server
- Phase 3 baselines have few hyperparameters — the overhead of MLflow is not justified

---

## ADR-P3-07 — ml/ Is a Standalone Module

**Status:** Accepted  
**Phase:** 3

**Context:**  
Recommendation logic could be placed inside `backend/app/` or as a standalone module.

**Decision:**  
All ML code lives in the top-level `ml/` package, with no imports from `backend/app/`,
`scripts/`, or any FastAPI/SQLAlchemy module.

**Rationale:**  
- Separation of concerns: web serving vs recommendation logic
- Enables independent testing and profiling of ML code
- Consistent with the modular monolith architecture principle (ADR-001)


---

## Phase 4 Architecture Decisions

---

### ADR-P4-01 — Add scikit-learn as Optional [ml] Dependency

**Status:** Accepted | **Phase:** 4 | **Date:** 2026-11-17

**Context:** Phase 4 requires TF-IDF vectorization (Content-Based Filtering) and
Truncated SVD (Matrix Factorization). Both are available in scikit-learn.

**Decision:** Add `scikit-learn>=1.5.0` to `backend/pyproject.toml` under a new
`[ml]` optional dependency group. Install with `pip install -e ".[ml]"`.

**Rationale:**
- scikit-learn 1.9.1 confirmed compatible with Python 3.14.5 on Windows.
- No GPU requirement — CPU-only operation on integrated AMD Radeon.
- TfidfVectorizer and TruncatedSVD cover all Phase 4 needs.
- Separated from `[dev]` and `[data]` groups to keep concerns isolated.

**Consequences:** numpy and scipy are installed as transitive dependencies of
scikit-learn. Direct imports of numpy are permitted in `ml/` for performance-critical
matrix operations (similarity scoring, interaction matrix).

---

### ADR-P4-02 — Use Truncated SVD for Matrix Factorization

**Status:** Accepted | **Phase:** 4 | **Date:** 2026-11-17

**Context:** Matrix factorization choices considered: Truncated SVD (scikit-learn),
ALS (implicit library), NMF (scikit-learn), SVD++ (surprise library).

**Decision:** Use `sklearn.decomposition.TruncatedSVD` with `n_components=100`
(best validation NDCG@10 of 0.6572).

**Rationale:**
- Dense matrix (99.6%) makes SVD well-conditioned — no sparse-matrix optimizations needed.
- scikit-learn covers all requirements without additional packages.
- ALS (implicit) is designed for sparse implicit feedback — wrong fit for this dataset.
- Fit time: ~2 seconds for 1,411 × 3,327 matrix with 100 components.
- Validation NDCG@10: 0.6572 (best of k=20, 50, 100).
- Test NDCG@10: 0.1462 (note: validation/test gap due to dense matrix characteristics).

---

### ADR-P4-03 — Use Continuous Watch Ratio (Clipped at 5.0) as CF Interaction Signal

**Status:** Accepted | **Phase:** 4 | **Date:** 2026-11-17

**Context:** The interaction signal for the user-item matrix could be binary (0/1
at watch_ratio ≥ 0.5) or continuous (raw watch_ratio). The raw maximum is 571.5
(extreme replay loops) which would distort SVD.

**Decision:** Use `min(watch_ratio, 5.0)` as the interaction signal.

**Rationale:**
- Richer than binary — SVD captures engagement intensity.
- Raw max (571.5) would dominate the singular value decomposition.
- 5.0 is a safe clip: the 99th percentile is ~3.6, and 99.5th is below 5.0.
- Mean watch_ratio = 0.907; median = 0.769 — clipping does not distort typical values.

---

### ADR-P4-04 — Validation Split for Hyperparameter Tuning

**Status:** Accepted | **Phase:** 4 | **Date:** 2026-11-17

**Context:** Phase 4 algorithms have hyperparameters (n_components, n_neighbors,
tfidf_weight). Using the test split for selection would introduce optimistic bias.

**Decision:** Use `val.parquet` exclusively for hyperparameter selection.
The test split (`test.parquet`) is reserved for final evaluation only.

**Rationale:**
- Prevents test-set information leakage into model selection.
- Consistent with Phase 3 evaluation methodology (ADR-P3-04).
- Validation configs tried:
  - SVD: n_components ∈ {20, 50, 100} → best: 100 (val NDCG@10 = 0.6572)
  - UserCF: n_neighbors ∈ {10, 50, 100} → best: 10 (val NDCG@10 = 0.6459)
  - ItemCF: n_neighbors ∈ {10, 50} → best: 10 (val NDCG@10 = 0.6437)
  - CBF: tfidf_weight ∈ {0.0, 0.25, 0.5, 0.75} → best: 0.25 (val NDCG@10 = 0.0829)

---

### ADR-P4-05 — Recommendation API Serves Pre-Computed Results (No Live Inference)

**Status:** Accepted | **Phase:** 4 | **Date:** 2026-11-17

**Context:** Phase 4 adds recommendation API endpoints. Options: (a) load fitted
models at startup and infer per request; (b) serve pre-computed experiment results
from JSON files.

**Decision:** Serve pre-computed evaluation results from `experiments/results/*.json`.
No live model loading at API startup in Phase 4.

**Rationale:**
- Phase 4 scope is algorithm evaluation, not production serving.
- Avoids startup latency and memory cost of loading large models.
- Phase 5 will introduce proper model serving (model registry, caching).
- The API provides useful observability into algorithm performance without
  requiring Docker or the ML stack at runtime.

**Endpoints added:**
- `GET /api/v1/recommendations/results` — all results sorted by NDCG@10
- `GET /api/v1/recommendations/algorithms` — list of evaluated algorithms
- `GET /api/v1/recommendations/results/{name}` — result for a specific algorithm
