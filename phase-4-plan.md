# Phase 4 — ML-Based Recommendation Systems
## Technical Implementation Blueprint

**Status:** Awaiting approval
**Preceded by:** Phase 3 — Baseline Recommendation Systems ✅ Complete
**Followed by:** Phase 5 — ML Ranking and Serving (LightGBM, MLflow, candidate generation)

---

## 0. Architecture Audit Findings

### 0.1 Reusable Infrastructure (do NOT modify)

| Component | Location | Reuse in Phase 4 |
|---|---|---|
| `BaseRecommender` ABC | `ml/baselines/base.py` | Extend for all Phase 4 algorithms |
| `Evaluator` class | `ml/evaluation/evaluator.py` | Use as-is — no changes needed |
| 7 evaluation metrics | `ml/evaluation/metrics.py` | Use as-is — pure Python, no dependencies |
| `record_experiment()` | `ml/experiments/recorder.py` | Use as-is — same JSON schema |
| `load_splits()` | `ml/data/splits.py` | Use as-is — same train/val/test Parquet |
| `load_item_category_map()` | `ml/features/item_features.py` | Extend to support multi-level categories |
| `compute_user_category_affinity()` | `ml/features/user_features.py` | Extend for richer user profiles |
| `ExperimentConfig` | `ml/config.py` | Extend with Phase 4 hyperparameters |
| FastAPI skeleton | `backend/app/` | Add new router for recommendations |
| Pytest fixtures | `backend/tests/ml/conftest.py` | Extend with ML-specific fixtures |

### 0.2 New Directories Required

```
ml/
├── content_based/          NEW — Content-Based Filtering
├── collaborative/          NEW — Collaborative Filtering
└── (existing dirs unchanged)

backend/app/api/v1/endpoints/
└── recommendations.py      NEW — Recommendation API endpoints

backend/tests/ml/
├── test_content_based.py   NEW
└── test_collaborative.py   NEW

docs/ml/
├── content-based-filtering.md   NEW
├── collaborative-filtering.md   NEW
└── phase4-results.md            NEW (generated after evaluation)

docs/phases/
└── phase-4-ml-guide.md          NEW

scripts/ml/
├── run_content_based.py    NEW
└── run_collaborative.py    NEW
```

### 0.3 Dataset Characteristics Affecting Algorithm Design

| Characteristic | Value | Phase 4 Implication |
|---|---|---|
| Matrix density | 99.62% | Near-full matrix — CF algorithms see very high coverage; SVD is viable without sparsity handling |
| Interactions | 4,676,570 | In-memory viable on 16 GB RAM; Polars DataFrame operations are efficient |
| Items with category | 3,327 / 3,327 (100%) | No missing item features — content-based is clean |
| Caption non-null | 87.37% | TF-IDF viable but 12.63% items have no caption; fallback needed |
| Category coverage | 20 first-level, 138 second-level | Rich categorical hierarchy for content-based item profiles |
| Relevance prevalence | 72.73% | High positive rate; classification easier than typical; focus on ranking quality |
| Watch ratio range | [0.0, 571.5] | Clip at reasonable threshold (e.g., 5.0) before using as regression target |
| Users in splits | 1,411 | Small enough that user-based CF is viable in-memory |
| Items in splits | 3,327 | Small enough that item-item similarity matrix fits in memory |

### 0.4 Decisions Required from Owner Before Implementation

| Decision | Options | Recommendation | Reason |
|---|---|---|---|
| **D1 — scikit-learn installation** | Add to `[data]` or new `[ml]` extras | New `[ml]` optional group | Keeps dev/data/ml concerns separate |
| **D2 — Matrix factorization method** | Truncated SVD (sklearn) vs ALS (implicit) | Truncated SVD | No GPU needed; scikit-learn sufficient; ALS needs `implicit` package |
| **D3 — Interaction signal for CF** | Binary (0/1 at wr≥0.5) vs continuous watch_ratio | Continuous watch_ratio clipped at 5.0 | Richer signal; SVD can handle float matrix |
| **D4 — API integration scope** | Phase 4 only, or defer API to Phase 5 | Implement in Phase 4 | Plan explicitly includes it; Phase 5 is ranking, not serving |
| **D5 — big_matrix introduction** | Evaluate on big_matrix in Phase 4 or defer | Defer to Phase 5 | big_matrix requires sparse matrix streaming; adds complexity without testing new algorithms |
| **D6 — Hit Rate@K as 8th metric** | Add Hit Rate@K to evaluation or keep 7 | Add Hit Rate@K | Requested in prompt; trivial to add alongside existing metrics |

---

## 1. Phase Scope and Non-Scope

### In Scope
- Phase 4A: Content-Based Filtering (item feature engineering + user profiles + cosine similarity)
- Phase 4B: Collaborative Filtering (user-based, item-based, matrix factorization via Truncated SVD)
- Cold-start handling with Phase 3 baseline fallback
- Evaluation against all 5 Phase 3 baselines on the same temporal splits
- FastAPI endpoints for recommendation serving
- JSON experiment results + Markdown comparison report
- Full test coverage
- Documentation updates

### Not In Scope
- Hybrid combining of CBF + CF (Phase 5)
- big_matrix evaluation (Phase 5)
- LightGBM/XGBoost ranking (Phase 5)
- MLflow tracking (Phase 5 per ADR-P3-06)
- Neural / deep learning approaches
- Real-time model serving or model deployment
- User authentication or personalization infrastructure

---

## 2. Architecture Design

### 2.1 Module Organization

```
ml/
├── content_based/
│   ├── __init__.py
│   ├── item_encoder.py          Item feature matrix builder (one-hot + TF-IDF + normalization)
│   ├── similarity.py            Cosine similarity computation + top-K retrieval
│   ├── user_profile.py          Weighted user preference vector from interaction history
│   └── content_recommender.py  ContentBasedRecommender (extends BaseRecommender)
│
├── collaborative/
│   ├── __init__.py
│   ├── interaction_matrix.py    User-item interaction matrix builder (Polars → scipy sparse)
│   ├── user_cf.py               UserBasedCF (extends BaseRecommender)
│   ├── item_cf.py               ItemBasedCF (extends BaseRecommender)
│   └── matrix_factorization.py  SVDRecommender (extends BaseRecommender)
│
├── features/                    (existing — extend item_features.py)
├── evaluation/                  (existing — unchanged)
├── experiments/                 (existing — unchanged)
├── data/                        (existing — unchanged)
├── baselines/                   (existing — unchanged)
└── config.py                    (extend with Phase 4 hyperparameters)
```

### 2.2 New Config Fields (ml/config.py additions)

```python
# Content-Based Filtering
cbf_category_weight: float = 1.0          # Weight for categorical features
cbf_tfidf_weight: float = 0.5             # Weight for caption TF-IDF features
cbf_normalize: bool = True                # L2-normalize item vectors
cbf_top_k_similar: int = 50              # Pre-compute top-K similar items per item

# Collaborative Filtering
cf_max_watch_ratio: float = 5.0           # Clip watch_ratio before using as interaction signal
cf_n_neighbors: int = 50                  # Neighbors for user/item CF
cf_similarity_min: float = 0.0            # Minimum similarity to include a neighbor
cf_svd_n_components: int = 50            # Latent dimensions for Truncated SVD
cf_svd_random_state: int = 42            # Reproducibility seed
```

### 2.3 BaseRecommender Extension Pattern

All Phase 4 algorithms inherit from the existing `BaseRecommender` ABC with no changes to the ABC itself. The `name` property, `fit()`, and `recommend()` interface are unchanged.

---

## 3. Phase 4A — Content-Based Filtering

### Work Package 4A-WP1: Item Feature Engineering

**Objective:** Build a numerical feature matrix for all 3,327 items using available metadata.

**Available item signals:**
- `first_level_category_name` (20 categories, 100% coverage) → one-hot encode
- `second_level_category_name` (138 values, ~100% coverage) → one-hot encode
- `feat_parsed` (integer feature IDs from `item_categories.parquet`) → multi-hot encode
- `caption` (87.37% non-null) → TF-IDF (treat as optional; items without captions get zero TF-IDF vector)

**File:** `ml/content_based/item_encoder.py`

**Key functions:**
```python
def build_item_feature_matrix(
    item_categories_df: pl.DataFrame,       # from item_categories.parquet
    caption_category_df: pl.DataFrame,      # from kuairec_caption_category.parquet
    cbf_tfidf_weight: float = 0.5,
    cbf_category_weight: float = 1.0,
    normalize: bool = True,
) -> tuple[dict[int, int], list[list[float]]]
# Returns: (video_id → matrix_row_index, dense feature matrix as list-of-lists)
```

**Design decisions:**
- Use Polars for all data manipulation; convert to Python lists for similarity computation
- One-hot encode `first_level_category_name` and `second_level_category_name`
- Multi-hot encode `feat_parsed` IDs (each item has a list of integer feature IDs)
- TF-IDF on `caption` using a pure-Python tf-idf or scikit-learn's `TfidfVectorizer`
- Concatenate all feature groups with configurable weights before L2 normalization
- Items with null caption get zero-filled TF-IDF segment
- Sparse-friendly design: list-of-lists or dict-of-dicts representation

**Tests:** `backend/tests/ml/test_content_based.py`
- Item matrix has correct number of rows (3,327 items)
- Feature vector dimension is deterministic given same inputs
- Null captions produce valid (zero) feature vector
- L2 normalization produces unit vectors
- One-hot encoding is correct for known categories

---

### Work Package 4A-WP2: Cosine Similarity Computation

**Objective:** Compute item-item cosine similarity and support efficient top-K retrieval.

**File:** `ml/content_based/similarity.py`

**Key functions:**
```python
def cosine_similarity_matrix(
    vectors: list[list[float]],
) -> list[list[float]]
# Full NxN similarity matrix (pure Python for small N=3327)

def top_k_similar_items(
    item_idx: int,
    similarity_matrix: list[list[float]],
    idx_to_id: dict[int, int],
    k: int = 50,
    exclude_self: bool = True,
) -> list[int]
# Returns sorted list of k most similar item IDs
```

**Design note:** N=3,327 items → 3,327 × 3,327 = ~11M pairs. A Python list-of-lists for the full similarity matrix uses ~90 MB of RAM (11M × 8 bytes). This is feasible on 16 GB RAM. We pre-compute and store the top-K per item, not the full matrix, to reduce memory.

Alternatively, if scikit-learn is approved, `sklearn.metrics.pairwise.cosine_similarity` handles this in seconds. We will use scikit-learn if D1 is approved.

**Tests:**
- Known 3-vector example produces correct cosine similarity values
- top_k_similar_items returns exactly k items
- Returned items do not include query item (exclude_self)
- Items with zero vectors handled without division error

---

### Work Package 4A-WP3: User Preference Profiles

**Objective:** Represent each user as a weighted average of item feature vectors based on interaction history.

**File:** `ml/content_based/user_profile.py`

**Key functions:**
```python
def build_user_profiles(
    train_df: pl.DataFrame,
    item_feature_matrix: list[list[float]],
    id_to_idx: dict[int, int],
    weight_by_watch_ratio: bool = True,
) -> dict[int, list[float]]
# Returns: {user_id → weighted_average_feature_vector}
```

**Algorithm:**
1. For each user, collect all (video_id, watch_ratio) pairs from training data
2. For each video_id, look up its feature vector from the item matrix
3. Weighted average: `profile = Σ(watch_ratio × feature_vector) / Σ(watch_ratio)`
4. If `weight_by_watch_ratio=False`, use uniform average (each interaction has weight 1)
5. L2-normalize the resulting profile vector

**Data leakage prevention:** Only training data (`train_df`) is used. Val and test splits are never accessed during `fit()`.

**Cold-start:** Users with zero training interactions get a zero profile vector. The recommender falls back to global popularity for these users.

**Tests:**
- User profiles have same dimensionality as item vectors
- Weighted profile gives higher weight to high watch_ratio items
- Zero-profile users (cold start) are handled gracefully
- Train-only data guarantee (val/test not used)

---

### Work Package 4A-WP4: ContentBasedRecommender

**Objective:** Implement the full content-based recommendation pipeline.

**File:** `ml/content_based/content_recommender.py`

```python
class ContentBasedRecommender(BaseRecommender):
    name = "ContentBasedRecommender"
    
    def fit(self, train_df: pl.DataFrame) -> None:
        # 1. Build item feature matrix
        # 2. Build user profiles
        # 3. Pre-compute all recommendations (or compute lazily)
        # 4. Set self._fitted = True
    
    def recommend(
        self, user_id: int, n: int, exclude_seen: set[int]
    ) -> list[int]:
        # 1. If cold-start user: fall back to popularity
        # 2. Compute cosine similarity of user_profile vs all item vectors
        # 3. Sort items by similarity descending
        # 4. Remove exclude_seen items
        # 5. Return top-n item IDs
```

**Parameters (passed at construction, not at fit time):**
- `item_categories_df: pl.DataFrame` — item feature source
- `caption_category_df: pl.DataFrame` — caption/category source
- `fallback: BaseRecommender | None` — cold-start fallback (default: PopularityRecommender)
- `weight_by_watch_ratio: bool = True`
- `tfidf_weight: float = 0.5`
- `random_seed: int = 42`

**Evaluation:** Run against all 5 Phase 3 baselines using the same `Evaluator` and test split.

---

## 4. Phase 4B — Collaborative Filtering

### Work Package 4B-WP1: Interaction Matrix Builder

**Objective:** Build the user-item interaction matrix from training data for all CF algorithms.

**File:** `ml/collaborative/interaction_matrix.py`

```python
def build_interaction_matrix(
    train_df: pl.DataFrame,
    max_watch_ratio: float = 5.0,
) -> tuple[
    list[list[float]],      # Dense matrix [n_users, n_items]
    dict[int, int],         # user_id → row_index
    dict[int, int],         # video_id → col_index
    dict[int, int],         # row_index → user_id
    dict[int, int],         # col_index → video_id
]
```

**Design:**
- Watch ratio is clipped at `max_watch_ratio` to remove outliers (max raw value is 571.5)
- Matrix is dense: 1,411 × 3,327 = ~4.7M cells × 8 bytes = ~38 MB — fits in RAM
- Missing interactions (the ~0.4% of unobserved pairs) are filled with 0.0
- Returns bidirectional index maps for O(1) lookup

**Tests:**
- Matrix dimensions are (n_unique_users × n_unique_items)
- Clipping works correctly (values above max_watch_ratio are capped)
- Index maps are consistent (round-trip user_id → row → user_id)
- Missing interactions produce 0.0 entries

---

### Work Package 4B-WP2: User-Based Collaborative Filtering

**Objective:** Recommend items liked by users similar to the target user.

**File:** `ml/collaborative/user_cf.py`

```python
class UserBasedCF(BaseRecommender):
    name = "UserBasedCF"
    
    def fit(self, train_df: pl.DataFrame) -> None:
        # 1. Build interaction matrix
        # 2. Pre-compute user-user cosine similarity
        # 3. For each user, store top-N neighbors
        # 4. Set self._fitted = True
    
    def recommend(
        self, user_id: int, n: int, exclude_seen: set[int]
    ) -> list[int]:
        # 1. If cold-start: fall back to popularity
        # 2. Find k nearest neighbors
        # 3. Aggregate neighbor ratings: score(item) = Σ sim(user, neighbor) × rating(neighbor, item)
        # 4. Exclude seen items
        # 5. Return top-n by score
```

**Parameters:**
- `n_neighbors: int = 50`
- `fallback: BaseRecommender | None`
- `random_seed: int = 42`

**Cold-start:** Users not in training data → fallback.

---

### Work Package 4B-WP3: Item-Based Collaborative Filtering

**Objective:** Recommend items similar to items the user has interacted with.

**File:** `ml/collaborative/item_cf.py`

```python
class ItemBasedCF(BaseRecommender):
    name = "ItemBasedCF"
    
    def fit(self, train_df: pl.DataFrame) -> None:
        # 1. Build interaction matrix (transposed — items as rows)
        # 2. Pre-compute item-item cosine similarity
        # 3. For each item, store top-N similar items
        # 4. Set self._fitted = True
    
    def recommend(
        self, user_id: int, n: int, exclude_seen: set[int]
    ) -> list[int]:
        # 1. If cold-start: fall back to popularity
        # 2. For each item user has interacted with, get its similar items
        # 3. Score = Σ sim(item, candidate) × user_rating(item)
        # 4. Exclude seen items, return top-n
```

---

### Work Package 4B-WP4: Matrix Factorization (Truncated SVD)

**Objective:** Learn latent user and item factors from the interaction matrix.

**File:** `ml/collaborative/matrix_factorization.py`

```python
class SVDRecommender(BaseRecommender):
    name = "SVDRecommender"
    
    def fit(self, train_df: pl.DataFrame) -> None:
        # 1. Build interaction matrix
        # 2. Apply TruncatedSVD (sklearn) with n_components=50
        # 3. Recover user embeddings: U × Σ
        # 4. Recover item embeddings: V^T
        # 5. Store for dot-product recommendation
        # 6. Set self._fitted = True
    
    def recommend(
        self, user_id: int, n: int, exclude_seen: set[int]
    ) -> list[int]:
        # 1. Look up user embedding
        # 2. Compute dot product with all item embeddings
        # 3. Sort by predicted score descending
        # 4. Exclude seen items, return top-n
```

**Parameters:**
- `n_components: int = 50` — latent dimensions
- `random_state: int = 42`
- `fallback: BaseRecommender | None`

**Justification for Truncated SVD over ALS:**
- No additional package beyond scikit-learn needed
- Dense matrix (99.62%) makes SVD well-conditioned
- Fast on 1,411 × 3,327 matrix (~0.1 seconds)
- Interpretable latent space
- ALS (from `implicit`) is optimized for implicit feedback on sparse matrices — overkill here

---

### Work Package 4B-WP5: Cold-Start Strategy

**Objective:** Define and implement a consistent cold-start fallback mechanism.

**Strategy:**
- All Phase 4 algorithms accept an optional `fallback: BaseRecommender` parameter
- Default fallback is `PopularityRecommender(metric="weighted_score")` — the best Phase 3 baseline
- Cold-start users (not in training data) → fallback immediately in `recommend()`
- Cold-start items (not in feature matrix) → excluded from CBF recommendations silently
- Minimum interaction threshold: 1 (match Phase 3's `min_test_interactions`)

**Documentation:** Cold-start behavior is documented in each recommender's docstring and in `docs/ml/collaborative-filtering.md`.

---

## 5. Evaluation Plan

### 5.1 Evaluation Methodology (unchanged from Phase 3)

- Temporal split: same train/val/test Parquet files
- Relevance threshold: `watch_ratio >= 0.5`
- K values: `[5, 10, 20]`
- Primary metric: **NDCG@10**
- Min test interactions: 1

### 5.2 New Metric: Hit Rate@K

**Add to `ml/evaluation/metrics.py`:**

```python
def hit_rate_at_k(
    recommended: list[int],
    relevant: set[int],
    k: int,
) -> float:
    """Binary: 1 if any of the top-K recommendations is relevant, else 0."""
    return float(any(item in relevant for item in recommended[:k]))
```

This is the simplest meaningful addition — it measures whether a recommendation list has at least one relevant item. Already implied by existing metrics but worth making explicit.

### 5.3 Hyperparameter Tuning on Validation Split

Phase 4 introduces model hyperparameters (k_neighbors, n_components). The validation split (`val.parquet`) is used exclusively for tuning. The test split is reserved for final evaluation.

**Tuning protocol:**
- ContentBasedRecommender: tune `tfidf_weight` ∈ {0.0, 0.25, 0.5, 0.75}
- UserBasedCF: tune `n_neighbors` ∈ {10, 25, 50, 100}
- ItemBasedCF: tune `n_neighbors` ∈ {10, 25, 50, 100}
- SVDRecommender: tune `n_components` ∈ {20, 50, 100}

Best config is selected by validation NDCG@10. Final metrics use test split only.

### 5.4 Comparison Table Format

Generated in `docs/ml/phase4-results.md` after evaluation:

| Algorithm | NDCG@10 | P@10 | R@10 | Coverage@10 | Hit Rate@10 | Notes |
|---|---|---|---|---|---|---|
| RandomRecommender (P3) | 0.3236 | 0.3233 | 0.0095 | 0.6674 | — | Floor |
| PopularityRecommender (P3) | 0.4461 | 0.3703 | 0.0109 | 0.1265 | — | Best P3 baseline |
| CategoryPopularityRecommender (P3) | 0.4096 | 0.3402 | 0.0100 | 0.1532 | — | |
| TrendingRecommender (P3) | 0.3096 | 0.2694 | 0.0080 | 0.1151 | — | |
| UserPreferenceRecommender (P3) | 0.3396 | 0.3516 | 0.0103 | 0.4824 | — | |
| ContentBasedRecommender (P4A) | TBD | TBD | TBD | TBD | TBD | |
| UserBasedCF (P4B) | TBD | TBD | TBD | TBD | TBD | |
| ItemBasedCF (P4B) | TBD | TBD | TBD | TBD | TBD | |
| SVDRecommender (P4B) | TBD | TBD | TBD | TBD | TBD | |

---

## 6. API Integration Design

### 6.1 New Endpoints

Add to `backend/app/api/v1/endpoints/recommendations.py`:

**`GET /api/v1/recommendations/user/{user_id}`**
- Returns personalized top-N recommendations for a user
- Request params: `user_id: int` (path), `n: int = 10` (query), `algorithm: str = "svd"` (query)
- Response: `RecommendationResponse`
- Error: 404 if user_id not found, 400 if algorithm is invalid, 503 if model not loaded

**`GET /api/v1/recommendations/similar/{video_id}`**
- Returns items similar to a given item
- Request params: `video_id: int` (path), `n: int = 10` (query)
- Response: `SimilarItemsResponse`
- Error: 404 if video_id not found, 503 if model not loaded

**`GET /api/v1/recommendations/models`**
- Returns list of loaded models and their metadata
- Response: `ModelsInfoResponse`

**`GET /api/v1/recommendations/results`**
- Returns the latest evaluation results summary
- Response: `EvaluationResultsResponse`

### 6.2 Response Schemas

```python
class RecommendationItem(BaseModel):
    video_id: int
    score: float
    rank: int

class RecommendationResponse(BaseModel):
    user_id: int
    algorithm: str
    recommendations: list[RecommendationItem]
    n_requested: int
    is_cold_start: bool

class SimilarItemsResponse(BaseModel):
    video_id: int
    similar_items: list[RecommendationItem]

class ModelInfo(BaseModel):
    name: str
    algorithm: str
    fitted: bool
    n_train_interactions: int | None

class ModelsInfoResponse(BaseModel):
    models: list[ModelInfo]
    phase: str

class EvaluationResultsResponse(BaseModel):
    phase: str
    algorithms: list[str]
    primary_metric: str
    results_summary: dict[str, float]   # algorithm_name → ndcg@10
```

### 6.3 Model Loading Strategy

Models are loaded at application startup (FastAPI `lifespan`). They are trained on the full training split during startup if no serialized model exists. This keeps the API layer simple — no on-demand training.

**Files:**
```
ml/serving/
├── __init__.py
└── model_registry.py   # Loads and caches fitted recommenders; lazy singleton pattern
```

### 6.4 Existing Router Extension

Add recommendations router to `backend/app/api/v1/router.py`:
```python
router.include_router(recommendations_router, tags=["recommendations"])
```

---

## 7. Dependency Plan

### 7.1 New Package: scikit-learn

Add to `backend/pyproject.toml` as a new optional dependency group `[ml]`:

```toml
[project.optional-dependencies]
ml = [
    "scikit-learn>=1.5.0",
]
```

Install with: `pip install -e ".[ml]"`

**Justification:** Required for `TfidfVectorizer` (CBF captions) and `TruncatedSVD` (matrix factorization). Both are lightweight; scikit-learn has no GPU requirement and works on Python 3.14.

### 7.2 No Other New Packages

| Package | Decision | Reason |
|---|---|---|
| `implicit` | ❌ Skip | Dense matrix; scikit-learn SVD sufficient |
| `numpy` | ⚠️ Implicit via scikit-learn | Do not import directly; let scikit-learn manage |
| `scipy` | ⚠️ Implicit via scikit-learn | Same; do not import directly |
| `MLflow` | ❌ Defer to Phase 5 | ADR-P3-06 decision |
| `faiss` | ❌ Defer to Phase 5 | Similarity at N=3,327 is fast enough without ANN |

---

## 8. Testing Strategy

### 8.1 New Test Files

**`backend/tests/ml/test_content_based.py`**
- `TestItemEncoder`: feature matrix shape, null caption handling, normalization
- `TestCosineSimilarity`: known values, self-similarity = 1.0, top_k count
- `TestUserProfile`: weighted average, cold-start zero vector, train-data-only guarantee
- `TestContentBasedRecommender`: fit + recommend, exclude seen, cold-start fallback, n items returned

**`backend/tests/ml/test_collaborative.py`**
- `TestInteractionMatrix`: shape, clipping, index maps round-trip
- `TestUserBasedCF`: fit + recommend, cold-start, n_neighbors effect, exclude seen
- `TestItemBasedCF`: fit + recommend, cold-start, exclude seen
- `TestSVDRecommender`: fit + recommend, dimensionality, cold-start, determinism with same seed

**`backend/tests/api/test_recommendations.py`**
- Happy path: valid user_id returns recommendations
- Invalid algorithm name returns 400
- Unknown user_id returns 404 or cold-start fallback
- `/models` endpoint returns list
- `/results` endpoint returns evaluation summary

### 8.2 Regression Tests

All 134 existing tests must continue passing. No existing files in `ml/baselines/`, `ml/evaluation/`, or `ml/data/` are modified.

---

## 9. File and Directory Changes

### New Files
```
ml/content_based/__init__.py
ml/content_based/item_encoder.py
ml/content_based/similarity.py
ml/content_based/user_profile.py
ml/content_based/content_recommender.py

ml/collaborative/__init__.py
ml/collaborative/interaction_matrix.py
ml/collaborative/user_cf.py
ml/collaborative/item_cf.py
ml/collaborative/matrix_factorization.py

ml/serving/__init__.py
ml/serving/model_registry.py

scripts/ml/run_content_based.py
scripts/ml/run_collaborative.py

backend/app/api/v1/endpoints/recommendations.py

backend/tests/ml/test_content_based.py
backend/tests/ml/test_collaborative.py
backend/tests/api/__init__.py
backend/tests/api/test_recommendations.py

docs/ml/content-based-filtering.md
docs/ml/collaborative-filtering.md
docs/ml/phase4-results.md         (generated after evaluation)
docs/phases/phase-4-ml-guide.md
```

### Modified Files
```
ml/config.py                      Add Phase 4 hyperparameter fields
ml/evaluation/metrics.py          Add hit_rate_at_k() function
backend/app/api/v1/router.py      Add recommendations router
backend/app/main.py               Add model loading to lifespan
backend/pyproject.toml            Add [ml] optional dependency group
AGENTS.md                         Update Phase 4 → Complete after implementation
README.md                         Update phase status table
docs/decisions/architecture-decisions.md   Add ADR-P4-01 through ADR-P4-05
phase-4-plan.md                   This file — update status after each WP
```

### Unchanged Files (do NOT touch)
```
ml/baselines/           All existing baselines
ml/evaluation/evaluator.py
ml/data/loader.py
ml/data/splits.py
ml/features/            Existing feature files
ml/experiments/recorder.py
backend/tests/ml/test_baselines_*.py
backend/tests/ml/test_metrics.py
backend/tests/ml/test_splits.py
backend/tests/test_health.py
backend/tests/test_system.py
```

---

## 10. Architecture Decision Records (Phase 4)

### ADR-P4-01 — Add scikit-learn as Optional [ml] Dependency
- **Status:** Proposed
- **Decision:** Add `scikit-learn>=1.5.0` to `pyproject.toml` under `[ml]` extras
- **Rationale:** Needed for TfidfVectorizer (CBF) and TruncatedSVD (MF); GPU-free; well-maintained

### ADR-P4-02 — Use Truncated SVD for Matrix Factorization
- **Status:** Proposed
- **Decision:** Use `sklearn.decomposition.TruncatedSVD` with `n_components=50`
- **Rationale:** Dense matrix makes SVD well-conditioned; scikit-learn sufficient; no need for `implicit` package

### ADR-P4-03 — Use Continuous Watch Ratio (Clipped) as CF Signal
- **Status:** Proposed
- **Decision:** Use `min(watch_ratio, 5.0)` as interaction signal in the user-item matrix
- **Rationale:** Richer than binary; outliers (max 571.5) would distort SVD; median 0.77 makes clip at 5.0 safe

### ADR-P4-04 — Validation Split for Hyperparameter Tuning
- **Status:** Proposed
- **Decision:** Use `val.parquet` for hyperparameter selection (n_neighbors, n_components, tfidf_weight); test split is reserved for final evaluation
- **Rationale:** Prevents optimistic test-set bias; consistent with industry standard

### ADR-P4-05 — Model Serving via In-Process Model Registry
- **Status:** Proposed
- **Decision:** Models are loaded at FastAPI startup into a singleton registry; no external model server
- **Rationale:** Phase 4 scope is local development; Phase 5 can introduce proper model serving

---

## 11. Implementation Sequence

The 11 work packages are ordered to minimize blocked dependencies:

```
WP1  Install scikit-learn, update pyproject.toml, verify import
WP2  Add hit_rate_at_k to ml/evaluation/metrics.py + test
WP3  Phase 4A: Item encoder (item_encoder.py) + test
WP4  Phase 4A: Similarity computation (similarity.py) + test
WP5  Phase 4A: User profiles (user_profile.py) + test
WP6  Phase 4A: ContentBasedRecommender + evaluation vs P3 baselines
WP7  Phase 4B: Interaction matrix builder (interaction_matrix.py) + test
WP8  Phase 4B: UserBasedCF + ItemBasedCF + evaluation
WP9  Phase 4B: SVDRecommender + evaluation + hyperparameter tuning on val split
WP10 API integration: recommendations endpoints + model registry + API tests
WP11 Documentation + comparison report + AGENTS.md + README + ADRs + commit
```

Each WP runs quality gates (pytest + ruff) before proceeding to the next.

---

## 12. Acceptance Criteria

| # | Criterion | Measurement |
|---|---|---|
| 1 | scikit-learn installable on Python 3.14, Windows | `import sklearn; print(sklearn.__version__)` |
| 2 | ContentBasedRecommender fits and recommends | 134+ tests pass |
| 3 | User profiles use train data only | Dedicated leakage prevention test |
| 4 | Cosine similarity produces correct values | Known-input unit test |
| 5 | UserBasedCF fits and recommends | Tests pass |
| 6 | ItemBasedCF fits and recommends | Tests pass |
| 7 | SVDRecommender fits and recommends | Tests pass |
| 8 | Cold-start returns fallback, not error | Explicit cold-start test for each algorithm |
| 9 | All 4 Phase 4 algorithms evaluated on test split | Experiment JSON files in experiments/results/ |
| 10 | Phase 4 results compared against all 5 P3 baselines | Comparison table in docs/ml/phase4-results.md |
| 11 | hit_rate_at_k implemented and tested | Passes test_metrics.py |
| 12 | API endpoints functional | 3 recommendation endpoints respond correctly |
| 13 | All 134 existing tests continue passing | `pytest backend/tests/ -q` → 134+ passed, 0 failed |
| 14 | Ruff clean | `ruff check ...` → All checks passed |
| 15 | No data files committed to Git | `git ls-files data/` → only .gitkeep files |
| 16 | Commit pushed to main only after owner approval | Explicit confirmation before push |

---

## 13. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| scikit-learn incompatible with Python 3.14 | Medium | High | Verify in WP1 immediately; fallback: implement pure-Python SVD if needed |
| Memory overflow during similarity matrix computation (N=3,327 × N=3,327) | Low | Medium | Pre-compute only top-K per item; never store full matrix |
| SVD diverges or produces NaN on clipped watch_ratio matrix | Low | Medium | Validate matrix statistics before SVD; use `check_input=False` with pre-validated data |
| Dense matrix CBF similarity dominated by category one-hot (many dims) | Medium | Low | Normalize all feature groups before concatenation; tune `tfidf_weight` |
| Evaluation runtime too slow (1,411 users × 3,327 items × 4 algorithms) | Medium | Low | Pre-compute recommendations in batch during evaluation; reuse seen-item sets |
| Phase 3 baselines not beaten by any Phase 4 algorithm | Medium | Low | Expected — dense matrix makes popularity hard to beat; document and analyze, do not hide |
| API model loading adds significant startup time | Low | Low | Load models lazily on first request, not at startup if too slow |

---

*Phase 4 plan prepared: 2026-11-17*
*Status: AWAITING OWNER APPROVAL*
*Owner: Approval required before any implementation begins*
