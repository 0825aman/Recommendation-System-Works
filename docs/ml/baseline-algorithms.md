# Baseline Algorithms

**Phase:** 3 — Baseline Recommendation Systems  
**ADR:** ADR-P3-01  
**Last updated:** Phase 3

---

## Overview

Phase 3 implements five non-ML heuristic baseline recommenders. These algorithms do not
learn from data in the statistical ML sense — they use hand-crafted scoring rules derived
directly from interaction counts, timestamps, and category affinity.

Their purpose is to establish a **performance floor**. Any ML algorithm introduced in
Phase 4 must beat these baselines on NDCG@10 to justify its added complexity.

All five algorithms inherit from [`ml/baselines/base.py`](../../ml/baselines/base.py) and
implement the same interface:

```python
class BaseRecommender:
    def fit(self, interactions: pl.DataFrame) -> None: ...
    def recommend(self, user_id: int, n: int, exclude_seen: set[int]) -> list[int]: ...
```

---

## Algorithm 1 — RandomRecommender

**File:** [`ml/baselines/random_recommender.py`](../../ml/baselines/random_recommender.py)

### Logic

Recommends `n` items chosen uniformly at random from the full item catalogue, excluding
any items the user has already seen.

```python
RandomRecommender(random_seed=42)
```

### Design rationale

Random recommendations form the **absolute floor** of the evaluation. Any algorithm
scoring below Random on NDCG@10 is performing worse than chance and is almost certainly
buggy. Random is deterministic given a seed, making test assertions reliable.

### Phase 3 results (K=10)

| NDCG@10 | P@10 | R@10 | Coverage@10 |
|---|---|---|---|
| 0.3236 | 0.3233 | 0.0095 | 0.6674 |

High coverage (66.7%) confirms the random distribution is spreading across the catalogue.

---

## Algorithm 2 — PopularityRecommender

**File:** [`ml/baselines/popularity_recommender.py`](../../ml/baselines/popularity_recommender.py)

### Logic

Scores items by a **weighted popularity score** (ADR-P3-05):

```
score(item) = mean(watch_ratio) × log(1 + interaction_count)
```

Items are ranked globally by this score. Every user receives the same ranking; only
`exclude_seen` personalises the output.

```python
PopularityRecommender(metric="weighted_score")  # default
PopularityRecommender(metric="interaction_count")  # raw count fallback
```

### Design rationale

The weighted score balances two signals:
- `mean(watch_ratio)` — quality: a niche item watched completely should rank higher than
  a popular item abandoned immediately.
- `log(1 + count)` — reach: items seen by more users are more likely to match new users.

Using `log` dampens the effect of blockbuster items and avoids extreme popularity bias.

### Phase 3 results (K=10)

| NDCG@10 | P@10 | R@10 | Coverage@10 |
|---|---|---|---|
| **0.4461** | 0.3703 | 0.0109 | 0.1265 |

**Best NDCG@10 of all Phase 3 baselines.** Low coverage (12.6%) confirms popularity
bias — the same popular items dominate all recommendation lists.

---

## Algorithm 3 — CategoryPopularityRecommender

**File:** [`ml/baselines/category_popularity_recommender.py`](../../ml/baselines/category_popularity_recommender.py)

### Logic

For each user, identifies the category they have interacted with most (by interaction
count). Candidates from that category are ranked by global weighted popularity score.
If no category match exists (cold user) or no item map is provided, falls back to
global PopularityRecommender ranking.

```python
CategoryPopularityRecommender()
```

Requires `item_to_category` dict passed at `fit()` time, loaded via
[`ml/features/item_features.py`](../../ml/features/item_features.py).

### Design rationale

A lightweight form of personalisation — users who consistently watch sports content
should receive sports recommendations ahead of unrelated popular content. Category affinity
requires only interaction counts, not any learned representation.

### Phase 3 results (K=10)

| NDCG@10 | P@10 | R@10 | Coverage@10 |
|---|---|---|---|
| 0.4096 | 0.3402 | 0.0100 | 0.1532 |

Slightly lower NDCG@10 than global popularity (0.4096 vs 0.4461), slightly higher coverage
(15.3% vs 12.6%). The category constraint occasionally surfaces less popular items that
happen to match a user's affinity, trading ranking precision for diversity.

---

## Algorithm 4 — TrendingRecommender

**File:** [`ml/baselines/trending_recommender.py`](../../ml/baselines/trending_recommender.py)

### Logic

Restricts the popularity calculation to interactions within a **recency window** (default
14 days) ending at the most recent timestamp in the training data. Items trending in this
window are ranked by their window-scoped weighted popularity score. Falls back to global
popularity when the window produces fewer than 10 items.

```python
TrendingRecommender(window_days=14)
```

### Design rationale

Global popularity is static — an item popular 2 months ago may no longer be relevant.
Trending captures **temporal momentum**: items gaining interaction velocity recently may
be more likely to appeal to current users. The 14-day window matches the KuaiRec 2.0
collection period (~63 days) at roughly 22% recency focus.

### Phase 3 results (K=10)

| NDCG@10 | P@10 | R@10 | Coverage@10 |
|---|---|---|---|
| 0.3096 | 0.2694 | 0.0080 | 0.1151 |

Lowest-scoring non-random baseline. The 14-day window in a 63-day dataset is quite
narrow — few items accumulated strong trending scores, reducing effective ranking
signal. A wider window (28 days) may perform better; this is a hyperparameter to
explore in Phase 4 comparison experiments.

---

## Algorithm 5 — UserPreferenceRecommender

**File:** [`ml/baselines/user_preference_recommender.py`](../../ml/baselines/user_preference_recommender.py)

### Logic

For each user, computes a **category affinity vector** — the weighted average watch_ratio
per category. Items are scored by looking up each item's category and using the user's
affinity score for that category as the base ranking signal, then breaking ties by global
popularity. Cold users (not in training data) fall back to global popularity.

```python
UserPreferenceRecommender()
```

Requires item→category map and uses [`ml/features/user_features.py`](../../ml/features/user_features.py)
to compute per-user category affinity.

### Design rationale

This is the most personalised of the five baselines. Rather than "which category did
this user interact with most?", it asks "how much did this user enjoy each category?"
(using watch_ratio quality, not raw count). This surfaces niche high-quality categories
a user watches deeply but infrequently.

### Phase 3 results (K=10)

| NDCG@10 | P@10 | R@10 | Coverage@10 |
|---|---|---|---|
| 0.3396 | 0.3516 | 0.0103 | 0.4824 |

Good coverage (48.2%) reflects personalisation spreading recommendations across the
catalogue. NDCG@10 is below global popularity because the category affinity signal
is noisy at this granularity — many items map to the same broad category, so
the within-category ranking still defaults to global popularity.

---

## Comparative Summary (K=10)

| Algorithm | NDCG@10 | P@10 | R@10 | Coverage@10 | Notes |
|---|---|---|---|---|---|
| RandomRecommender | 0.3236 | 0.3233 | 0.0095 | 0.6674 | Floor baseline |
| **PopularityRecommender** | **0.4461** | **0.3703** | **0.0109** | 0.1265 | **Best NDCG** |
| CategoryPopularityRecommender | 0.4096 | 0.3402 | 0.0100 | 0.1532 | Personalised variant |
| TrendingRecommender | 0.3096 | 0.2694 | 0.0080 | 0.1151 | Recency-focused |
| UserPreferenceRecommender | 0.3396 | 0.3516 | 0.0103 | 0.4824 | Best coverage |

**Key takeaway:** Global popularity is a surprisingly strong baseline on the dense
small_matrix — it dominates NDCG@10 because nearly everyone watches the most popular
items (99.6% density). Phase 4 algorithms on the sparser big_matrix will show a wider
performance gap between popularity and learned models.

---

## Common Interface

All baselines share the same contract defined in `BaseRecommender`:

```python
# Fit on training interactions
rec.fit(train_df)

# Get top-N recommendations, excluding already-seen items
recs: list[int] = rec.recommend(user_id=42, n=10, exclude_seen={100, 101})

# Access algorithm name for experiment logging
name: str = rec.name
```

The `fit()` method accepts a Polars DataFrame with at minimum columns:
`user_id`, `video_id`, `watch_ratio`.

---

## Adding a New Baseline

1. Create `ml/baselines/your_algorithm.py`.
2. Subclass `BaseRecommender` and implement `fit()` and `recommend()`.
3. Add tests in `backend/tests/ml/test_baselines_advanced.py`.
4. Register in `scripts/ml/run_baselines.py` for automatic benchmarking.
5. Run `backend\.venv\Scripts\pytest.exe backend/tests/ -v` and
   `backend\.venv\Scripts\ruff.exe check ml/` before committing.
