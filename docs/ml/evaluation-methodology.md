# Evaluation Methodology

**Phase:** 3 — Baseline Recommendation Systems  
**ADRs:** ADR-P3-02, ADR-P3-04, ADR-P3-05  
**Last updated:** Phase 3

---

## Overview

This document describes the offline evaluation methodology used in Phase 3 to benchmark
baseline recommendation algorithms on the KuaiRec 2.0 small_matrix dataset.

Offline evaluation answers the question: *given only historical interaction data, how well
does an algorithm rank items that users actually watched?*

---

## Dataset

| Property | Value |
|---|---|
| Source | KuaiRec 2.0 — `small_matrix` |
| Users | 1,411 |
| Items | 3,327 |
| Interactions | 4,676,570 |
| Matrix density | 99.6% |
| Temporal range | 2020-07-05 to 2020-09-05 |
| Null-timestamp rows | 3.89% of interactions |

The small_matrix is a **fully observed dense matrix** — almost every user has rated almost
every item. This produces higher baseline Precision@K values than sparse industrial datasets.
When big_matrix is introduced in Phase 4, metrics will drop significantly and reflect a
more realistic sparse recommendation scenario.

---

## Target Label — Binary Relevance (ADR-P3-02)

An interaction is considered **relevant** when:

```
watch_ratio >= 0.5
```

A user watching at least 50% of a video is taken as a signal of genuine interest.

| Statistic | Value |
|---|---|
| Relevant interactions (≥ 0.5) | 72.73% |
| Non-relevant interactions (< 0.5) | 27.27% |

This threshold was chosen because:
- It is a natural midpoint in [0, 1] watch behaviour.
- Watching half a video implies sustained engagement, not an accidental click.
- It maps cleanly to binary relevance required by standard ranking metrics.

---

## Train / Validation / Test Split (ADR-P3-04)

A **temporal split** is used to simulate real-world deployment conditions: a model trained
on past interactions is evaluated on future interactions.

| Split | Proportion | Rows |
|---|---|---|
| Train | 70% | 3,328,196 |
| Validation | 15% | 674,186 |
| Test | 15% | 674,188 |

**Strategy:**
1. Sort all timed interactions by `timestamp` (Float64 unix seconds).
2. Assign the bottom 70% to train, the next 15% to validation, the top 15% to test.
3. Null-timestamp rows (3.89%) cannot be ordered — they are assigned to train.

**Data leakage guarantee:** No test-split timestamp is less than or equal to any
train-split timestamp (verified by `test_no_data_leakage_train_to_test`).

Split files are written to `data/processed/kuairec/splits/` as Parquet and are excluded
from Git.

---

## Ranking Metrics

All metrics are computed per-user and averaged across users who have at least one relevant
item in the test split (`min_test_interactions = 1`).

### Precision@K

Fraction of the top-K recommended items that are relevant.

```
Precision@K = |relevant ∩ top-K| / K
```

### Recall@K

Fraction of all relevant items for a user that appear in the top-K recommendations.

```
Recall@K = |relevant ∩ top-K| / |relevant|
```

### NDCG@K — Normalized Discounted Cumulative Gain

Measures ranking quality: a relevant item at rank 1 is worth more than at rank K.

```
DCG@K  = Σ rel_i / log2(i + 1)   for i = 1..K
IDCG@K = DCG of the ideal (perfect) ranking
NDCG@K = DCG@K / IDCG@K
```

NDCG@K is the **primary metric** for all Phase 3 comparisons because it rewards
correct ordering, not just set coverage.

### MAP@K — Mean Average Precision

Average of Precision@k at each position k where a relevant item appears, averaged over
all users. Penalises algorithms that rank relevant items late.

```
AP@K = (1 / |relevant|) × Σ Precision@k × rel(k)   for k = 1..K
```

---

## Beyond-Accuracy Metrics

### Catalog Coverage@K

The fraction of the full item catalogue that appears in at least one recommendation list,
across all users, at cutoff K.

```
Coverage@K = |∪ recommended_lists| / |catalogue|
```

Coverage measures **breadth** — an algorithm recommending the same 100 items to every
user has near-zero coverage even at high Precision@K.

### Intra-List Diversity@K

The fraction of unique categories in a single recommendation list.

```
Diversity@K = |unique categories in list| / K
```

Averaged over all users. Requires item→category mapping from `item_categories.parquet`.

### Novelty@K

The average self-information of recommended items — rare items score higher than popular ones.

```
Novelty@K = (1/|users|) × Σ_u (1/K) × Σ_i -log2(p(i))
p(i) = interaction_count(i) / total_interactions
```

---

## Evaluation Cutoff Values

Metrics are computed at K ∈ {5, 10, 20}.

The primary reporting cutoff is **K = 10** (consistent with industry practice and
Netflix / YouTube literature).

---

## Sanity Check

After every evaluation run, the following assertion is enforced:

```
NDCG@10(PopularityRecommender) >= NDCG@10(RandomRecommender)
```

If this check fails it indicates a bug in the evaluation pipeline or data preparation,
not in the algorithms. This check passed for Phase 3: 0.4461 ≥ 0.3236.

---

## Implementation

| File | Responsibility |
|---|---|
| [`ml/evaluation/metrics.py`](../../ml/evaluation/metrics.py) | 7 pure-Python metric functions |
| [`ml/evaluation/evaluator.py`](../../ml/evaluation/evaluator.py) | `Evaluator` class — per-user loop, aggregation |
| [`ml/data/splits.py`](../../ml/data/splits.py) | Temporal split logic and manifest writer |
| [`scripts/ml/prepare_splits.py`](../../scripts/ml/prepare_splits.py) | CLI runner for split creation |

No NumPy or external ML libraries are used in the evaluation layer. All metrics are
implemented in pure Python to minimise dependency risk and simplify debugging.

---

## Limitations

1. **Dense matrix bias:** The 99.6% dense small_matrix inflates Recall@K values relative
   to realistic sparse datasets. Phase 4 big_matrix evaluation will address this.
2. **Offline-online gap:** Offline NDCG does not guarantee online click-through or
   engagement improvements. A/B testing (Phase 5) will bridge this gap.
3. **No temporal decay:** Older interactions are weighted identically to recent ones in
   the per-user relevance sets. Future phases may introduce time-weighted evaluation.
4. **Single threshold:** Binary relevance at 0.5 discards signal from partially-watched
   videos. Graded relevance (e.g., watch_ratio as a score) is deferred to Phase 4.
