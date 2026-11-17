"""Content-Based Filtering recommender — Phase 4A.

Recommends items based on similarity between a user's preference profile
and item feature vectors built from KuaiRec metadata.

Pipeline:
  fit()       → builds item feature matrix + user profiles from train_df
  recommend() → scores all items vs user profile, returns top-K

Cold-start:
  Users not in training data receive recommendations from the fallback
  recommender (default: PopularityRecommender with weighted_score).
  Items not in the feature matrix are excluded from recommendations.

Usage:
    from ml.content_based.content_recommender import ContentBasedRecommender
    from ml.data.loader import load_item_categories, load_caption_category

    rec = ContentBasedRecommender(
        item_categories_df=load_item_categories(),
        caption_category_df=load_caption_category(),
    )
    rec.fit(train_df)
    recs = rec.recommend(user_id=42, n=10, exclude_seen={100, 101})
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl

from ml.baselines.base import BaseRecommender
from ml.baselines.popularity_recommender import PopularityRecommender
from ml.content_based.item_encoder import build_item_feature_matrix
from ml.content_based.similarity import build_score_matrix, top_k_similar_items
from ml.content_based.user_profile import build_user_profiles


class ContentBasedRecommender(BaseRecommender):
    """Content-Based Filtering recommender using item metadata features.

    Builds item feature vectors from:
      - One-hot encoded category hierarchy (first & second level)
      - Multi-hot encoded feat_parsed IDs
      - TF-IDF caption embeddings (optional)

    User profiles are weighted averages of item vectors for items the
    user interacted with in training, weighted by watch_ratio.

    Args:
        item_categories_df:   DataFrame with video_id, feat_parsed columns.
        caption_category_df:  DataFrame with video_id, caption,
                              first_level_category_name,
                              second_level_category_name.
        tfidf_weight:         Weight for TF-IDF caption features (0.0 disables).
        category_weight:      Weight for categorical one-hot/multi-hot features.
        tfidf_max_features:   Maximum TF-IDF vocabulary size.
        weight_by_watch_ratio: If True, build profiles weighted by watch_ratio.
        fallback:             Recommender for cold-start users. Defaults to
                              PopularityRecommender with weighted_score.
        random_seed:          Passed to BaseRecommender.
    """

    def __init__(
        self,
        item_categories_df: pl.DataFrame,
        caption_category_df: pl.DataFrame,
        tfidf_weight: float = 0.5,
        category_weight: float = 1.0,
        tfidf_max_features: int = 500,
        weight_by_watch_ratio: bool = True,
        fallback: BaseRecommender | None = None,
        random_seed: int = 42,
    ) -> None:
        super().__init__(random_seed=random_seed)
        self._item_categories_df = item_categories_df
        self._caption_category_df = caption_category_df
        self._tfidf_weight = tfidf_weight
        self._category_weight = category_weight
        self._tfidf_max_features = tfidf_max_features
        self._weight_by_watch_ratio = weight_by_watch_ratio
        self._fallback: BaseRecommender = fallback or PopularityRecommender(
            popularity_metric="weighted_score"
        )

        # Populated during fit()
        self._id_to_idx: dict[int, int] = {}
        self._idx_to_id: dict[int, int] = {}
        self._item_matrix: list[list[float]] = []
        self._user_profiles: dict[int, list[float]] = {}
        self._score_matrix: np.ndarray | None = None
        self._user_id_to_row: dict[int, int] = {}
        self._fit_seconds: float = 0.0

    @property
    def name(self) -> str:
        return (
            f"ContentBasedRecommender(tfidf={self._tfidf_weight},"
            f"cat={self._category_weight})"
        )

    def fit(self, train_df: pl.DataFrame) -> None:
        """Build item feature matrix and user profiles from training data.

        Args:
            train_df: Training interactions (user_id, video_id, watch_ratio).
        """
        t0 = time.time()

        # 1. Build item feature matrix
        self._id_to_idx, self._idx_to_id, self._item_matrix = build_item_feature_matrix(
            item_categories_df=self._item_categories_df,
            caption_category_df=self._caption_category_df,
            category_weight=self._category_weight,
            tfidf_weight=self._tfidf_weight,
            normalize=True,
            tfidf_max_features=self._tfidf_max_features,
        )

        # 2. Build user profiles from training data only
        self._user_profiles = build_user_profiles(
            train_df=train_df,
            item_feature_matrix=self._item_matrix,
            id_to_idx=self._id_to_idx,
            weight_by_watch_ratio=self._weight_by_watch_ratio,
            normalize=True,
        )

        # 3. Pre-compute score matrix: (n_users, n_items) via numpy matmul
        user_ids = sorted(self._user_profiles.keys())
        self._user_id_to_row = {uid: i for i, uid in enumerate(user_ids)}
        self._score_matrix = build_score_matrix(
            user_profiles=self._user_profiles,
            item_matrix=self._item_matrix,
            user_id_order=user_ids,
        )

        # 4. Fit fallback on same training data
        self._fallback.fit(train_df)

        self._fit_seconds = time.time() - t0
        self._fitted = True

    def recommend(
        self,
        user_id: int,
        n: int,
        exclude_seen: set[int],
    ) -> list[int]:
        """Return top-N content-based recommendations for a user.

        Uses the pre-computed score matrix when available (fast path), otherwise
        scores on-the-fly via top_k_similar_items.

        Args:
            user_id:      Target user.
            n:            Number of recommendations to return.
            exclude_seen: Set of video_ids the user has already seen.

        Returns:
            Ranked list of up to n video_ids (best match first).
            Falls back to popularity for cold-start users (zero profile).
        """
        self._check_fitted()

        profile = self._user_profiles.get(user_id)

        # Cold-start: user not in training data
        if profile is None:
            return self._fallback.recommend(user_id, n, exclude_seen)

        # Cold-start: user has zero profile
        if all(x == 0.0 for x in profile):
            return self._fallback.recommend(user_id, n, exclude_seen)

        # Fast path: use pre-computed score row if available
        if self._score_matrix is not None and user_id in self._user_id_to_row:
            row_idx = self._user_id_to_row[user_id]
            scores_row = self._score_matrix[row_idx]  # numpy 1D array (n_items,)
            # Get indices sorted by score descending
            sorted_indices = np.argsort(scores_row)[::-1]
            result = []
            for idx in sorted_indices:
                vid = self._idx_to_id[int(idx)]
                if vid not in exclude_seen:
                    result.append(vid)
                    if len(result) >= n:
                        break
            return result

        # Slow path fallback (used when score matrix not pre-computed)
        return top_k_similar_items(
            query_vector=profile,
            item_matrix=self._item_matrix,
            idx_to_id=self._idx_to_id,
            k=n,
            exclude_ids=exclude_seen,
        )

    @property
    def fit_seconds(self) -> float:
        """Wall-clock time taken by the last fit() call."""
        return self._fit_seconds
