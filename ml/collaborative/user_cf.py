"""User-Based Collaborative Filtering recommender — Phase 4B.

Finds users similar to the target user based on their interaction patterns,
then recommends items those similar users liked that the target hasn't seen.

Similarity is computed as cosine similarity between interaction row vectors
(user × item watch_ratio matrix, clipped). Computed once at fit() time and
cached for fast recommendation.

Usage:
    from ml.collaborative.user_cf import UserBasedCF

    rec = UserBasedCF(n_neighbors=50)
    rec.fit(train_df)
    recs = rec.recommend(user_id=42, n=10, exclude_seen={100, 101})
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl

from ml.baselines.base import BaseRecommender
from ml.baselines.popularity_recommender import PopularityRecommender
from ml.collaborative.interaction_matrix import build_interaction_matrix


class UserBasedCF(BaseRecommender):
    """User-Based Collaborative Filtering via cosine-similarity neighbours.

    For each target user:
      1. Retrieve the k most similar users (cosine similarity on row vectors).
      2. Score items as the similarity-weighted sum of neighbour ratings.
      3. Exclude seen items and return top-N.

    Args:
        n_neighbors:     Number of similar users to consider.
        max_watch_ratio: Clip watch_ratio before building interaction matrix.
        fallback:        Recommender for cold-start users (not in training).
        random_seed:     Passed to BaseRecommender.
    """

    def __init__(
        self,
        n_neighbors: int = 50,
        max_watch_ratio: float = 5.0,
        fallback: BaseRecommender | None = None,
        random_seed: int = 42,
    ) -> None:
        super().__init__(random_seed=random_seed)
        self._n_neighbors = n_neighbors
        self._max_watch_ratio = max_watch_ratio
        self._fallback: BaseRecommender = fallback or PopularityRecommender(
            popularity_metric="weighted_score"
        )

        # Populated during fit()
        self._matrix: np.ndarray | None = None
        self._user_to_row: dict[int, int] = {}
        self._item_to_col: dict[int, int] = {}
        self._row_to_user: dict[int, int] = {}
        self._col_to_item: dict[int, int] = {}
        self._user_sim: np.ndarray | None = None   # (n_users, n_users) cosine sim
        self._fit_seconds: float = 0.0

    @property
    def name(self) -> str:
        return f"UserBasedCF(k={self._n_neighbors})"

    def fit(self, train_df: pl.DataFrame) -> None:
        """Build interaction matrix and pre-compute user-user similarities.

        Args:
            train_df: Training interactions (user_id, video_id, watch_ratio).
        """
        t0 = time.time()

        (
            self._matrix,
            self._user_to_row,
            self._item_to_col,
            self._row_to_user,
            self._col_to_item,
        ) = build_interaction_matrix(train_df, max_watch_ratio=self._max_watch_ratio)

        # Normalise user rows for cosine similarity: (n_users, n_items)
        norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid division by zero for cold-start rows
        normalised = self._matrix / norms

        # User-user cosine similarity: (n_users, n_users)
        self._user_sim = normalised @ normalised.T

        # Fit fallback
        self._fallback.fit(train_df)

        self._fit_seconds = time.time() - t0
        self._fitted = True

    def recommend(
        self,
        user_id: int,
        n: int,
        exclude_seen: set[int],
    ) -> list[int]:
        """Return top-N UserCF recommendations.

        Args:
            user_id:      Target user.
            n:            Number of recommendations.
            exclude_seen: Items to exclude.

        Returns:
            Ranked list of up to n video_ids.
        """
        self._check_fitted()

        if user_id not in self._user_to_row:
            return self._fallback.recommend(user_id, n, exclude_seen)

        assert self._user_sim is not None
        assert self._matrix is not None

        u_row = self._user_to_row[user_id]
        sim_row = self._user_sim[u_row]  # shape (n_users,)

        # Get top-k neighbours (excluding self)
        sim_row[u_row] = -1.0  # exclude self
        neighbor_indices = np.argpartition(sim_row, -self._n_neighbors)[-self._n_neighbors:]
        # Sort neighbours by similarity descending
        neighbor_indices = neighbor_indices[np.argsort(sim_row[neighbor_indices])[::-1]]

        # Score items: Σ sim(u, v) × rating(v, i) for each neighbour v
        item_scores = np.zeros(self._matrix.shape[1], dtype=np.float32)
        for nb_idx in neighbor_indices:
            nb_sim = float(sim_row[nb_idx])
            if nb_sim <= 0.0:
                continue
            item_scores += nb_sim * self._matrix[nb_idx]

        # Build result: sort items by score, exclude seen
        sorted_col_indices = np.argsort(item_scores)[::-1]
        result = []
        for col_idx in sorted_col_indices:
            vid = self._col_to_item[int(col_idx)]
            if vid not in exclude_seen:
                result.append(vid)
                if len(result) >= n:
                    break

        # Fill remainder from fallback if needed
        if len(result) < n:
            fallback_recs = self._fallback.recommend(
                user_id, n - len(result), exclude_seen | set(result)
            )
            result.extend(fallback_recs)

        return result[:n]

    @property
    def fit_seconds(self) -> float:
        return self._fit_seconds
