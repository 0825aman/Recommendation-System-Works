"""Random recommendation baseline.

Recommends items uniformly at random from the catalog, excluding
items already seen by the user. Establishes the absolute performance floor.
"""

from __future__ import annotations

import random

import polars as pl

from ml.baselines.base import BaseRecommender


class RandomRecommender(BaseRecommender):
    """Recommend N items chosen uniformly at random from the item catalog.

    This is the simplest possible recommender. Any algorithm that cannot
    outperform random recommendations is not useful.

    Cold-start: works for all users (no history required).
    """

    @property
    def name(self) -> str:
        return "RandomRecommender"

    def fit(self, train_df: pl.DataFrame) -> None:
        """Build the item catalog from training data.

        Args:
            train_df: Training interaction DataFrame.
        """
        self._catalog: list[int] = train_df["video_id"].unique().sort().to_list()
        self._rng = random.Random(self.random_seed)
        self._fitted = True

    def recommend(
        self,
        user_id: int,
        n: int,
        exclude_seen: set[int],
    ) -> list[int]:
        """Return N randomly selected items, excluding seen items.

        Args:
            user_id:      Target user (not used — random baseline ignores user history).
            n:            Number of recommendations.
            exclude_seen: Items to exclude from recommendations.

        Returns:
            List of up to N item IDs in random order.
        """
        self._check_fitted()
        candidates = [v for v in self._catalog if v not in exclude_seen]
        k = min(n, len(candidates))
        return self._rng.sample(candidates, k)
