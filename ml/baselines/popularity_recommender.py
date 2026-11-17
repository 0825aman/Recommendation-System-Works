"""Global popularity-based recommendation baseline.

Ranks all items by a weighted popularity score computed on training data:
    score = mean(watch_ratio) * log(1 + interaction_count)

This is the primary practical baseline. Any ML model that cannot outperform
popularity-based recommendations is not worth deploying.
"""

from __future__ import annotations

import math

import polars as pl

from ml.baselines.base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """Recommend globally most popular items (by weighted score) to all users.

    The same ranked list is returned to every user, with seen items excluded.
    Cold-start: works for all users (no history required).
    """

    def __init__(
        self,
        random_seed: int = 42,
        popularity_metric: str = "weighted_score",
    ) -> None:
        """Initialise the popularity recommender.

        Args:
            random_seed:       Seed for tie-breaking (rarely needed).
            popularity_metric: 'weighted_score' (default) or 'interaction_count'.
        """
        super().__init__(random_seed=random_seed)
        if popularity_metric not in ("weighted_score", "interaction_count"):
            raise ValueError(
                f"popularity_metric must be 'weighted_score' or 'interaction_count', "
                f"got '{popularity_metric}'"
            )
        self.popularity_metric = popularity_metric

    @property
    def name(self) -> str:
        return f"PopularityRecommender({self.popularity_metric})"

    def fit(self, train_df: pl.DataFrame) -> None:
        """Compute item popularity scores from training data.

        weighted_score = mean(watch_ratio) * log(1 + interaction_count)

        Args:
            train_df: Training interaction DataFrame.
        """
        agg = (
            train_df.group_by("video_id")
            .agg([
                pl.len().alias("count"),
                pl.col("watch_ratio").mean().alias("mean_wr"),
            ])
        )

        if self.popularity_metric == "weighted_score":
            agg = agg.with_columns(
                (pl.col("mean_wr") * (pl.col("count") + 1).log(base=math.e))
                .alias("score")
            )
        else:  # interaction_count
            agg = agg.with_columns(pl.col("count").cast(pl.Float64).alias("score"))

        # Store as sorted list of (video_id, score) — descending by score
        sorted_df = agg.sort("score", descending=True)
        self._ranked_items: list[int] = sorted_df["video_id"].to_list()
        self._fitted = True

    def recommend(
        self,
        user_id: int,
        n: int,
        exclude_seen: set[int],
    ) -> list[int]:
        """Return top-N most popular items, excluding seen items.

        Args:
            user_id:      Target user (not used — global ranking ignores user).
            n:            Number of recommendations.
            exclude_seen: Items to exclude.

        Returns:
            Ordered list of up to N item IDs (most popular first).
        """
        self._check_fitted()
        result = []
        for item in self._ranked_items:
            if item not in exclude_seen:
                result.append(item)
                if len(result) >= n:
                    break
        return result
