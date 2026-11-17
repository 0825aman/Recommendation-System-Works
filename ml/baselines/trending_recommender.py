"""Trending/recent popularity recommendation baseline.

Computes popularity scores using only interactions within a recent time window
from the end of the training period. Items trending recently outrank items that
were popular earlier but have since declined.

Requires: timestamp column in the interaction DataFrame.
"""

from __future__ import annotations

import math

import polars as pl

from ml.baselines.base import BaseRecommender

_SECONDS_PER_DAY = 86_400.0


class TrendingRecommender(BaseRecommender):
    """Recommend currently trending items based on recent interactions.

    Items are ranked by popularity within the most recent N days of training.
    Rows with null timestamps are excluded from recency scoring.

    Cold-start: works for all users (no personal history required).
    Falls back to all-time popularity when the recency window is empty.
    """

    def __init__(
        self,
        random_seed: int = 42,
        recency_window_days: int = 14,
    ) -> None:
        """Initialise the trending recommender.

        Args:
            random_seed:          Seed for tie-breaking.
            recency_window_days:  Number of days from the end of training to use.
        """
        super().__init__(random_seed=random_seed)
        self.recency_window_days = recency_window_days

    @property
    def name(self) -> str:
        return f"TrendingRecommender(window={self.recency_window_days}d)"

    def fit(self, train_df: pl.DataFrame) -> None:
        """Compute trending scores from recent training interactions.

        Args:
            train_df: Training interaction DataFrame.
                      Must contain 'timestamp' (Float64) and 'watch_ratio' columns.
        """
        timed = train_df.filter(pl.col("timestamp").is_not_null())

        if timed.height == 0:
            # No timed rows — fall back to full training data interaction count
            self._ranked_items = (
                train_df.group_by("video_id")
                .agg(pl.len().alias("count"))
                .sort("count", descending=True)["video_id"]
                .to_list()
            )
            self._fitted = True
            return

        max_ts = float(timed["timestamp"].max())  # type: ignore[arg-type]
        cutoff_ts = max_ts - self.recency_window_days * _SECONDS_PER_DAY

        recent = timed.filter(pl.col("timestamp") >= cutoff_ts)

        if recent.height == 0:
            # Window is empty — fall back to all-time popularity
            fallback = (
                timed.group_by("video_id")
                .agg([
                    pl.len().alias("count"),
                    pl.col("watch_ratio").mean().alias("mean_wr"),
                ])
                .with_columns(
                    (pl.col("mean_wr") * (pl.col("count") + 1).log(base=math.e)).alias("score")
                )
                .sort("score", descending=True)
            )
            self._ranked_items = fallback["video_id"].to_list()
        else:
            ranked = (
                recent.group_by("video_id")
                .agg([
                    pl.len().alias("count"),
                    pl.col("watch_ratio").mean().alias("mean_wr"),
                ])
                .with_columns(
                    (pl.col("mean_wr") * (pl.col("count") + 1).log(base=math.e)).alias("score")
                )
                .sort("score", descending=True)
            )
            self._ranked_items = ranked["video_id"].to_list()

        self._fitted = True

    def recommend(
        self,
        user_id: int,
        n: int,
        exclude_seen: set[int],
    ) -> list[int]:
        """Return top-N trending items, excluding seen items.

        Args:
            user_id:      Target user (not used — trending is global).
            n:            Number of recommendations.
            exclude_seen: Items to exclude.

        Returns:
            Ordered list of up to N item IDs (most trending first).
        """
        self._check_fitted()
        result = []
        for item in self._ranked_items:
            if item not in exclude_seen:
                result.append(item)
                if len(result) >= n:
                    break
        return result
