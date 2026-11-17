"""User-preference heuristic recommendation baseline.

Uses each user's category engagement history (mean watch_ratio per category)
to build a soft preference vector. Recommends items from preferred categories,
ranked by category affinity then by item popularity within the category.

This is the most personalised heuristic baseline before ML.
"""

from __future__ import annotations

import math

import polars as pl

from ml.baselines.base import BaseRecommender
from ml.features.user_features import compute_user_category_affinity


class UserPreferenceRecommender(BaseRecommender):
    """Recommend items from categories the user has historically engaged with most.

    Key difference from CategoryPopularityRecommender:
    - CategoryPopularity uses interaction COUNT per category (how often)
    - UserPreference uses mean WATCH_RATIO per category (how deeply)

    Cold-start: new users receive global popularity fallback.
    """

    def __init__(
        self,
        random_seed: int = 42,
        item_to_category: dict[int, str] | None = None,
        n_top_categories: int = 3,
    ) -> None:
        """Initialise the user-preference recommender.

        Args:
            random_seed:       Seed for tie-breaking.
            item_to_category:  Mapping from video_id to category name.
            n_top_categories:  Number of top user-affinity categories to use.
        """
        super().__init__(random_seed=random_seed)
        self._item_to_category: dict[int, str] = item_to_category or {}
        self.n_top_categories = n_top_categories

    @property
    def name(self) -> str:
        return "UserPreferenceRecommender"

    def fit(self, train_df: pl.DataFrame) -> None:
        """Build user affinity vectors and per-category item popularity.

        Args:
            train_df: Training interaction DataFrame.
        """
        # User category affinity: user → {category: mean_watch_ratio}
        self._user_affinity = compute_user_category_affinity(
            train_df, self._item_to_category
        )

        # Per-category ranked items (by weighted popularity)
        df_with_cat = train_df.with_columns(
            pl.col("video_id").replace_strict(self._item_to_category, default=None).alias("category")
        )
        cat_pop = (
            df_with_cat.filter(pl.col("category").is_not_null())
            .group_by(["category", "video_id"])
            .agg([
                pl.len().alias("count"),
                pl.col("watch_ratio").mean().alias("mean_wr"),
            ])
            .with_columns(
                (pl.col("mean_wr") * (pl.col("count") + 1).log(base=math.e)).alias("score")
            )
            .sort("score", descending=True)
        )

        self._category_items: dict[str, list[int]] = {}
        for row in cat_pop.iter_rows(named=True):
            cat = row["category"]
            if cat not in self._category_items:
                self._category_items[cat] = []
            self._category_items[cat].append(row["video_id"])

        # Global fallback
        global_pop = (
            train_df.group_by("video_id")
            .agg([
                pl.len().alias("count"),
                pl.col("watch_ratio").mean().alias("mean_wr"),
            ])
            .with_columns(
                (pl.col("mean_wr") * (pl.col("count") + 1).log(base=math.e)).alias("score")
            )
            .sort("score", descending=True)
        )
        self._global_ranked: list[int] = global_pop["video_id"].to_list()
        self._fitted = True

    def recommend(
        self,
        user_id: int,
        n: int,
        exclude_seen: set[int],
    ) -> list[int]:
        """Return top-N items from the user's highest-affinity categories.

        Args:
            user_id:      Target user.
            n:            Number of recommendations.
            exclude_seen: Items to exclude.

        Returns:
            Ordered list of up to N item IDs.
        """
        self._check_fitted()

        affinity = self._user_affinity.get(user_id)
        if not affinity:
            # Cold user: fall back to global popularity
            return [i for i in self._global_ranked if i not in exclude_seen][:n]

        # Sort categories by affinity (mean_watch_ratio, descending)
        top_cats = sorted(affinity, key=lambda c: affinity[c], reverse=True)
        top_cats = top_cats[: self.n_top_categories]

        result: list[int] = []
        seen_in_result: set[int] = set()

        cat_iterators = [iter(self._category_items.get(c, [])) for c in top_cats]
        exhausted = [False] * len(cat_iterators)

        while len(result) < n and not all(exhausted):
            for i, it in enumerate(cat_iterators):
                if exhausted[i]:
                    continue
                try:
                    item = next(it)
                    while item in exclude_seen or item in seen_in_result:
                        item = next(it)
                    result.append(item)
                    seen_in_result.add(item)
                    if len(result) >= n:
                        break
                except StopIteration:
                    exhausted[i] = True

        # Fill remaining from global fallback
        if len(result) < n:
            for item in self._global_ranked:
                if item not in exclude_seen and item not in seen_in_result:
                    result.append(item)
                    seen_in_result.add(item)
                    if len(result) >= n:
                        break

        return result
