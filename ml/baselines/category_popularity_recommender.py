"""Category-based popularity recommendation baseline.

For each user, identifies their preferred categories from training history,
then recommends the most popular items within those categories.

This adds the first form of personalisation over global popularity:
items are ranked by category affinity before popularity.
"""

from __future__ import annotations

import math

import polars as pl

from ml.baselines.base import BaseRecommender


class CategoryPopularityRecommender(BaseRecommender):
    """Recommend popular items from the user's preferred categories.

    Training steps:
        1. Compute per-category popularity scores (same weighted formula as
           PopularityRecommender) — only on training data.
        2. For each user, record which categories they interacted with
           and how many times (category interaction count).

    Serving:
        - Sort user's categories by interaction count (descending).
        - Take top-N_CATEGORIES preferred categories.
        - Interleave popular items from those categories, excluding seen items.
        - Fall back to global popularity for cold users (no training history).
    """

    def __init__(
        self,
        random_seed: int = 42,
        item_to_category: dict[int, str] | None = None,
        n_top_categories: int = 3,
    ) -> None:
        """Initialise the category popularity recommender.

        Args:
            random_seed:       Seed for tie-breaking.
            item_to_category:  Mapping from video_id to category name.
                               If None, only the global fallback is used.
            n_top_categories:  Number of top user categories to consider.
        """
        super().__init__(random_seed=random_seed)
        self._item_to_category: dict[int, str] = item_to_category or {}
        self.n_top_categories = n_top_categories

    @property
    def name(self) -> str:
        return "CategoryPopularityRecommender"

    def fit(self, train_df: pl.DataFrame) -> None:
        """Build category popularity tables and user category preferences.

        Args:
            train_df: Training interaction DataFrame.
        """
        # --- Per-category popularity: weighted score per item per category ---
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

        # category → sorted list of item IDs (most popular first)
        self._category_items: dict[str, list[int]] = {}
        for row in cat_pop.iter_rows(named=True):
            cat = row["category"]
            if cat not in self._category_items:
                self._category_items[cat] = []
            self._category_items[cat].append(row["video_id"])

        # Global fallback: popularity-ranked list of all items
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

        # User category preferences: user_id → {category: interaction_count}
        user_cat = (
            df_with_cat.filter(pl.col("category").is_not_null())
            .group_by(["user_id", "category"])
            .agg(pl.len().alias("count"))
        )
        self._user_category_counts: dict[int, dict[str, int]] = {}
        for row in user_cat.iter_rows(named=True):
            uid = row["user_id"]
            if uid not in self._user_category_counts:
                self._user_category_counts[uid] = {}
            self._user_category_counts[uid][row["category"]] = row["count"]

        self._fitted = True

    def recommend(
        self,
        user_id: int,
        n: int,
        exclude_seen: set[int],
    ) -> list[int]:
        """Return top-N items from the user's preferred categories.

        Args:
            user_id:      Target user.
            n:            Number of recommendations.
            exclude_seen: Items to exclude.

        Returns:
            Ordered list of up to N item IDs.
        """
        self._check_fitted()

        user_cats = self._user_category_counts.get(user_id)

        if not user_cats:
            # Cold user: fall back to global popularity
            return [i for i in self._global_ranked if i not in exclude_seen][:n]

        # Sort categories by interaction count
        top_cats = sorted(user_cats, key=lambda c: user_cats[c], reverse=True)
        top_cats = top_cats[: self.n_top_categories]

        result: list[int] = []
        seen_in_result: set[int] = set()

        # Round-robin through top categories
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

        # Fill remaining slots from global fallback
        if len(result) < n:
            for item in self._global_ranked:
                if item not in exclude_seen and item not in seen_in_result:
                    result.append(item)
                    seen_in_result.add(item)
                    if len(result) >= n:
                        break

        return result
