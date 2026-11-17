"""Shared pytest fixtures for ML tests.

All fixtures use synthetic data — no real KuaiRec files required.
Sizes are intentionally small for fast unit tests.
"""

from __future__ import annotations

import polars as pl
import pytest


@pytest.fixture
def tiny_interactions() -> pl.DataFrame:
    """5 users x 10 items, known watch_ratios.

    Users:   0, 1, 2, 3, 4
    Items:   100-109
    Density: 50 of 50 possible pairs (fully dense subset)
    """
    rows = []
    base_ts = 1_593_878_903.0
    for uid in range(5):
        for vid in range(100, 110):
            # watch_ratio pattern: uid 0 always relevant, uid 4 never relevant
            wr = 0.9 - uid * 0.2 + (vid - 100) * 0.01
            wr = round(max(0.0, min(wr, 2.0)), 3)
            rows.append({
                "user_id": uid,
                "video_id": vid,
                "watch_ratio": wr,
                "timestamp": base_ts + uid * 10_000 + (vid - 100) * 100,
                "date": 20200705,
            })
    return pl.DataFrame(rows).with_columns([
        pl.col("user_id").cast(pl.Int64),
        pl.col("video_id").cast(pl.Int64),
        pl.col("watch_ratio").cast(pl.Float64),
        pl.col("timestamp").cast(pl.Float64),
        pl.col("date").cast(pl.Int64),
    ])


@pytest.fixture
def tiny_with_nulls(tiny_interactions: pl.DataFrame) -> pl.DataFrame:
    """Same as tiny_interactions but first 3 rows have null timestamp."""
    return tiny_interactions.with_columns(
        pl.when(pl.int_range(pl.len()) < 3)
        .then(None)
        .otherwise(pl.col("timestamp"))
        .alias("timestamp")
    )


@pytest.fixture
def item_catalog() -> list[int]:
    """Full item catalog: items 100-119."""
    return list(range(100, 120))


@pytest.fixture
def category_map() -> dict[int, str]:
    """video_id -> category name mapping."""
    cats = ["dance", "food", "sport", "pet", "music"]
    return {100 + i: cats[i % len(cats)] for i in range(20)}
