"""Pytest fixtures for data pipeline tests.

Provides synthetic CSV fixtures that match the KuaiRec 2.0 schema
without requiring real dataset files. All fixtures are written to
a temporary directory and cleaned up after the test session.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl
import pytest


@pytest.fixture(scope="session")
def tmp_data_dir() -> Path:
    """Temporary directory for test fixtures."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture(scope="session")
def small_matrix_csv(tmp_data_dir: Path) -> Path:
    """Synthetic small_matrix.csv with 10 rows matching KuaiRec schema."""
    path = tmp_data_dir / "small_matrix.csv"
    df = pl.DataFrame({
        "user_id": [0, 0, 0, 1, 1, 1, 2, 2, 3, 3],
        "video_id": [100, 101, 102, 100, 103, 104, 101, 105, 100, 106],
        "play_duration": [5000, 10000, 3000, 8000, 12000, 1000, 9000, 15000, 7000, 4000],
        "video_duration": [10000, 10000, 10000, 10000, 10000, 10000, 10000, 15000, 10000, 8000],
        "time": [
            "2020-07-05 00:08:23.438", "2020-07-05 00:13:41.297", None,
            "2020-07-06 10:00:00.000", "2020-07-06 10:05:00.000", None,
            "2020-07-07 08:00:00.000", "2020-07-07 08:30:00.000",
            "2020-07-08 09:00:00.000", "2020-07-08 09:15:00.000",
        ],
        "date": [20200705, 20200705, None, 20200706, 20200706, None, 20200707, 20200707, 20200708, 20200708],
        "timestamp": [1593878903.0, 1593879221.0, None, 1593979200.0, 1593979500.0, None, 1594065600.0, 1594067400.0, 1594152000.0, 1594152900.0],
        "watch_ratio": [0.5, 1.0, 0.3, 0.8, 1.2, 0.1, 0.9, 1.0, 0.7, 0.5],
    })
    df.write_csv(path)
    return path


@pytest.fixture(scope="session")
def user_features_csv(tmp_data_dir: Path) -> Path:
    """Synthetic user_features.csv with 4 users."""
    path = tmp_data_dir / "user_features.csv"
    df = pl.DataFrame({
        "user_id": [0, 1, 2, 3],
        "user_active_degree": ["high_active", "full_active", "middle_active", "UNKNOWN"],
        "is_lowactive_period": [0, 0, 1, 0],
        "is_live_streamer": [0, 1, 0, 0],
        "is_video_author": [1, 0, 0, 0],
        "follow_user_num": [5, 10, 2, 0],
        "follow_user_num_range": ["(0,10]", "(10,50]", "(0,10]", "0"],
        "fans_user_num": [0, 100, 0, 0],
        "fans_user_num_range": ["0", "[100,1k)", "0", "0"],
        "friend_user_num": [0, 2, 0, 1],
        "friend_user_num_range": ["0", "[1,5)", "0", "[1,5)"],
        "register_days": [107, 365, 30, 90],
        "register_days_range": ["61-90", "366-730", "15-30", "61-90"],
        "onehot_feat0": [0, 1, 0, 0],
        "onehot_feat1": [1, 2, 0, 3],
        "onehot_feat2": [17, 5, 0, 12],
        "onehot_feat3": [638, 100, 0, 200],
        "onehot_feat4": [2, 1, 0, 3],
        "onehot_feat5": [0, 1, 0, 2],
        "onehot_feat6": [1, 0, 2, 1],
        "onehot_feat7": [6, 10, 0, 20],
        "onehot_feat8": [184, 50, 0, 100],
        "onehot_feat9": [6, 3, 0, 1],
        "onehot_feat10": [3, 2, 0, 1],
        "onehot_feat11": [0, 1, 2, 0],
        "onehot_feat12": [0, 1, 0, 0],
        "onehot_feat13": [0, 0, 1, 0],
        "onehot_feat14": [0, 0, 0, 1],
        "onehot_feat15": [0, 1, 0, 0],
        "onehot_feat16": [0, 0, 0, 0],
        "onehot_feat17": [1, 0, 0, 0],
    })
    df.write_csv(path)
    return path


@pytest.fixture(scope="session")
def item_categories_csv(tmp_data_dir: Path) -> Path:
    """Synthetic item_categories.csv with 7 videos."""
    path = tmp_data_dir / "item_categories.csv"
    df = pl.DataFrame({
        "video_id": [100, 101, 102, 103, 104, 105, 106],
        "feat": ["[27, 9]", "[5]", "[12, 3, 7]", "[27]", "[9, 15]", "[3]", "[27, 9, 5]"],
    })
    df.write_csv(path)
    return path


@pytest.fixture(scope="session")
def caption_category_csv(tmp_data_dir: Path) -> Path:
    """Synthetic kuairec_caption_category.csv with 7 videos."""
    path = tmp_data_dir / "kuairec_caption_category.csv"
    df = pl.DataFrame({
        "video_id": [100, 101, 102, 103, 104, 105, 106],
        "manual_cover_text": ["text1", "text2", None, "text4", "text5", "text6", "text7"],
        "caption": ["caption1", "caption2", None, "caption4", "caption5", None, "caption7"],
        "topic_tag": ["[tag1]", "[tag2,tag3]", "[tag1]", "[tag2]", "[tag3]", "[tag1]", "[tag2]"],
        "first_level_category_id": [17, 5, 17, 10, 5, 17, 10],
        "first_level_category_name": ["pet", "food", "pet", "sport", "food", "pet", "sport"],
        "second_level_category_id": [233, 50, 233, 100, 50, 233, 100],
        "second_level_category_name": ["cat", "snack", "dog", "run", "pizza", "rabbit", "swim"],
        "third_level_category_id": [1169, 500, 1170, 300, 501, 1171, 301],
        "third_level_category_name": ["cute_cat", "chips", "poodle", "marathon", "pasta", "bunny", "freestyle"],
    })
    df.write_csv(path)
    return path
