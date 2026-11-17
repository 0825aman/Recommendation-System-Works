"""Tests for Phase 4B Collaborative Filtering components.

Tests interaction_matrix, UserBasedCF, ItemBasedCF, and SVDRecommender.
Uses synthetic fixtures — no real Parquet files required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ml.collaborative.interaction_matrix import build_interaction_matrix
from ml.collaborative.item_cf import ItemBasedCF
from ml.collaborative.matrix_factorization import SVDRecommender
from ml.collaborative.user_cf import UserBasedCF

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_train_cf() -> pl.DataFrame:
    """5 users x 10 items, dense interactions for CF testing."""
    rows = []
    for uid in range(5):
        for vid in range(200, 210):
            wr = 0.9 - uid * 0.15 + (vid - 200) * 0.02
            wr = round(max(0.0, min(wr, 2.0)), 3)
            rows.append({"user_id": uid, "video_id": vid, "watch_ratio": wr})
    return pl.DataFrame(rows).with_columns([
        pl.col("user_id").cast(pl.Int64),
        pl.col("video_id").cast(pl.Int64),
        pl.col("watch_ratio").cast(pl.Float64),
    ])


# ---------------------------------------------------------------------------
# Tests: build_interaction_matrix
# ---------------------------------------------------------------------------


class TestInteractionMatrix:
    def test_matrix_shape(self, tiny_train_cf: pl.DataFrame) -> None:
        matrix, _user_to_row, _item_to_col, _, _ = build_interaction_matrix(tiny_train_cf)
        assert matrix.shape == (5, 10)

    def test_index_maps_cover_all_users_and_items(
        self, tiny_train_cf: pl.DataFrame
    ) -> None:
        _, user_to_row, item_to_col, _, _ = build_interaction_matrix(tiny_train_cf)
        assert set(user_to_row.keys()) == set(range(5))
        assert set(item_to_col.keys()) == set(range(200, 210))

    def test_round_trip_user_id(self, tiny_train_cf: pl.DataFrame) -> None:
        _, user_to_row, _, row_to_user, _ = build_interaction_matrix(tiny_train_cf)
        for uid, row in user_to_row.items():
            assert row_to_user[row] == uid

    def test_round_trip_item_id(self, tiny_train_cf: pl.DataFrame) -> None:
        _, _, item_to_col, _, col_to_item = build_interaction_matrix(tiny_train_cf)
        for iid, col in item_to_col.items():
            assert col_to_item[col] == iid

    def test_clipping_applied(self) -> None:
        # Create a row with watch_ratio = 100 → should be clipped to 5.0
        df = pl.DataFrame({
            "user_id": [0, 0],
            "video_id": [1, 2],
            "watch_ratio": [100.0, 1.0],
        }).with_columns([
            pl.col("user_id").cast(pl.Int64),
            pl.col("video_id").cast(pl.Int64),
            pl.col("watch_ratio").cast(pl.Float64),
        ])
        matrix, user_to_row, item_to_col, _, _ = build_interaction_matrix(df, max_watch_ratio=5.0)
        u_row = user_to_row[0]
        item_col = item_to_col[1]
        assert matrix[u_row, item_col] == pytest.approx(5.0)

    def test_zero_entries_for_unobserved(self, tiny_train_cf: pl.DataFrame) -> None:
        # In this dense fixture all entries should be non-zero (fully observed)
        matrix, _, _, _, _ = build_interaction_matrix(tiny_train_cf)
        assert float(matrix.min()) > 0.0


# ---------------------------------------------------------------------------
# Tests: UserBasedCF
# ---------------------------------------------------------------------------


class TestUserBasedCF:
    def test_fit_does_not_raise(self, tiny_train_cf: pl.DataFrame) -> None:
        rec = UserBasedCF(n_neighbors=3)
        rec.fit(tiny_train_cf)
        assert rec._fitted is True

    def test_recommend_returns_n_items(self, tiny_train_cf: pl.DataFrame) -> None:
        rec = UserBasedCF(n_neighbors=3)
        rec.fit(tiny_train_cf)
        recs = rec.recommend(user_id=0, n=5, exclude_seen=set())
        assert len(recs) == 5

    def test_recommend_no_duplicates(self, tiny_train_cf: pl.DataFrame) -> None:
        rec = UserBasedCF(n_neighbors=3)
        rec.fit(tiny_train_cf)
        recs = rec.recommend(user_id=0, n=8, exclude_seen=set())
        assert len(recs) == len(set(recs))

    def test_recommend_excludes_seen(self, tiny_train_cf: pl.DataFrame) -> None:
        rec = UserBasedCF(n_neighbors=3)
        rec.fit(tiny_train_cf)
        seen = {200, 201, 202}
        recs = rec.recommend(user_id=0, n=5, exclude_seen=seen)
        assert not any(r in seen for r in recs)

    def test_cold_start_uses_fallback(self, tiny_train_cf: pl.DataFrame) -> None:
        rec = UserBasedCF(n_neighbors=3)
        rec.fit(tiny_train_cf)
        recs = rec.recommend(user_id=9999, n=5, exclude_seen=set())
        assert len(recs) > 0

    def test_recommend_before_fit_raises(self) -> None:
        rec = UserBasedCF()
        with pytest.raises(RuntimeError):
            rec.recommend(user_id=0, n=5, exclude_seen=set())

    def test_name_contains_k(self) -> None:
        rec = UserBasedCF(n_neighbors=25)
        assert "25" in rec.name


# ---------------------------------------------------------------------------
# Tests: ItemBasedCF
# ---------------------------------------------------------------------------


class TestItemBasedCF:
    def test_fit_does_not_raise(self, tiny_train_cf: pl.DataFrame) -> None:
        rec = ItemBasedCF(n_neighbors=3)
        rec.fit(tiny_train_cf)
        assert rec._fitted is True

    def test_recommend_returns_n_items(self, tiny_train_cf: pl.DataFrame) -> None:
        rec = ItemBasedCF(n_neighbors=3)
        rec.fit(tiny_train_cf)
        recs = rec.recommend(user_id=0, n=5, exclude_seen=set())
        assert len(recs) == 5

    def test_recommend_no_duplicates(self, tiny_train_cf: pl.DataFrame) -> None:
        rec = ItemBasedCF(n_neighbors=3)
        rec.fit(tiny_train_cf)
        recs = rec.recommend(user_id=0, n=8, exclude_seen=set())
        assert len(recs) == len(set(recs))

    def test_recommend_excludes_seen(self, tiny_train_cf: pl.DataFrame) -> None:
        rec = ItemBasedCF(n_neighbors=3)
        rec.fit(tiny_train_cf)
        seen = {200, 201}
        recs = rec.recommend(user_id=0, n=5, exclude_seen=seen)
        assert not any(r in seen for r in recs)

    def test_cold_start_uses_fallback(self, tiny_train_cf: pl.DataFrame) -> None:
        rec = ItemBasedCF(n_neighbors=3)
        rec.fit(tiny_train_cf)
        recs = rec.recommend(user_id=9999, n=5, exclude_seen=set())
        assert len(recs) > 0

    def test_recommend_before_fit_raises(self) -> None:
        rec = ItemBasedCF()
        with pytest.raises(RuntimeError):
            rec.recommend(user_id=0, n=5, exclude_seen=set())

    def test_name_contains_k(self) -> None:
        rec = ItemBasedCF(n_neighbors=25)
        assert "25" in rec.name


# ---------------------------------------------------------------------------
# Tests: SVDRecommender
# ---------------------------------------------------------------------------


class TestSVDRecommender:
    def test_fit_does_not_raise(self, tiny_train_cf: pl.DataFrame) -> None:
        rec = SVDRecommender(n_components=4)
        rec.fit(tiny_train_cf)
        assert rec._fitted is True

    def test_recommend_returns_n_items(self, tiny_train_cf: pl.DataFrame) -> None:
        rec = SVDRecommender(n_components=4)
        rec.fit(tiny_train_cf)
        recs = rec.recommend(user_id=0, n=5, exclude_seen=set())
        assert len(recs) == 5

    def test_recommend_no_duplicates(self, tiny_train_cf: pl.DataFrame) -> None:
        rec = SVDRecommender(n_components=4)
        rec.fit(tiny_train_cf)
        recs = rec.recommend(user_id=0, n=8, exclude_seen=set())
        assert len(recs) == len(set(recs))

    def test_recommend_excludes_seen(self, tiny_train_cf: pl.DataFrame) -> None:
        rec = SVDRecommender(n_components=4)
        rec.fit(tiny_train_cf)
        seen = {200, 201}
        recs = rec.recommend(user_id=0, n=5, exclude_seen=seen)
        assert not any(r in seen for r in recs)

    def test_cold_start_uses_fallback(self, tiny_train_cf: pl.DataFrame) -> None:
        rec = SVDRecommender(n_components=4)
        rec.fit(tiny_train_cf)
        recs = rec.recommend(user_id=9999, n=5, exclude_seen=set())
        assert len(recs) > 0

    def test_deterministic_with_same_seed(self, tiny_train_cf: pl.DataFrame) -> None:
        rec1 = SVDRecommender(n_components=4, random_state=42)
        rec1.fit(tiny_train_cf)
        rec2 = SVDRecommender(n_components=4, random_state=42)
        rec2.fit(tiny_train_cf)
        assert (
            rec1.recommend(user_id=0, n=5, exclude_seen=set())
            == rec2.recommend(user_id=0, n=5, exclude_seen=set())
        )

    def test_recommend_before_fit_raises(self) -> None:
        rec = SVDRecommender()
        with pytest.raises(RuntimeError):
            rec.recommend(user_id=0, n=5, exclude_seen=set())

    def test_name_contains_k(self) -> None:
        rec = SVDRecommender(n_components=30)
        assert "30" in rec.name

    def test_embeddings_shape(self, tiny_train_cf: pl.DataFrame) -> None:
        n_comp = 4
        rec = SVDRecommender(n_components=n_comp)
        rec.fit(tiny_train_cf)
        assert rec._user_embeddings is not None
        assert rec._item_embeddings is not None
        assert rec._user_embeddings.shape == (5, n_comp)
        assert rec._item_embeddings.shape == (10, n_comp)
