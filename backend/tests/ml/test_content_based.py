"""Tests for Phase 4 Content-Based Filtering components.

Tests item_encoder, similarity, user_profile, and ContentBasedRecommender.
Uses synthetic fixtures — no real Parquet files required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ml.content_based.item_encoder import (
    _l2_norm,
    _multi_hot_encode,
    _one_hot_encode,
    build_item_feature_matrix,
)
from ml.content_based.similarity import (
    cosine_similarity_pair,
    score_items_for_vector,
    top_k_similar_items,
)
from ml.content_based.user_profile import build_user_profiles

if TYPE_CHECKING:
    from ml.content_based.content_recommender import ContentBasedRecommender

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_caption_df() -> pl.DataFrame:
    """5 items with category hierarchy and captions."""
    return pl.DataFrame(
        {
            "video_id": [100, 101, 102, 103, 104],
            "caption": ["dance video", "food show", None, "sport match", "pet clips"],
            "first_level_category_name": ["舞蹈", "美食", "美食", "体育", "宠物"],
            "second_level_category_name": ["hip-hop", "cooking", "street-food", "football", "cats"],
        }
    )


@pytest.fixture()
def tiny_item_categories_df() -> pl.DataFrame:
    """5 items with feat_parsed feature IDs."""
    return pl.DataFrame(
        {
            "video_id": [100, 101, 102, 103, 104],
            "feat": ["[1,2]", "[2,3]", "[3]", "[1,4]", "[4,5]"],
            "feat_parsed": [[1, 2], [2, 3], [3], [1, 4], [4, 5]],
        }
    )


# ---------------------------------------------------------------------------
# Tests: _one_hot_encode
# ---------------------------------------------------------------------------


class TestOneHotEncode:
    def test_known_category_produces_correct_column(self) -> None:
        cats = ["A", "B", "A"]
        categories = ["A", "B"]
        result = _one_hot_encode(cats, categories)
        assert result[0] == [1.0, 0.0]  # A
        assert result[1] == [0.0, 1.0]  # B
        assert result[2] == [1.0, 0.0]  # A

    def test_none_produces_zero_row(self) -> None:
        result = _one_hot_encode([None], ["A", "B"])
        assert result[0] == [0.0, 0.0]

    def test_unknown_category_produces_zero_row(self) -> None:
        result = _one_hot_encode(["Z"], ["A", "B"])
        assert result[0] == [0.0, 0.0]

    def test_output_width_matches_category_count(self) -> None:
        result = _one_hot_encode(["A", "B"], ["A", "B", "C"])
        assert all(len(row) == 3 for row in result)


# ---------------------------------------------------------------------------
# Tests: _multi_hot_encode
# ---------------------------------------------------------------------------


class TestMultiHotEncode:
    def test_basic_encoding(self) -> None:
        feat_lists = [[1, 2], [2, 3]]
        all_ids = [1, 2, 3]
        result = _multi_hot_encode(feat_lists, all_ids)
        assert result[0] == [1.0, 1.0, 0.0]
        assert result[1] == [0.0, 1.0, 1.0]

    def test_none_produces_zero_row(self) -> None:
        result = _multi_hot_encode([None], [1, 2, 3])
        assert result[0] == [0.0, 0.0, 0.0]

    def test_empty_list_produces_zero_row(self) -> None:
        result = _multi_hot_encode([[]], [1, 2, 3])
        assert result[0] == [0.0, 0.0, 0.0]

    def test_unknown_feat_id_ignored(self) -> None:
        # ID 99 not in all_ids → no error, just ignored
        result = _multi_hot_encode([[1, 99]], [1, 2])
        assert result[0] == [1.0, 0.0]


# ---------------------------------------------------------------------------
# Tests: _l2_norm
# ---------------------------------------------------------------------------


class TestL2Norm:
    def test_unit_vector(self) -> None:
        assert _l2_norm([1.0, 0.0, 0.0]) == pytest.approx(1.0)

    def test_known_norm(self) -> None:
        assert _l2_norm([3.0, 4.0]) == pytest.approx(5.0)

    def test_zero_vector(self) -> None:
        assert _l2_norm([0.0, 0.0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: build_item_feature_matrix
# ---------------------------------------------------------------------------


class TestBuildItemFeatureMatrix:
    def test_returns_correct_number_of_rows(
        self,
        tiny_caption_df: pl.DataFrame,
        tiny_item_categories_df: pl.DataFrame,
    ) -> None:
        _id_to_idx, _idx_to_id, matrix = build_item_feature_matrix(
            item_categories_df=tiny_item_categories_df,
            caption_category_df=tiny_caption_df,
            tfidf_weight=0.0,  # Disable TF-IDF for determinism
        )
        assert len(matrix) == 5

    def test_id_maps_are_consistent(
        self,
        tiny_caption_df: pl.DataFrame,
        tiny_item_categories_df: pl.DataFrame,
    ) -> None:
        id_to_idx, idx_to_id, _ = build_item_feature_matrix(
            item_categories_df=tiny_item_categories_df,
            caption_category_df=tiny_caption_df,
            tfidf_weight=0.0,
        )
        for vid, idx in id_to_idx.items():
            assert idx_to_id[idx] == vid

    def test_all_video_ids_present(
        self,
        tiny_caption_df: pl.DataFrame,
        tiny_item_categories_df: pl.DataFrame,
    ) -> None:
        id_to_idx, _, _ = build_item_feature_matrix(
            item_categories_df=tiny_item_categories_df,
            caption_category_df=tiny_caption_df,
            tfidf_weight=0.0,
        )
        assert set(id_to_idx.keys()) == {100, 101, 102, 103, 104}

    def test_rows_are_unit_vectors_when_normalized(
        self,
        tiny_caption_df: pl.DataFrame,
        tiny_item_categories_df: pl.DataFrame,
    ) -> None:
        _, _, matrix = build_item_feature_matrix(
            item_categories_df=tiny_item_categories_df,
            caption_category_df=tiny_caption_df,
            normalize=True,
            tfidf_weight=0.0,
        )
        for row in matrix:
            norm = _l2_norm(row)
            assert norm == pytest.approx(1.0, abs=1e-6)

    def test_null_caption_produces_valid_row(
        self,
        tiny_caption_df: pl.DataFrame,
        tiny_item_categories_df: pl.DataFrame,
    ) -> None:
        # video_id=102 has null caption — should not raise
        id_to_idx, _, matrix = build_item_feature_matrix(
            item_categories_df=tiny_item_categories_df,
            caption_category_df=tiny_caption_df,
            tfidf_weight=0.0,
        )
        idx = id_to_idx[102]
        # Row should be valid (not all zeros — it has category features)
        row = matrix[idx]
        assert len(row) > 0
        assert not all(x == 0.0 for x in row)

    def test_feature_dimension_is_deterministic(
        self,
        tiny_caption_df: pl.DataFrame,
        tiny_item_categories_df: pl.DataFrame,
    ) -> None:
        _, _, m1 = build_item_feature_matrix(
            item_categories_df=tiny_item_categories_df,
            caption_category_df=tiny_caption_df,
            tfidf_weight=0.0,
        )
        _, _, m2 = build_item_feature_matrix(
            item_categories_df=tiny_item_categories_df,
            caption_category_df=tiny_caption_df,
            tfidf_weight=0.0,
        )
        assert len(m1[0]) == len(m2[0])

    def test_tfidf_increases_dimension(
        self,
        tiny_caption_df: pl.DataFrame,
        tiny_item_categories_df: pl.DataFrame,
    ) -> None:
        _, _, m_no_tfidf = build_item_feature_matrix(
            item_categories_df=tiny_item_categories_df,
            caption_category_df=tiny_caption_df,
            tfidf_weight=0.0,
        )
        _, _, m_with_tfidf = build_item_feature_matrix(
            item_categories_df=tiny_item_categories_df,
            caption_category_df=tiny_caption_df,
            tfidf_weight=0.5,
            tfidf_max_features=50,
        )
        # With TF-IDF the feature dimension must be strictly larger
        assert len(m_with_tfidf[0]) > len(m_no_tfidf[0])

    def test_no_normalize_rows_not_unit(
        self,
        tiny_caption_df: pl.DataFrame,
        tiny_item_categories_df: pl.DataFrame,
    ) -> None:
        _, _, matrix = build_item_feature_matrix(
            item_categories_df=tiny_item_categories_df,
            caption_category_df=tiny_caption_df,
            normalize=False,
            tfidf_weight=0.0,
        )
        # Rows with multiple one-hot bits will have norm > 1
        norms = [_l2_norm(row) for row in matrix]
        assert any(n > 1.0 + 1e-6 for n in norms)


# ---------------------------------------------------------------------------
# Tests: cosine_similarity_pair
# ---------------------------------------------------------------------------


class TestCosineSimilarityPair:
    def test_identical_vectors_return_one(self) -> None:
        assert cosine_similarity_pair([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self) -> None:
        assert cosine_similarity_pair([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_return_minus_one(self) -> None:
        assert cosine_similarity_pair([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self) -> None:
        assert cosine_similarity_pair([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_known_value(self) -> None:
        # [1, 1] and [1, 0]: cos = 1/sqrt(2) ≈ 0.7071
        result = cosine_similarity_pair([1.0, 1.0], [1.0, 0.0])
        assert result == pytest.approx(1.0 / (2**0.5), rel=1e-5)

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError):
            cosine_similarity_pair([1.0, 2.0], [1.0])


# ---------------------------------------------------------------------------
# Tests: score_items_for_vector and top_k_similar_items
# ---------------------------------------------------------------------------


class TestTopKSimilarItems:
    @pytest.fixture()
    def simple_matrix(self) -> tuple[list[list[float]], dict[int, int]]:
        # 3 items, 2D vectors (already unit length)
        matrix = [
            [1.0, 0.0],  # item 10
            [0.0, 1.0],  # item 11
            [1.0, 0.0],  # item 12 — identical to item 10
        ]
        idx_to_id = {0: 10, 1: 11, 2: 12}
        return matrix, idx_to_id

    def test_returns_k_items(
        self, simple_matrix: tuple[list[list[float]], dict[int, int]]
    ) -> None:
        matrix, idx_to_id = simple_matrix
        result = top_k_similar_items([1.0, 0.0], matrix, idx_to_id, k=2)
        assert len(result) == 2

    def test_most_similar_first(
        self, simple_matrix: tuple[list[list[float]], dict[int, int]]
    ) -> None:
        matrix, idx_to_id = simple_matrix
        # Query [1,0] is closest to items 10 and 12 (both [1,0])
        result = top_k_similar_items([1.0, 0.0], matrix, idx_to_id, k=3)
        # First two should be 10 and 12 (in either order), last is 11
        assert set(result[:2]) == {10, 12}
        assert result[2] == 11

    def test_excludes_specified_ids(
        self, simple_matrix: tuple[list[list[float]], dict[int, int]]
    ) -> None:
        matrix, idx_to_id = simple_matrix
        result = top_k_similar_items([1.0, 0.0], matrix, idx_to_id, k=3, exclude_ids={10})
        assert 10 not in result

    def test_zero_query_vector_returns_items(
        self, simple_matrix: tuple[list[list[float]], dict[int, int]]
    ) -> None:
        matrix, idx_to_id = simple_matrix
        result = top_k_similar_items([0.0, 0.0], matrix, idx_to_id, k=2)
        assert len(result) == 2

    def test_k_zero_returns_empty(
        self, simple_matrix: tuple[list[list[float]], dict[int, int]]
    ) -> None:
        matrix, idx_to_id = simple_matrix
        result = top_k_similar_items([1.0, 0.0], matrix, idx_to_id, k=0)
        assert result == []

    def test_score_items_returns_all_items(
        self, simple_matrix: tuple[list[list[float]], dict[int, int]]
    ) -> None:
        matrix, idx_to_id = simple_matrix
        scored = score_items_for_vector([1.0, 0.0], matrix, idx_to_id)
        assert len(scored) == 3
        assert all(isinstance(vid, int) and isinstance(score, float) for vid, score in scored)


# ---------------------------------------------------------------------------
# Tests: build_user_profiles
# ---------------------------------------------------------------------------


class TestBuildUserProfiles:
    @pytest.fixture()
    def simple_item_matrix(self) -> tuple[list[list[float]], dict[int, int]]:
        """3 items, 2D unit vectors."""
        matrix = [[1.0, 0.0], [0.0, 1.0], [0.707, 0.707]]
        id_to_idx = {100: 0, 101: 1, 102: 2}
        return matrix, id_to_idx

    @pytest.fixture()
    def simple_train_df(self) -> pl.DataFrame:
        """2 users: user 0 likes item 100; user 1 likes items 100 and 101."""
        return pl.DataFrame(
            {
                "user_id": [0, 1, 1],
                "video_id": [100, 100, 101],
                "watch_ratio": [1.0, 0.8, 0.6],
            }
        )

    def test_returns_profile_for_each_user(
        self,
        simple_train_df: pl.DataFrame,
        simple_item_matrix: tuple[list[list[float]], dict[int, int]],
    ) -> None:
        matrix, id_to_idx = simple_item_matrix
        profiles = build_user_profiles(simple_train_df, matrix, id_to_idx)
        assert set(profiles.keys()) == {0, 1}

    def test_profile_has_correct_dimension(
        self,
        simple_train_df: pl.DataFrame,
        simple_item_matrix: tuple[list[list[float]], dict[int, int]],
    ) -> None:
        matrix, id_to_idx = simple_item_matrix
        profiles = build_user_profiles(simple_train_df, matrix, id_to_idx)
        for vec in profiles.values():
            assert len(vec) == 2

    def test_single_item_user_profile_matches_item(
        self,
        simple_train_df: pl.DataFrame,
        simple_item_matrix: tuple[list[list[float]], dict[int, int]],
    ) -> None:
        matrix, id_to_idx = simple_item_matrix
        # User 0 only interacted with item 100 → profile should be [1.0, 0.0]
        profiles = build_user_profiles(simple_train_df, matrix, id_to_idx, normalize=False)
        assert profiles[0] == pytest.approx([1.0, 0.0])

    def test_normalized_profile_is_unit_vector(
        self,
        simple_train_df: pl.DataFrame,
        simple_item_matrix: tuple[list[list[float]], dict[int, int]],
    ) -> None:
        matrix, id_to_idx = simple_item_matrix
        profiles = build_user_profiles(simple_train_df, matrix, id_to_idx, normalize=True)
        for vec in profiles.values():
            norm = sum(x * x for x in vec) ** 0.5
            assert norm == pytest.approx(1.0, abs=1e-6)

    def test_cold_start_user_gets_zero_vector(
        self,
        simple_item_matrix: tuple[list[list[float]], dict[int, int]],
    ) -> None:
        matrix, id_to_idx = simple_item_matrix
        # User 99 interacted with item 999 which is NOT in id_to_idx
        cold_df = pl.DataFrame(
            {"user_id": [99], "video_id": [999], "watch_ratio": [0.9]}
        )
        profiles = build_user_profiles(cold_df, matrix, id_to_idx)
        assert profiles[99] == [0.0, 0.0]

    def test_weighted_vs_uniform_profiles_differ(
        self,
        simple_item_matrix: tuple[list[list[float]], dict[int, int]],
    ) -> None:
        matrix, id_to_idx = simple_item_matrix
        # User 1 has watch_ratios 0.8 and 0.6 → weighted differs from uniform
        df = pl.DataFrame(
            {"user_id": [1, 1], "video_id": [100, 101], "watch_ratio": [0.9, 0.1]}
        )
        weighted = build_user_profiles(df, matrix, id_to_idx, weight_by_watch_ratio=True)
        uniform = build_user_profiles(df, matrix, id_to_idx, weight_by_watch_ratio=False)
        # Weighted should favour item 100 (high watch_ratio)
        # uniform averages them equally
        assert weighted[1] != pytest.approx(uniform[1], abs=1e-3)


# ---------------------------------------------------------------------------
# Tests: ContentBasedRecommender (integration-level with synthetic data)
# ---------------------------------------------------------------------------


class TestContentBasedRecommender:
    """Integration tests for ContentBasedRecommender using tiny synthetic data."""

    @pytest.fixture()
    def cbf_rec(
        self,
        tiny_caption_df: pl.DataFrame,
        tiny_item_categories_df: pl.DataFrame,
    ) -> ContentBasedRecommender:
        from ml.content_based.content_recommender import (
            ContentBasedRecommender,
        )

        return ContentBasedRecommender(
            item_categories_df=tiny_item_categories_df,
            caption_category_df=tiny_caption_df,
            tfidf_weight=0.0,  # Disable TF-IDF for determinism in unit tests
        )

    @pytest.fixture()
    def tiny_train_df(self) -> pl.DataFrame:
        """3 users, 5 items (matches tiny_caption_df video_ids 100-104)."""
        return pl.DataFrame(
            {
                "user_id": [0, 0, 1, 1, 2],
                "video_id": [100, 101, 102, 103, 104],
                "watch_ratio": [0.9, 0.7, 0.6, 0.8, 1.0],
            }
        ).with_columns([
            pl.col("user_id").cast(pl.Int64),
            pl.col("video_id").cast(pl.Int64),
            pl.col("watch_ratio").cast(pl.Float64),
        ])

    def test_fit_does_not_raise(
        self,
        cbf_rec: ContentBasedRecommender,
        tiny_train_df: pl.DataFrame,
    ) -> None:
        cbf_rec.fit(tiny_train_df)
        assert cbf_rec._fitted is True

    def test_recommend_returns_n_items(
        self,
        cbf_rec: ContentBasedRecommender,
        tiny_train_df: pl.DataFrame,
    ) -> None:
        cbf_rec.fit(tiny_train_df)
        recs = cbf_rec.recommend(user_id=0, n=3, exclude_seen=set())
        assert len(recs) == 3

    def test_recommend_no_duplicates(
        self,
        cbf_rec: ContentBasedRecommender,
        tiny_train_df: pl.DataFrame,
    ) -> None:
        cbf_rec.fit(tiny_train_df)
        recs = cbf_rec.recommend(user_id=0, n=5, exclude_seen=set())
        assert len(recs) == len(set(recs))

    def test_recommend_excludes_seen_items(
        self,
        cbf_rec: ContentBasedRecommender,
        tiny_train_df: pl.DataFrame,
    ) -> None:
        cbf_rec.fit(tiny_train_df)
        seen = {100, 101}
        recs = cbf_rec.recommend(user_id=0, n=3, exclude_seen=seen)
        assert not any(r in seen for r in recs)

    def test_cold_start_user_returns_fallback(
        self,
        cbf_rec: ContentBasedRecommender,
        tiny_train_df: pl.DataFrame,
    ) -> None:
        cbf_rec.fit(tiny_train_df)
        # User 999 is not in training data → should not raise
        recs = cbf_rec.recommend(user_id=999, n=3, exclude_seen=set())
        assert len(recs) > 0  # Fallback provides recommendations

    def test_recommend_before_fit_raises(
        self,
        cbf_rec: ContentBasedRecommender,
    ) -> None:
        with pytest.raises(RuntimeError):
            cbf_rec.recommend(user_id=0, n=3, exclude_seen=set())

    def test_name_contains_tfidf_weight(
        self,
        cbf_rec: ContentBasedRecommender,
    ) -> None:
        assert "ContentBasedRecommender" in cbf_rec.name
