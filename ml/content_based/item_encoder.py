"""Item feature encoder for Content-Based Filtering.

Builds a numerical feature matrix for items using KuaiRec 2.0 metadata:
  - One-hot encoded first_level_category_name  (20 categories)
  - One-hot encoded second_level_category_name (138 categories)
  - Multi-hot encoded feat_parsed IDs from item_categories.parquet
  - TF-IDF weighted caption embeddings (87% coverage, zeros for missing)

The resulting matrix is L2-normalised per row so that cosine similarity
equals the dot product between any two row vectors.

Usage:
    from ml.content_based.item_encoder import build_item_feature_matrix

    id_to_idx, idx_to_id, matrix = build_item_feature_matrix(
        item_categories_df=item_categories_df,
        caption_category_df=caption_category_df,
    )
"""

from __future__ import annotations

import math

import polars as pl

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _l2_norm(vec: list[float]) -> float:
    """Return the L2 norm of a vector."""
    return math.sqrt(sum(x * x for x in vec))


def _normalize_rows(matrix: list[list[float]]) -> list[list[float]]:
    """L2-normalize each row in-place and return the matrix."""
    for i, row in enumerate(matrix):
        norm = _l2_norm(row)
        if norm > 0.0:
            matrix[i] = [x / norm for x in row]
    return matrix


def _one_hot_encode(
    values: list[str | None],
    categories: list[str],
) -> list[list[float]]:
    """One-hot encode a list of category values.

    Args:
        values:     Per-item category string (or None for missing).
        categories: Sorted list of all known categories (determines column order).

    Returns:
        Matrix of shape (n_items, n_categories) with 1.0 at the matching index.
    """
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    n_cats = len(categories)
    result = []
    for v in values:
        row = [0.0] * n_cats
        if v is not None and v in cat_to_idx:
            row[cat_to_idx[v]] = 1.0
        result.append(row)
    return result


def _multi_hot_encode(
    feat_lists: list[list[int] | None],
    all_feat_ids: list[int],
) -> list[list[float]]:
    """Multi-hot encode lists of integer feature IDs.

    Args:
        feat_lists:   Per-item list of feature IDs (or None).
        all_feat_ids: Sorted list of all known feature IDs.

    Returns:
        Matrix of shape (n_items, n_feat_ids) with 1.0 for each present ID.
    """
    feat_to_idx = {f: i for i, f in enumerate(all_feat_ids)}
    n_feats = len(all_feat_ids)
    result = []
    for flist in feat_lists:
        row = [0.0] * n_feats
        if flist:
            for fid in flist:
                if fid in feat_to_idx:
                    row[feat_to_idx[fid]] = 1.0
        result.append(row)
    return result


def _tfidf_encode(
    captions: list[str | None],
    max_features: int = 500,
    min_df: int = 2,
) -> list[list[float]]:
    """TF-IDF encode caption strings using scikit-learn TfidfVectorizer.

    Items with None captions receive a zero vector.

    Args:
        captions:     Per-item caption string or None.
        max_features: Maximum vocabulary size.
        min_df:       Minimum document frequency for a term to be included.

    Returns:
        Dense matrix of shape (n_items, vocab_size).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Replace None with empty string; TF-IDF will produce zero vector
    texts = [c if c is not None else "" for c in captions]

    # Only fit on non-empty texts so vocabulary reflects real content
    non_empty = [t for t in texts if t.strip()]
    if not non_empty:
        # All captions missing — return zero matrix
        return [[0.0] * max_features for _ in captions]

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=True,  # Apply log(1 + tf) instead of raw tf
        strip_accents="unicode",
        analyzer="char_wb",  # Character n-grams — robust for CJK text
        ngram_range=(2, 4),
    )
    # Fit on non-empty, transform on all (empty → zero vector)
    vectorizer.fit(non_empty)
    tfidf_sparse = vectorizer.transform(texts)

    # Convert sparse matrix to list-of-lists
    tfidf_array = tfidf_sparse.toarray()
    return [list(row) for row in tfidf_array]


def _scale_and_concat(
    blocks: list[tuple[list[list[float]], float]],
) -> list[list[float]]:
    """Concatenate weighted feature blocks into a single matrix.

    Args:
        blocks: List of (matrix, weight) pairs. All matrices must have the
                same number of rows. Each matrix row is multiplied by weight
                before concatenation.

    Returns:
        Concatenated matrix of shape (n_items, sum_of_col_counts).
    """
    if not blocks:
        return []
    n_rows = len(blocks[0][0])
    rows = [[] for _ in range(n_rows)]
    for matrix, weight in blocks:
        for i, row in enumerate(matrix):
            rows[i].extend(w * weight for w in row)
    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_item_feature_matrix(
    item_categories_df: pl.DataFrame,
    caption_category_df: pl.DataFrame,
    category_weight: float = 1.0,
    tfidf_weight: float = 0.5,
    normalize: bool = True,
    tfidf_max_features: int = 500,
) -> tuple[dict[int, int], dict[int, int], list[list[float]]]:
    """Build the item feature matrix from KuaiRec metadata.

    Feature groups (concatenated in order):
      1. One-hot: first_level_category_name × category_weight
      2. One-hot: second_level_category_name × category_weight
      3. Multi-hot: feat_parsed IDs × category_weight
      4. TF-IDF: caption × tfidf_weight  (zeros for missing captions)

    Args:
        item_categories_df:   DataFrame with columns video_id, feat, feat_parsed.
        caption_category_df:  DataFrame with columns video_id, caption,
                              first_level_category_name,
                              second_level_category_name.
        category_weight:      Scaling factor for categorical features.
        tfidf_weight:         Scaling factor for TF-IDF caption features.
                              Set to 0.0 to disable caption features entirely.
        normalize:            If True, L2-normalize each row vector.
        tfidf_max_features:   Maximum TF-IDF vocabulary size.

    Returns:
        Tuple of:
          id_to_idx (dict[int, int]):  video_id → row index in matrix
          idx_to_id (dict[int, int]):  row index → video_id
          matrix    (list[list[float]]): Shape (n_items, n_features)
    """
    # ------------------------------------------------------------------
    # 1. Merge item_categories and caption_category on video_id
    # ------------------------------------------------------------------
    # caption_category_df has the category name columns we need
    merged = caption_category_df.select(
        ["video_id", "caption", "first_level_category_name", "second_level_category_name"]
    ).join(
        item_categories_df.select(["video_id", "feat_parsed"]),
        on="video_id",
        how="left",
    )

    # Deterministic ordering by video_id for reproducibility
    merged = merged.sort("video_id")

    video_ids: list[int] = merged["video_id"].to_list()
    n_items = len(video_ids)

    id_to_idx: dict[int, int] = {vid: i for i, vid in enumerate(video_ids)}
    idx_to_id: dict[int, int] = {i: vid for i, vid in enumerate(video_ids)}

    # ------------------------------------------------------------------
    # 2. Extract column values as Python lists
    # ------------------------------------------------------------------
    first_cats: list[str | None] = merged["first_level_category_name"].to_list()
    second_cats: list[str | None] = merged["second_level_category_name"].to_list()
    feat_lists: list[list[int] | None] = merged["feat_parsed"].to_list()
    captions: list[str | None] = merged["caption"].to_list()

    # ------------------------------------------------------------------
    # 3. Encode each feature group
    # ------------------------------------------------------------------
    # a) First-level category one-hot
    all_first = sorted({c for c in first_cats if c is not None})
    oh_first = _one_hot_encode(first_cats, all_first)

    # b) Second-level category one-hot
    all_second = sorted({c for c in second_cats if c is not None})
    oh_second = _one_hot_encode(second_cats, all_second)

    # c) feat_parsed multi-hot
    all_feat_ids: list[int] = sorted(
        {fid for flist in feat_lists if flist for fid in flist}
    )
    mh_feat = _multi_hot_encode(feat_lists, all_feat_ids)

    # Build feature blocks list
    feature_blocks: list[tuple[list[list[float]], float]] = [
        (oh_first, category_weight),
        (oh_second, category_weight),
        (mh_feat, category_weight),
    ]

    # d) TF-IDF captions (optional)
    if tfidf_weight > 0.0 and tfidf_max_features > 0:
        tfidf_matrix = _tfidf_encode(captions, max_features=tfidf_max_features)
        # Pad or trim to tfidf_max_features columns if needed
        actual_cols = len(tfidf_matrix[0]) if tfidf_matrix else 0
        if actual_cols > 0:
            feature_blocks.append((tfidf_matrix, tfidf_weight))

    # ------------------------------------------------------------------
    # 4. Concatenate blocks
    # ------------------------------------------------------------------
    if not feature_blocks:
        # Fallback: identity matrix (should not happen with valid data)
        matrix = [[1.0 if i == j else 0.0 for j in range(n_items)] for i in range(n_items)]
    else:
        matrix = _scale_and_concat(feature_blocks)

    # ------------------------------------------------------------------
    # 5. Optionally L2-normalize rows
    # ------------------------------------------------------------------
    if normalize:
        matrix = _normalize_rows(matrix)

    return id_to_idx, idx_to_id, matrix


def get_feature_dim(
    item_categories_df: pl.DataFrame,
    caption_category_df: pl.DataFrame,
    tfidf_weight: float = 0.5,
    tfidf_max_features: int = 500,
) -> int:
    """Return the feature dimensionality without building the full matrix.

    Useful for validation and debugging.
    """
    all_first = caption_category_df["first_level_category_name"].drop_nulls().unique().len()
    all_second = caption_category_df["second_level_category_name"].drop_nulls().unique().len()

    feat_lists: list[list[int] | None] = item_categories_df["feat_parsed"].to_list()
    n_feat_ids = len({fid for flist in feat_lists if flist for fid in flist})

    tfidf_dim = tfidf_max_features if tfidf_weight > 0.0 else 0
    return all_first + all_second + n_feat_ids + tfidf_dim
