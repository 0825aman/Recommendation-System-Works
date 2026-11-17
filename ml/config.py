"""Experiment configuration for Phase 3 baselines.

All configurable parameters live here. Pass an ExperimentConfig
instance to any baseline or evaluator to ensure reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Project root (two levels above this file: ml/config.py → ml/ → project root)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Default data paths
_PARQUET_DIR = _PROJECT_ROOT / "data" / "processed" / "kuairec" / "parquet"
_SPLITS_DIR = _PROJECT_ROOT / "data" / "processed" / "kuairec" / "splits"
_RESULTS_DIR = _PROJECT_ROOT / "experiments" / "results"


@dataclass
class ExperimentConfig:
    """Full configuration for a Phase 3 baseline experiment.

    All fields have sensible defaults. Override only what you need.

    Example:
        config = ExperimentConfig(random_seed=123)
    """

    # --- Dataset ---
    small_matrix_path: Path = field(default_factory=lambda: _PARQUET_DIR / "small_matrix.parquet")
    user_features_path: Path = field(default_factory=lambda: _PARQUET_DIR / "user_features.parquet")
    item_categories_path: Path = field(default_factory=lambda: _PARQUET_DIR / "item_categories.parquet")
    caption_category_path: Path = field(
        default_factory=lambda: _PARQUET_DIR / "kuairec_caption_category.parquet"
    )

    # --- Splits ---
    splits_dir: Path = field(default_factory=lambda: _SPLITS_DIR)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    # test_ratio is implied: 1 - train_ratio - val_ratio

    # --- Target label ---
    relevance_threshold: float = 0.5
    """watch_ratio threshold for binary relevant/non-relevant classification.

    Interactions with watch_ratio >= relevance_threshold are treated as relevant.
    Default 0.5 documented in ADR-P3-02.
    """

    # --- Evaluation ---
    k_values: list[int] = field(default_factory=lambda: [5, 10, 20])
    min_test_interactions: int = 1
    """Minimum number of relevant test interactions for a user to be evaluated."""

    # --- Algorithm parameters ---
    random_seed: int = 42
    popularity_metric: str = "weighted_score"
    """Popularity scoring method. Options: 'weighted_score', 'interaction_count'.
    Default 'weighted_score' = mean(watch_ratio) * log(1 + count). ADR-P3-05.
    """
    trending_window_days: int = 14
    """Recency window (days) for the trending recommender."""
    n_top_categories: int = 3
    """Number of top user-preferred categories used by category recommender."""

    # --- Experiment tracking ---
    results_dir: Path = field(default_factory=lambda: _RESULTS_DIR)
    phase: str = "3"
