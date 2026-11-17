"""Abstract base class for all Phase 3 baseline recommenders.

Every recommender in ml/baselines/ must implement this interface.
This ensures the evaluation framework can call any recommender
through the same contract.

Design:
    - fit(train_df)   : build all indexes/statistics from training data
    - recommend(...)  : return a ranked list of item IDs

Constraints:
    - No FastAPI, SQLAlchemy, or backend imports allowed in this module
    - All recommenders must accept a random_seed for reproducibility
    - recommend() must respect the exclude_seen set
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import polars as pl


class BaseRecommender(ABC):
    """Abstract recommender interface.

    All baseline recommenders in Phase 3 must subclass this.

    Subclass contract:
        1. Call super().__init__(random_seed) in __init__.
        2. Implement fit(train_df) to build all indexes.
        3. Implement recommend(...) to return ranked item IDs.
        4. Set self._fitted = True at the end of fit().
        5. Call self._check_fitted() at the start of recommend().
    """

    def __init__(self, random_seed: int = 42) -> None:
        """Initialise the recommender.

        Args:
            random_seed: Seed for any random operations. Ensures reproducibility.
        """
        self.random_seed = random_seed
        self._fitted: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name used in experiment records."""
        ...

    @abstractmethod
    def fit(self, train_df: pl.DataFrame) -> None:
        """Build all indexes, statistics, and lookup tables from training data.

        Args:
            train_df: Training split of the interaction matrix.
                Expected columns: user_id (Int64), video_id (Int64),
                watch_ratio (Float64). May include time, date, timestamp.

        Side-effects:
            Sets self._fitted = True when complete.
        """
        ...

    @abstractmethod
    def recommend(
        self,
        user_id: int,
        n: int,
        exclude_seen: set[int],
    ) -> list[int]:
        """Return a ranked list of recommended item IDs for a user.

        Args:
            user_id:      Target user identifier.
            n:            Number of recommendations to return.
            exclude_seen: Set of item IDs already seen by this user
                          (from training data). These must NOT appear in output.

        Returns:
            Ordered list of video_id integers, best first.
            Length <= n (may be shorter if catalog is exhausted after exclusion).

        Raises:
            RuntimeError: If recommend() is called before fit().
        """
        ...

    def _check_fitted(self) -> None:
        """Raise if recommend() is called before fit()."""
        if not self._fitted:
            raise RuntimeError(
                f"{self.name}: recommend() called before fit(). "
                "Call fit(train_df) first."
            )

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return f"{self.__class__.__name__}(seed={self.random_seed}, {status})"
