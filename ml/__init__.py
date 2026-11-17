"""Hybrid Recommendation Intelligence Platform — ML module.

All recommendation logic, feature engineering, evaluation, and experiments
live in this package. This module is fully standalone — no FastAPI, no
SQLAlchemy, no backend imports.

Usage:
    from ml.baselines.base import BaseRecommender
    from ml.evaluation.metrics import precision_at_k
"""
