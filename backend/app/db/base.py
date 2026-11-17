"""SQLAlchemy declarative base.

All ORM models must inherit from Base defined here.
Import Base in alembic/env.py so migrations can discover all models.
"""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Shared declarative base for all SQLAlchemy ORM models."""
