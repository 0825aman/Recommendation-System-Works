"""Application configuration via Pydantic Settings.

Values are read from environment variables and .env file.
Never log or expose the values of these settings.
"""

from __future__ import annotations

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application configuration.

    All values are loaded from environment variables.
    Sensitive values must never appear in logs or API responses.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_env: str = Field(default="development", description="Environment name")
    app_version: str = Field(default="0.1.0", description="Application version")
    secret_key: str = Field(
        default="change-me-in-production",
        description="Secret key — never expose in responses or logs",
    )

    # Backend server
    backend_host: str = Field(default="0.0.0.0", description="Uvicorn bind host")
    backend_port: int = Field(default=8000, description="Uvicorn bind port")
    backend_log_level: str = Field(default="INFO", description="Log level")

    # Database
    db_host: str = Field(default="localhost", description="PostgreSQL host")
    db_port: int = Field(default=5432, description="PostgreSQL port")
    db_name: str = Field(default="recommendation_db", description="Database name")
    db_user: str = Field(default="recommendation_user", description="Database user")
    db_password: str = Field(
        default="change-me",
        description="Database password — never expose",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url(self) -> str:
        """Compose the SQLAlchemy database URL from individual fields.

        Uses the postgresql+psycopg2 dialect explicitly so SQLAlchemy 2.x
        uses psycopg2-binary (not psycopg v3).
        """
        return (
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_development(self) -> bool:
        """True when running in development mode."""
        return self.app_env.lower() == "development"


def get_settings() -> Settings:
    """Return application settings instance.

    Import and call this function to access settings throughout the app.
    Use FastAPI Depends(get_settings) for dependency injection in routes.
    """
    return Settings()
