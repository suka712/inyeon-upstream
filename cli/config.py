from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    CLI settings loaded from environment variables.

    Prefix: INYEON_
    Example: INYEON_API_URL=http://localhost:8000
    """

    model_config = SettingsConfigDict(
        env_prefix="INYEON_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Backend API
    api_url: str = "https://inyeon-upstream-production.up.railway.app"
    timeout: int = 120

    # Output preferences
    default_format: str = "rich"  # rich, json, plain


def get_config_file() -> Path | None:
    """
    Find config file in order of priority.

    Checks:
        1. ./.inyeon.toml (project-level)
        2. ~/.config/inyeon/config.toml (user-level)
    """
    paths = [
        Path.cwd() / ".inyeon.toml",
        Path.home() / ".config" / "inyeon" / "config.toml",
    ]

    for path in paths:
        if path.exists():
            return path

    return None


# Global settings instance
settings = Settings()
