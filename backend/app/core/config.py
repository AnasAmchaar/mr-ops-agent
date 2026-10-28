"""
Pydantic Settings — single source of truth for all runtime configuration.
Loaded from environment variables or `.env` file. Every subsystem imports from here.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────
    app_env: str = Field(default="development")
    log_level: str = Field(default="INFO")
    cors_origins: List[str] = Field(default=["http://localhost:3000"])

    # ── Database ─────────────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql+asyncpg://mrops:password@localhost:5432/mrops"
    )
    redis_url: str = Field(default="redis://localhost:6379/0")

    # ── LLM Providers ────────────────────────────────────────────────
    groq_api_key: str = Field(default="")
    openrouter_api_key: str = Field(default="")

    # ── WhatsApp Cloud API ───────────────────────────────────────────
    whatsapp_verify_token: str = Field(default="mrops-verify-token")
    whatsapp_api_version: str = Field(default="v21.0")
    meta_app_secret: str = Field(default="")

    # ── Google OAuth2 ────────────────────────────────────────────────
    google_client_id: str = Field(default="")
    google_client_secret: str = Field(default="")
    google_redirect_uri: str = Field(
        default="http://localhost:8000/api/v1/integrations/google/callback"
    )

    # ── Security ─────────────────────────────────────────────────────
    encryption_key: str = Field(
        default="0" * 64,
        description="64-char hex string → 32 bytes for AES-256-GCM",
    )
    jwt_secret: str = Field(default="change-me-in-production")
    jwt_algorithm: str = Field(default="HS256")
    jwt_expiry_minutes: int = Field(default=60)

    # ── Derived Properties ───────────────────────────────────────────
    @property
    def encryption_key_bytes(self) -> bytes:
        return bytes.fromhex(self.encryption_key)

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors(cls, v: str | List[str]) -> List[str]:
        if isinstance(v, str):
            return json.loads(v)
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton accessor — cached after first call."""
    return Settings()
