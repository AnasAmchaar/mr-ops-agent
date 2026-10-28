"""
SQLAlchemy 2.0 Typed Models — integration credentials and tenant scoping.

TenantIntegration stores one row per connected OAuth2 integration
(Google Calendar / Gmail / Meet) per tenant. Refresh tokens are stored
AES-256-GCM sealed — never plaintext. All queries MUST scope by tenant_id.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import DateTime, Enum as SAEnum, String, Text, Uuid
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all models."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class IntegrationStatus(str, Enum):
    PENDING = "PENDING"      # OAuth started, callback not completed
    CONNECTED = "CONNECTED"  # refresh token stored and valid
    REVOKED = "REVOKED"      # user disconnected or Google revoked access
    ERROR = "ERROR"          # refresh persistently failing (e.g. re-auth required)


class IntegrationType(str, Enum):
    GOOGLE_CALENDAR = "google_calendar"
    GOOGLE_MEET = "google_meet"
    GMAIL = "gmail"


class TenantIntegration(Base):
    """A tenant's connected OAuth2 integration with sealed refresh token."""

    __tablename__ = "tenant_integrations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, index=True, nullable=False, doc="Multi-tenant scope key — every query filters on this"
    )
    integration_type: Mapped[IntegrationType] = mapped_column(
        SAEnum(IntegrationType, name="integration_type_enum", native_enum=False),
        nullable=False,
    )
    status: Mapped[IntegrationStatus] = mapped_column(
        SAEnum(IntegrationStatus, name="integration_status_enum", native_enum=False),
        default=IntegrationStatus.PENDING,
        nullable=False,
    )

    # AES-256-GCM sealed refresh token (app.core.security sealing format).
    # Empty for PENDING rows; populated on successful OAuth callback.
    encrypted_refresh_token: Mapped[str] = mapped_column(Text, default="", nullable=False)

    # Incrementally granted scopes as a space-joined string (Google format)
    granted_scopes: Mapped[str] = mapped_column(Text, default="", nullable=False)

    # Row version for optimistic concurrency on token rotation
    token_version: Mapped[int] = mapped_column(default=0, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )
    last_refreshed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_refresh_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Google-unique per account email for human identification in UI
    connected_email: Mapped[str | None] = mapped_column(String(320), nullable=True)

    def __repr__(self) -> str:
        return (
            f"<TenantIntegration tenant={self.tenant_id} "
            f"type={self.integration_type} status={self.status}>"
        )
