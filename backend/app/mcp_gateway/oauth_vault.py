"""
OAuth Vault — tenant-scoped Google OAuth2 credential management.

Responsibilities:
- Load the tenant's AES-256-GCM sealed refresh token from PostgreSQL.
- Exchange (and cache) short-lived access tokens via Google's token endpoint.
- Redis caching keyed strictly by tenant_id + integration type, with TTL
  slightly below Google's token lifetime (access tokens live ~3600s).
- Detect permanent auth failures (revoked/grant deleted → 400
  `invalid_grant`) and mark the TenantIntegration row ERROR/REVOKED so
  users are forced through re-authorization in the UI.

Security invariants:
- Plaintext refresh tokens NEVER leave this module, are never logged,
  and are never persisted anywhere unencrypted.
- Every lookup is scoped by tenant_id; cross-tenant access is impossible.
- All I/O is async (SQLAlchemy async session, httpx, redis.asyncio).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import httpx
import redis.asyncio as aioredis
import structlog
from sqlalchemy import select

from app.core.config import get_settings
from app.core.database import get_db_session, get_redis
from app.core.security import InvalidAuth, decrypt_refresh_token
from app.models.integration import IntegrationStatus, TenantIntegration

logger = structlog.get_logger(__name__)

GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"

# Refresh the cached access token this many seconds before its real expiry
_ACCESS_TOKEN_CLOCK_SKEW = 60
# Redis key namespace — tenant-scoped
_REDIS_KEY = "oauth:access_token:{tenant_id}:{integration_type}"


class OAuthVaultError(Exception):
    """Base class for vault failures reported to MCP tool executors."""


class IntegrationNotConnected(OAuthVaultError):
    """Tenant has not completed OAuth for this integration."""


class GrantInvalid(OAuthVaultError):
    """Refresh token revoked / expired — tenant must re-authorize."""


class GoogleAuthAPIError(OAuthVaultError):
    """Transient upstream failure from Google's token endpoint."""


class AccessToken:
    """Typed wrapper for a valid Google access token."""

    def __init__(self, token: str, expires_at: datetime, scopes: str) -> None:
        self.value = token
        self.expires_at = expires_at
        self.scopes = scopes

    @property
    def expires_in_seconds(self) -> int:
        return max(0, int((self.expires_at - datetime.now(timezone.utc)).total_seconds()))


class OAuthVault:
    """
    Retrieves Google access tokens for a tenant's connected integrations.

    Usage:
        vault = get_oauth_vault()
        access_token = await vault.get_access_token(tenant_id, "google_calendar")
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._redis: Optional[aioredis.Redis] = None

    def _get_redis(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = get_redis()
        return self._redis

    # ── Public API ───────────────────────────────────────────────────

    async def get_access_token(
        self,
        tenant_id: str,
        integration_type: str,
        *,
        force_refresh: bool = False,
    ) -> AccessToken:
        """
        Return a valid access token for the tenant's integration.

        Resolution order:
        1. Redis cache hit (not within clock-skew of expiry)  → return.
        2. DB refresh token → decrypt → POST /token → cache + return.

        Raises:
            IntegrationNotConnected: tenant never completed OAuth.
            GrantInvalid: refresh token revoked — re-auth required.
            GoogleAuthAPIError: transient upstream failure.
        """
        # 1. Redis cache
        if not force_refresh:
            cached = await self._read_cached_token(tenant_id, integration_type)
            if cached is not None:
                return cached

        # 2. Refresh flow
        return await self._refresh_access_token(tenant_id, integration_type)

    async def mark_revoked(self, tenant_id: str, integration_type: str) -> None:
        """Mark the integration REVOKED and purge cache (e.g. user disconnect)."""
        await self._set_integration_status(tenant_id, integration_type, IntegrationStatus.REVOKED)
        await self._invalidate_cache(tenant_id, integration_type)

    async def invalidate(self, tenant_id: str, integration_type: str) -> None:
        """Purge the Redis-cached access token (e.g. scopes changed)."""
        await self._invalidate_cache(tenant_id, integration_type)

    # ── Internal: Redis cache ────────────────────────────────────────

    async def _read_cached_token(
        self, tenant_id: str, integration_type: str
    ) -> Optional[AccessToken]:
        key = _REDIS_KEY.format(tenant_id=tenant_id, integration_type=integration_type)
        try:
            blob = await self._get_redis().hgetall(key)
        except Exception as exc:  # Redis unavailability must not kill the request path
            logger.warning("oauth_cache_read_failed", error=str(exc))
            return None
        if not blob:
            return None

        token, iat, ttl_str = blob.get("token"), blob.get("iat"), blob.get("ttl")
        scopes = blob.get("scopes", "")
        if not token or not iat or not ttl_str:
            return None

        expires_at = datetime.fromtimestamp(float(iat) + float(ttl_str), tz=timezone.utc)
        # Return only if meaningfully valid (above skew window)
        if (expires_at - datetime.now(timezone.utc)).total_seconds() <= _ACCESS_TOKEN_CLOCK_SKEW:
            return None
        return AccessToken(token=token, expires_at=expires_at, scopes=scopes)

    async def _cache_token(
        self,
        tenant_id: str,
        integration_type: str,
        token: str,
        ttl_seconds: int,
        scopes: str,
    ) -> None:
        key = _REDIS_KEY.format(tenant_id=tenant_id, integration_type=integration_type)
        now_ts = datetime.now(timezone.utc).timestamp()
        cache_ttl = max(1, int(ttl_seconds) - _ACCESS_TOKEN_CLOCK_SKEW)
        try:
            pipe = self._get_redis().pipeline()
            pipe.hset(key, mapping={
                "token": token,
                "iat": now_ts,
                "ttl": ttl_seconds,
                "scopes": scopes,
                "tenant_id": tenant_id,  # defense-in-depth audit field
            })
            pipe.expire(key, cache_ttl)
            await pipe.execute()
        except Exception as exc:
            logger.warning("oauth_cache_write_failed", error=str(exc))

    async def _invalidate_cache(self, tenant_id: str, integration_type: str) -> None:
        key = _REDIS_KEY.format(tenant_id=tenant_id, integration_type=integration_type)
        try:
            await self._get_redis().delete(key)
        except Exception as exc:
            logger.warning("oauth_cache_invalidate_failed", error=str(exc))

    # ── Internal: DB token resolution & refresh ──────────────────────

    async def _load_refresh_token(
        self, tenant_id: str, integration_type: str
    ) -> tuple[str, str]:
        """
        Load and unseal the stored refresh token.

        Returns:
            (plaintext_refresh_token, granted_scopes)

        Raises:
            IntegrationNotConnected
            InvalidAuth propagation for tampered ciphertexts
        """
        row = await self._find_integration_row(tenant_id, integration_type)
        if row is None or row.status not in (IntegrationStatus.CONNECTED,):
            raise IntegrationNotConnected(
                f"Integration '{integration_type}' is not connected for this tenant."
            )
        if not row.encrypted_refresh_token:
            raise IntegrationNotConnected(
                f"Integration '{integration_type}' exists but holds no refresh token."
            )
        plaintext = decrypt_refresh_token(row.encrypted_refresh_token)
        return plaintext, row.granted_scopes

    async def _find_integration_row(
        self, tenant_id: str, integration_type: str
    ) -> Optional[TenantIntegration]:
        tenant_uuid = _parse_uuid(tenant_id)
        itype = IntegrationType(integration_type)
        async with get_db_session() as session:
            stmt = (
                select(TenantIntegration)
                .where(TenantIntegration.tenant_id == tenant_uuid)
                .where(TenantIntegration.integration_type == itype)
                .order_by(TenantIntegration.created_at.desc())
                .limit(1)
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def _refresh_access_token(
        self, tenant_id: str, integration_type: str
    ) -> AccessToken:
        settings = self._settings
        refresh_token, granted_scopes = await self._load_refresh_token(tenant_id, integration_type)

        form: Dict[str, str] = {
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(GOOGLE_TOKEN_URL, data=form)

        if resp.status_code == 400:
            detail = _extract_error_detail(resp)
            # invalid_grant: token revoked or expired — permanent failure
            await self._set_integration_status(
                tenant_id, integration_type,
                IntegrationStatus.REVOKED, error_detail=detail or "invalid_grant",
            )
            await self._invalidate_cache(tenant_id, integration_type)
            raise GrantInvalid(
                f"Google revoked the refresh token for '{integration_type}'. "
                "Tenant must re-authorize."
            )

        if resp.status_code != 200:
            await self._set_integration_status(
                tenant_id, integration_type,
                IntegrationStatus.ERROR,
                error_detail=f"HTTP {resp.status_code}: {resp.text[:200]}",
            )
            raise GoogleAuthAPIError(
                f"Google token endpoint returned HTTP {resp.status_code}"
            )

        body: Dict[str, Any] = resp.json()
        access_token = body.get("access_token")
        expires_in = int(body.get("expires_in", 3600))
        scope = body.get("scope", granted_scopes)
        if not access_token:
            raise GoogleAuthAPIError("Google token response missing access_token field.")

        expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

        await self._cache_token(
            tenant_id=tenant_id,
            integration_type=integration_type,
            token=access_token,
            ttl_seconds=expires_in,
            scopes=scope,
        )
        await self._record_refresh_success(tenant_id, integration_type)

        logger.info(
            "oauth_token_refreshed",
            tenant_id=tenant_id,
            integration_type=integration_type,
            expires_in=expires_in,
        )
        return AccessToken(token=access_token, expires_at=expires_at, scopes=scope)

    async def _record_refresh_success(self, tenant_id: str, integration_type: str) -> None:
        await self._update_integration_row(
            tenant_id, integration_type,
            status=IntegrationStatus.CONNECTED,
            last_refreshed_at=datetime.now(timezone.utc),
            last_refresh_error=None,
        )

    async def _set_integration_status(
        self,
        tenant_id: str,
        integration_type: str,
        status: IntegrationStatus,
        *,
        error_detail: Optional[str] = None,
    ) -> None:
        await self._update_integration_row(
            tenant_id,
            integration_type,
            status=status,
            last_refresh_error=error_detail,
        )
        logger.warning(
            "oauth_integration_status_changed",
            tenant_id=tenant_id,
            integration_type=integration_type,
            status=status.value,
            detail=error_detail,
        )

    async def _update_integration_row(
        self,
        tenant_id: str,
        integration_type: str,
        *,
        status: Optional[IntegrationStatus] = None,
        last_refreshed_at: Optional[datetime] = None,
        last_refresh_error: Optional[str] = None,
        clear_refresh_error: bool = False,
    ) -> None:
        """Column-wise update on exactly one tenant-scoped integration row."""
        row = await self._find_integration_row(tenant_id, integration_type)
        if row is None:
            return
        async with get_db_session() as session:
            stmt = select(TenantIntegration).where(TenantIntegration.id == row.id)
            db_row = (await session.execute(stmt)).scalar_one_or_none()
            if db_row is None:
                return
            if status is not None:
                db_row.status = status
            if last_refreshed_at is not None:
                db_row.last_refreshed_at = last_refreshed_at
            if last_refresh_error is not None:
                db_row.last_refresh_error = last_refresh_error
            if clear_refresh_error:
                db_row.last_refresh_error = None


def _extract_error_detail(resp: httpx.Response) -> Optional[str]:
    try:
        return resp.json().get("error")
    except Exception:
        return resp.text[:200] if resp.text else None


def _parse_uuid(value: str):
    import uuid as uuid_mod
    return uuid_mod.UUID(value)


# ── Singleton ────────────────────────────────────────────────────────

_vault: OAuthVault | None = None


def get_oauth_vault() -> OAuthVault:
    """Accessor for the process-wide OAuthVault singleton."""
    global _vault
    if _vault is None:
        _vault = OAuthVault()
    return _vault
