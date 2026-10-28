"""
Security Primitives — AES-256-GCM at-rest encryption and JWT session tokens.

Responsibilities:
- Encrypt/decrypt tenant OAuth refresh tokens at rest (never plaintext on disk).
- Issue/validate short-lived JWT access tokens for frontend/API sessions.
- Key material sourced from `Settings.encryption_key` (64-char hex -> 32 bytes).

Design notes:
- AESGCM (Galois/Counter Mode) provides authenticated encryption: tampering
  with ciphertext, nonce, or associated data fails decryption with InvalidTag.
- A fresh 96-bit random nonce is generated per encryption call, so identical
  plaintexts never produce identical ciphertexts.
- JWT tokens are always verified (signature + expiry + audience) before trust.
"""
from __future__ import annotations

import base64
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
import structlog
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import BaseModel, Field

from app.core.config import get_settings

logger = structlog.get_logger(__name__)

_NONCE_LENGTH = 12  # 96-bit nonce — recommended for AESGCM
_AAD = b"mr-ops:v1:refresh_token"


# ── AES-256-GCM Seal / Unseal ────────────────────────────────────────

def _get_aesgcm() -> AESGCM:
    key = get_settings().encryption_key_bytes
    if len(key) != 32:
        raise RuntimeError(
            f"ENCRYPTION_KEY must decode to exactly 32 bytes for AES-256, got {len(key)}. "
            "Provide a 64-char hex string."
        )
    return AESGCM(key)


def encrypt_refresh_token(plaintext: str) -> str:
    """
    Encrypt a token with AES-256-GCM.

    Output layout: base64(nonce[12] || ciphertext_with_tag).
    """
    nonce = os.urandom(_NONCE_LENGTH)
    ciphertext = _get_aesgcm().encrypt(
        nonce,
        plaintext.encode("utf-8"),
        associated_data=_AAD,
    )
    return base64.b64encode(nonce + ciphertext).decode("ascii")


def decrypt_refresh_token(sealed: str) -> str:
    """
    Decrypt an AES-256-GCM sealed token.

    Raises:
        InvalidAuth: tampered ciphertext, wrong key, or malformed blob.
    """
    try:
        raw = base64.b64decode(sealed, validate=True)
    except ValueError as exc:
        raise InvalidAuth("Sealed blob is not valid base64.") from exc

    if len(raw) <= _NONCE_LENGTH:
        raise InvalidAuth("Sealed blob too short to contain nonce + ciphertext.")

    nonce, ciphertext = raw[:_NONCE_LENGTH], raw[_NONCE_LENGTH:]
    try:
        plaintext_bytes = _get_aesgcm().decrypt(nonce, ciphertext, associated_data=_AAD)
    except InvalidTag:
        logger.warning("token_decryption_failed", reason="invalid_tag")
        raise InvalidAuth(
            "Ciphertext authentication failed — tampered or wrong key."
        ) from None

    return plaintext_bytes.decode("utf-8")


# Module-level exception type — `decrypt_refresh_token` resolves it at call time.
class InvalidAuth(Exception):
    """Raised when authenticated decryption fails. Importable by callers."""

    def __init__(self, detail: str = "Authentication failed.") -> None:
        super().__init__(detail)
        self.detail = detail


# ── JWT Session Tokens ───────────────────────────────────────────────

class JWTPayload(BaseModel):
    """Strictly typed validated JWT claims after successful verification."""
    sub: str = Field(..., min_length=1, description="Subject — tenant owner / user ID (UUID)")
    tenant_id: str = Field(..., min_length=1, description="Multi-tenant scope key (UUID)")
    exp: int = Field(..., gt=0)
    iat: int
    iss: str
    aud: str


def _assert_jwt_key_strength(settings: Any) -> None:
    """HMAC keys below 256 bits are rejected in production (RFC 7518 §3.2)."""
    if settings.is_production and len(settings.jwt_secret.encode("utf-8")) < 32:
        raise RuntimeError("JWT_SECRET must be at least 32 bytes in production.")


def issue_jwt(*, subject: str, tenant_id: str) -> str:
    """Issue a signed short-lived access token for API sessions."""
    settings = get_settings()
    _assert_jwt_key_strength(settings)
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "tenant_id": tenant_id,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=settings.jwt_expiry_minutes)).timestamp()),
        "iss": "mr-ops",
        "aud": "mr-ops-api",
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def verify_jwt(token: str) -> JWTPayload:
    """
    Verify signature, expiration, audience, and validate claims strictly.

    Raises:
        jwt.InvalidTokenError: invalid signature / expired / wrong aud-iss.
    """
    settings = get_settings()
    _assert_jwt_key_strength(settings)
    claims: dict[str, Any] = jwt.decode(
        token,
        settings.jwt_secret,
        algorithms=[settings.jwt_algorithm],
        audience="mr-ops-api",
        issuer="mr-ops",
        require=["sub", "tenant_id", "exp", "iat", "iss", "aud"],
    )
    return JWTPayload.model_validate(claims)
