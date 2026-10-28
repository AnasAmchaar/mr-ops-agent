"""
Async database session factory and Redis connection pool.
Both are initialized during FastAPI lifespan and torn down on shutdown.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings

# ── Module-level singletons (initialized via init_db / init_redis) ───
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None
_redis_pool: aioredis.Redis | None = None  # type: ignore[type-arg]


# ── SQLAlchemy Async Engine ──────────────────────────────────────────
def init_db() -> async_sessionmaker[AsyncSession]:
    """Create the async engine and session factory. Call once at startup."""
    global _engine, _session_factory
    settings = get_settings()
    _engine = create_async_engine(
        settings.database_url,
        echo=(not settings.is_production),
        pool_size=20,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=300,
    )
    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    return _session_factory


async def close_db() -> None:
    """Dispose engine connections. Call on shutdown."""
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency-injectable async session context manager."""
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ── Redis Async Pool ─────────────────────────────────────────────────
def init_redis() -> aioredis.Redis:  # type: ignore[type-arg]
    """Create a shared Redis connection pool. Call once at startup."""
    global _redis_pool
    settings = get_settings()
    _redis_pool = aioredis.from_url(
        settings.redis_url,
        decode_responses=True,
        max_connections=50,
    )
    return _redis_pool


async def close_redis() -> None:
    """Gracefully close the Redis pool."""
    global _redis_pool
    if _redis_pool is not None:
        await _redis_pool.aclose()
        _redis_pool = None


def get_redis() -> aioredis.Redis:  # type: ignore[type-arg]
    """Accessor for the shared Redis pool."""
    if _redis_pool is None:
        raise RuntimeError("Redis not initialized. Call init_redis() first.")
    return _redis_pool
