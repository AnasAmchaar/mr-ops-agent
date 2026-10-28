"""
Thread-safe ExecutionContext state manager.

Provides helpers for persisting and restoring execution state to/from
Redis and PostgreSQL. This enables:
- HITL suspension and resumption across server restarts
- Crash recovery for in-flight executions
- Multi-worker execution resumption (Arq workers can pick up suspended contexts)
"""
from __future__ import annotations

from typing import Optional

import redis.asyncio as aioredis
import structlog

from app.schemas.execution import ExecutionContext, ExecutionStatus

logger = structlog.get_logger(__name__)

# Redis key format for persisted execution state
_CONTEXT_KEY_PREFIX = "exec_ctx"


def _redis_key(tenant_id: str, execution_id: str) -> str:
    """Generate a tenant-scoped Redis key for execution context."""
    return f"{_CONTEXT_KEY_PREFIX}:{tenant_id}:{execution_id}"


class ExecutionContextManager:
    """
    Manages serialization and deserialization of ExecutionContext
    to Redis for suspension/resumption and crash recovery.
    """

    def __init__(self, redis_client: aioredis.Redis) -> None:  # type: ignore[type-arg]
        self._redis = redis_client

    async def save(
        self,
        ctx: ExecutionContext,
        *,
        ttl_seconds: int = 86400,  # 24 hours default (WhatsApp session window)
    ) -> None:
        """
        Persist an ExecutionContext to Redis.

        Args:
            ctx: The execution context to save
            ttl_seconds: Time-to-live in seconds. Defaults to 24h to match
                        WhatsApp's customer service window.
        """
        key = _redis_key(ctx.tenant_id, ctx.execution_id)
        serialized = ctx.model_dump_json()
        await self._redis.setex(key, ttl_seconds, serialized)
        logger.debug(
            "context_saved",
            execution_id=ctx.execution_id,
            tenant_id=ctx.tenant_id,
            status=ctx.status.value,
            ttl=ttl_seconds,
        )

    async def load(
        self,
        tenant_id: str,
        execution_id: str,
    ) -> Optional[ExecutionContext]:
        """
        Load an ExecutionContext from Redis.

        Returns:
            The deserialized context, or None if not found or expired.
        """
        key = _redis_key(tenant_id, execution_id)
        raw = await self._redis.get(key)
        if raw is None:
            logger.debug(
                "context_not_found",
                execution_id=execution_id,
                tenant_id=tenant_id,
            )
            return None

        ctx = ExecutionContext.model_validate_json(raw)
        logger.debug(
            "context_loaded",
            execution_id=execution_id,
            tenant_id=tenant_id,
            status=ctx.status.value,
            step_count=ctx.step_count,
        )
        return ctx

    async def delete(self, tenant_id: str, execution_id: str) -> None:
        """Remove a persisted execution context."""
        key = _redis_key(tenant_id, execution_id)
        await self._redis.delete(key)

    async def resume(
        self,
        tenant_id: str,
        execution_id: str,
        *,
        approval_data: Optional[dict] = None,
    ) -> Optional[ExecutionContext]:
        """
        Resume a suspended (HITL) execution.

        Args:
            tenant_id: Tenant scope
            execution_id: The suspended execution ID
            approval_data: Optional dict of approval variables to merge
                          (e.g. {"approved": True, "approver": "admin@..."})

        Returns:
            The resumed ExecutionContext ready for GraphRunner, or None
            if the context was not found or not in PAUSED_FOR_HUMAN state.
        """
        ctx = await self.load(tenant_id, execution_id)
        if ctx is None:
            return None

        if ctx.status != ExecutionStatus.PAUSED_FOR_HUMAN:
            logger.warning(
                "resume_invalid_state",
                execution_id=execution_id,
                current_status=ctx.status.value,
            )
            return None

        ctx.status = ExecutionStatus.RUNNING
        if approval_data:
            ctx.variables.update(approval_data)

        # Save the resumed state
        await self.save(ctx)

        logger.info(
            "execution_resumed",
            execution_id=execution_id,
            tenant_id=tenant_id,
        )
        return ctx

    async def find_active_session(
        self,
        tenant_id: str,
        session_id: str,
    ) -> Optional[ExecutionContext]:
        """
        Search for an active or paused execution for a given session.
        Uses a secondary Redis index key: `session:{tenant_id}:{session_id}`

        Returns the most recent active context, or None.
        """
        index_key = f"session:{tenant_id}:{session_id}"
        execution_id = await self._redis.get(index_key)
        if execution_id is None:
            return None
        return await self.load(tenant_id, execution_id)

    async def index_session(
        self,
        ctx: ExecutionContext,
        ttl_seconds: int = 86400,
    ) -> None:
        """
        Create a secondary index mapping session_id -> execution_id
        for quick session lookup (e.g., WhatsApp phone number -> active execution).
        """
        index_key = f"session:{ctx.tenant_id}:{ctx.session_id}"
        await self._redis.setex(index_key, ttl_seconds, ctx.execution_id)

    async def clear_session_index(
        self,
        tenant_id: str,
        session_id: str,
    ) -> None:
        """Remove the session -> execution_id index."""
        index_key = f"session:{tenant_id}:{session_id}"
        await self._redis.delete(index_key)
