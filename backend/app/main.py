"""
FastAPI Application Entrypoint & Lifespan Manager.

Initializes all infrastructure connections (PostgreSQL, Redis, LLM clients)
on startup and tears them down gracefully on shutdown. API routes are
registered from versioned router modules.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.database import close_db, close_redis, init_db, init_redis
from app.llm.gateway import LLMGateway
from app.mcp_gateway.client_manager import (
    close_mcp_client_manager,
    init_mcp_client_manager,
)

logger = structlog.get_logger(__name__)

# ── Module-level singleton for LLM Gateway ───────────────────────────
_llm_gateway: LLMGateway | None = None


def get_llm_gateway() -> LLMGateway:
    """Accessor for the shared LLM Gateway instance."""
    if _llm_gateway is None:
        raise RuntimeError("LLM Gateway not initialized. Application not started.")
    return _llm_gateway


# ── Lifespan Manager ────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifecycle manager.
    Initializes all connections on startup, yields control to the app,
    and cleanly shuts everything down on exit.
    """
    global _llm_gateway
    settings = get_settings()

    logger.info("startup_begin", env=settings.app_env)

    # Initialize infrastructure
    init_db()
    init_redis()
    _llm_gateway = LLMGateway()
    init_mcp_client_manager()

    logger.info(
        "startup_complete",
        database="connected",
        redis="connected",
        llm_gateway="initialized",
        mcp_gateway="initialized",
    )

    yield

    # Shutdown
    logger.info("shutdown_begin")
    await close_mcp_client_manager()
    if _llm_gateway:
        await _llm_gateway.close()
        _llm_gateway = None
    await close_redis()
    await close_db()
    logger.info("shutdown_complete")


# ── Application Factory ─────────────────────────────────────────────

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Mr.Ops — AI Agent Orchestration Platform",
        description="Multi-tenant low-code AI agent builder with visual DAG workflows",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check
    @app.get("/health", tags=["Infrastructure"])
    async def health_check() -> dict[str, str]:
        return {"status": "healthy", "service": "mr-ops-backend"}

    # API routes will be registered here in subsequent phases:
    # from app.api.v1 import workflows, simulator, integrations
    # app.include_router(workflows.router, prefix="/api/v1")
    # app.include_router(simulator.router, prefix="/api/v1")
    # app.include_router(integrations.router, prefix="/api/v1")

    return app


# ── Uvicorn entrypoint ──────────────────────────────────────────────
app = create_app()
