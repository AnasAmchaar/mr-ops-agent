# Mr.Ops — AI Agent Orchestration Platform (Backend)

> Multi-tenant low-code AI agent builder with visual DAG workflows.
> Frameworkless orchestration engine: ReAct agents, MCP tool integration, Groq/OpenRouter LLM gateway, WhatsApp channels, and human-in-the-loop approval gates.

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/fastapi-0.115-green)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![CI](https://github.com/<user>/mr-ops-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/<user>/mr-ops-agent/actions)

---

## What it is

**Mr.Ops** is the backend of a multi-tenant AI agent orchestration platform. Business owners compose agent workflows on a React Flow canvas; the backend compiles the graph into a DAG and executes it with a frameworkless state machine. Agents can reason, call tools (Google Calendar, Gmail, Meet), suspend for human approval, and reply over WhatsApp — all scoped per-tenant with encrypted OAuth credentials.

## Features

| Area | Capability |
|------|------------|
| **Workflow engine** | `GraphRunner` DAG state machine with edge routing, cycle detection, `max_steps` circuit breaker |
| **Nodes** | WhatsApp trigger/reply, intent router (Groq structured JSON), ReAct agent, MCP tool action, HITL approval |
| **Agent loop** | Frameworkless ReAct (Reasoning + Acting) with concurrent tool execution via `asyncio.gather` |
| **LLM gateway** | Unified Groq + OpenRouter client with retries, CAG cache-control injection, Whisper transcription |
| **MCP gateway** | Tenant-scoped tool discovery & invocation; Google Calendar/Gmail/Meet servers; OAuth vault with AES-256-GCM sealed refresh tokens |
| **State & recovery** | `ExecutionContext` persisted to Redis; HITL suspension & resume across restarts; crash recovery |
| **FinOps** | Token metering, latency tracking, per-model cost estimation, CAG cache savings |
| **Security** | JWT session tokens, AES-256-GCM at-rest encryption, tenant-scoped DB queries |

## Tech stack

- **Python** 3.12+
- **FastAPI** — web framework & OpenAPI
- **SQLAlchemy 2.0 (async)** + **asyncpg** — PostgreSQL
- **Redis** — pub/sub traces, session state, OAuth cache
- **Pydantic v2** — settings, schemas, validation
- **structlog** — structured logging
- **tenacity** — retry with exponential backoff
- **alembic** — database migrations
- **arq** — background job queue
- **mcp** — Model Context Protocol
- **cryptography** — AES-256-GCM
- **PyJWT** — session tokens

## Project structure

```
mr-ops-agent/
├── backend/
│   ├── pyproject.toml          # dependencies, ruff, mypy config
│   └── app/
│       ├── main.py             # FastAPI factory, lifespan, health check
│       ├── core/
│       │   ├── config.py       # Pydantic Settings (single source of truth)
│       │   ├── database.py     # Async engine, session factory, Redis pool
│       │   └── security.py     # AES-256-GCM encryption, JWT issue/verify
│       ├── engine/
│       │   ├── runner.py       # GraphRunner DAG execution state machine
│       │   ├── registry.py     # Decorator-based NodeRegistry
│       │   ├── context.py      # ExecutionContextManager (Redis persist/resume)
│       │   └── nodes/          # BaseNode + 7 node implementations
│       ├── llm/
│       │   ├── gateway.py      # Unified Groq + OpenRouter + Whisper
│       │   ├── telemetry.py    # FinOps cost calculator
│       │   └── caching.py      # CAG cache-control helpers
│       ├── mcp_gateway/
│       │   ├── client_manager.py  # Tenant-scoped MCP tool execution
│       │   ├── oauth_vault.py     # Google OAuth2 credential management
│       │   ├── servers/           # Google Calendar / Gmail / Meet
│       │   └── tools/             # Typed ToolDefinition contract
│       ├── models/             # SQLAlchemy: TenantIntegration (encrypted refresh tokens)
│       └── schemas/            # WorkflowBlueprint, ExecutionContext, NodeOutput, TraceEvent
├── README.md
├── .gitignore
├── LICENSE
└── .github/workflows/ci.yml
```

## Prerequisites

- Python 3.12+
- PostgreSQL 14+
- Redis 7+
- Groq API key (optional — for intent routing)
- OpenRouter API key (optional — for ReAct agent)
- Google OAuth2 client ID/secret (for Calendar/Gmail/Meet integrations)

## Install

```powershell
# Clone
git clone https://github.com/<your-username>/mr-ops-agent.git
cd mr-ops-agent

# Create & activate virtualenv
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install package + dev dependencies
pip install -e ".[dev]"

# Copy environment template and fill in real values
copy ..\..\.env.example .env
# edit .env with your DATABASE_URL, REDIS_URL, API keys, secrets
```

## Run

```powershell
cd backend
.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Health: `GET http://localhost:8000/health`
- Interactive docs: `http://localhost:8000/docs` (disabled in production)

## Node types

| Node type | Purpose |
|-----------|---------|
| `whatsapp_trigger` | WhatsApp webhook entry point |
| `intent_router` | Groq-powered intent classification → edge routing |
| `react_agent` | Frameworkless ReAct agent with MCP tools |
| `mcp_tool_action` | Deterministic single-tool execution |
| `hitl_approval` | Human-in-the-loop suspension & resume |
| `whatsapp_reply` | Send response back via WhatsApp Cloud API |

## API

```
GET /health   → {"status": "healthy", "service": "mr-ops-backend"}
```

Versioned API routers (`/api/v1/workflows`, `/api/v1/simulator`, `/api/v1/integrations`) are scaffolded and will be registered in `main.py` in the next phase.

## Lint / typecheck / test

```powershell
cd backend
.venv\Scripts\Activate.ps1

ruff check .
ruff format --check .
mypy .
pytest
```

CI runs these on every push/PR via GitHub Actions.

## Environment variables

See `.env.example`. Required at minimum: `DATABASE_URL`, `REDIS_URL`, `JWT_SECRET`, `ENCRYPTION_KEY`. LLM and WhatsApp keys are optional depending on which channels you enable.

**Never commit `.env`.** It is gitignored.

## Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/your-feature`)
3. Commit (`git commit -m "feat: ..."`)
4. Push and open a pull request

Please run `ruff check`, `ruff format`, `mypy`, and `pytest` before opening a PR.

## License

MIT License. See `LICENSE`.

## Security

If you discover a security vulnerability, please open a private advisory or contact the maintainer. Do not disclose sensitive details in public issues.
