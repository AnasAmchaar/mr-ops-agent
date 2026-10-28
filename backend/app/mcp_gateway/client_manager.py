"""
MCP Client Manager — dynamic tool discovery and tenant-scoped invocation.

This is the single entry point used by engine nodes (ReactAgentNode,
MCPActionNode) to discover tool schemas and execute MCP tools.

Execution pipeline per tool call:
1. Resolve the owning internal MCP server from the globally-unique tool name
   (guarded by a name-collision check at init).
2. Validate arguments against the tool's Pydantic input model — LLM output
   is never trusted raw at this boundary (negative constraint #4).
3. Fetch the tenant's valid Google access token via OAuthVault (Redis cache
   → refresh flow), scoped strictly by tenant_id.
4. Invoke the handler, translating typed errors:
   - GoogleAPIError(401/403) → auth problem surfaced as failure with a
     re-authorization hint (status ERROR/REVOKED was already recorded).
   - GrantInvalid → "re-connect integration" message for the agent/sandbox.
   - ValidationError → schema mismatch surfaced for LLM self-correction.

Redis/DB/HTTP are all async — nothing blocks the event loop.

This V1 ships in-process transport: engine nodes and internal servers share
the process, and tenancy is bound server-side (never read from tool args).
Out-of-process FastMCP transports over the same handlers are exposed via
`servers.create_all_mcp_servers()` for SSE/streamable-HTTP deployment.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import structlog
import uuid as uuid_mod
from pydantic import BaseModel, ValidationError
from sqlalchemy import select

from app.core.database import get_db_session
from app.mcp_gateway.oauth_vault import (
    AccessToken,
    GoogleAuthAPIError,
    GrantInvalid,
    IntegrationNotConnected,
    get_oauth_vault,
)
from app.mcp_gateway.servers import ALL_TOOLS, SERVER_TOOLS, get_tool, tool_name_to_server
from app.mcp_gateway.tools.base import GoogleAPIError, google_auth_failed
from app.models.integration import IntegrationStatus, IntegrationType, TenantIntegration

logger = structlog.get_logger(__name__)


# ── Wire Contracts (imported by Phase 2 engine nodes) ────────────────

class ToolSchema(BaseModel):
    """JSON Schema representation of an MCP tool for LLM function calling."""
    name: str
    description: str
    parameters: dict[str, Any]


class ToolResult(BaseModel):
    """Result from executing an MCP tool."""
    tool_name: str
    success: bool
    output: Any = None
    error: Optional[str] = None


class MCPClientManager:
    """
    Multi-tenant MCP tool discovery and execution manager.

    Responsibilities:
    - Discover available tools scoped to a tenant's connected integrations.
    - Convert MCP tool definitions to OpenAI function-calling format.
    - Execute tools with tenant-bound OAuth2 credentials.
    """

    def __init__(self) -> None:
        self._vault = get_oauth_vault()
        self._validate_registry()

    @staticmethod
    def _validate_registry() -> None:
        """Tool names must be globally unique so dispatch is unambiguous."""
        seen: dict[str, str] = {}
        for tool in ALL_TOOLS:
            owner = seen.get(tool.name)
            if owner is not None:
                raise RuntimeError(
                    f"Duplicate tool name '{tool.name}' registered by "
                    f"servers '{owner}' and '{tool.server}'"
                )
            seen[tool.name] = tool.server
        logger.info(
            "mcp_registry_validated",
            servers=sorted(SERVER_TOOLS.keys()),
            tool_count=len(ALL_TOOLS),
        )

    # ── Tool Discovery ───────────────────────────────────────────────

    async def get_tool_schemas(
        self,
        tenant_id: str,
        server_names: List[str],
    ) -> List[Dict[str, Any]]:
        """
        OpenAI function-calling schemas for the requested MCP servers,
        filtered to the integrations this tenant actually connected.

        Args:
            tenant_id: The tenant whose integrations to inspect.
            server_names: MCP server names (e.g. ["google_calendar", "gmail"]).

        Returns:
            [{"type": "function", "function": {name, description, parameters}}]
        """
        connected = await self._connected_servers(tenant_id, server_names)

        schemas: List[Dict[str, Any]] = []
        for server_name in connected:
            for tool in SERVER_TOOLS.get(server_name, ()):
                schemas.append(tool.to_openai_tool())

        if not connected and server_names:
            logger.info(
                "mcp_no_connected_servers",
                tenant_id=tenant_id,
                requested=server_names,
            )
        else:
            logger.info(
                "mcp_tool_schemas_served",
                tenant_id=tenant_id,
                servers=connected,
                tool_count=len(schemas),
            )
        return schemas

    async def _connected_servers(
        self, tenant_id: str, server_names: List[str]
    ) -> List[str]:
        """Query DB for CONNECTED integrations among the requested servers."""
        connected: List[str] = []
        async with get_db_session() as session:
            for server_name in server_names:
                integration_type = _server_to_integration_type(server_name)
                if integration_type is None:
                    continue
                stmt = (
                    select(TenantIntegration)
                    .where(TenantIntegration.tenant_id == _parse_uuid(tenant_id))
                    .where(TenantIntegration.integration_type == integration_type)
                    .where(TenantIntegration.status == IntegrationStatus.CONNECTED)
                    .limit(1)
                )
                result = await session.execute(stmt)
                if result.scalar_one_or_none() is not None:
                    connected.append(server_name)
                else:
                    logger.info(
                        "mcp_server_not_connected",
                        tenant_id=tenant_id,
                        server=server_name,
                    )
        return connected

    # ── Tool Execution ───────────────────────────────────────────────

    async def execute_tool(
        self,
        tenant_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> ToolResult:
        """
        Execute a single MCP tool with tenant-scoped credentials.

        Args:
            tenant_id: Tenant scope for OAuth credential resolution.
            tool_name: Globally unique tool name (e.g. "calendar_schedule_with_meet").
            arguments: Tool arguments as parsed from the LLM tool_call.

        Returns:
            ToolResult with success status and output/error.
        """
        definition = get_tool(tool_name)
        if definition is None:
            logger.warning("mcp_tool_unknown", tenant_id=tenant_id, tool_name=tool_name)
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Unknown tool '{tool_name}'. Available tools: "
                      + ", ".join(sorted(t.name for t in ALL_TOOLS)),
            )

        owned_server = tool_name_to_server(tool_name)
        if owned_server is None:  # pragma: no cover — registry invariant
            return ToolResult(tool_name=tool_name, success=False, error="Tool registry inconsistency.")

        # 1. Strict argument validation against the typed schema
        try:
            validated_input = definition.input_model.model_validate(arguments)
        except ValidationError as exc:
            logger.info(
                "mcp_tool_args_invalid",
                tenant_id=tenant_id,
                tool_name=tool_name,
                errors=exc.errors(include_url=False),
            )
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=(
                    f"Invalid arguments for '{tool_name}'. "
                    f"Schema errors: {exc.errors(include_url=False)}"
                ),
            )

        # 2. Tenant-bound OAuth token (never sourced from tool arguments)
        try:
            access_token: AccessToken | None = await self._vault.get_access_token(
                tenant_id=tenant_id,
                integration_type=_server_to_integration_type(owned_server) or "",
            )
        except IntegrationNotConnected as exc:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output=None,
                error=(
                    f"{owned_server} integration is not connected for this tenant. "
                    "Guide the user to connect it in integrations settings."
                ),
            )
        except GrantInvalid:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=(
                    f"The tenant's {owned_server} authorization was revoked. "
                    "The business owner must re-connect the integration."
                ),
            )
        except GoogleAuthAPIError as exc:
            logger.error(
                "oauth_vault_transient_failure",
                tenant_id=tenant_id,
                server=owned_server,
                error=str(exc),
            )
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Authentication backend temporarily unavailable. Retry shortly.",
            )

        # 3. Dispatch to the typed handler
        try:
            output = await definition.handler(validated_input, access_token.value)
        except GoogleAPIError as exc:
            if google_auth_failed(exc):
                logger.warning(
                    "google_api_auth_failure",
                    tenant_id=tenant_id,
                    tool_name=tool_name,
                    status=exc.status_code,
                )
                await self._vault.invalidate(tenant_id, owned_server)
                return ToolResult(
                    tool_name=tool_name,
                    success=False,
                    error=(
                        "Google rejected the request (auth failure). The "
                        "integration may need re-authorization."
                    ),
                )
            logger.warning(
                "google_api_error",
                tenant_id=tenant_id,
                tool_name=tool_name,
                status=exc.status_code,
                message=exc.message,
            )
            return ToolResult(tool_name=tool_name, success=False, error=exc.message)
        except Exception as exc:
            logger.exception(
                "mcp_tool_unhandled_exception",
                tenant_id=tenant_id,
                tool_name=tool_name,
            )
            return ToolResult(tool_name=tool_name, success=False, error=str(exc))

        logger.info(
            "mcp_tool_executed",
            tenant_id=tenant_id,
            server=owned_server,
            tool_name=tool_name,
        )
        return ToolResult(tool_name=tool_name, success=True, output=output)

    # ── Lifecycle ────────────────────────────────────────────────────

    async def close(self) -> None:
        """Release the shared Google API HTTP client."""
        from app.mcp_gateway.tools.base import close_http_client

        await close_http_client()


# ── Module Helpers ───────────────────────────────────────────────────

def _parse_uuid(value: str):
    """Normalize a tenant_id string into a UUID for SQL matching."""
    return uuid_mod.UUID(value)


_SERVER_TO_INTEGRATION_TYPE: dict[str, IntegrationType] = {
    "google_calendar": IntegrationType.GOOGLE_CALENDAR,
    "google_meet": IntegrationType.GOOGLE_MEET,
    "gmail": IntegrationType.GMAIL,
}


def _server_to_integration_type(server_name: str) -> IntegrationType | None:
    """Map an MCP server name to its DB integration type enumeration."""
    return _SERVER_TO_INTEGRATION_TYPE.get(server_name)


# ── Singleton ────────────────────────────────────────────────────────

_manager: MCPClientManager | None = None


def init_mcp_client_manager() -> MCPClientManager:
    """Initialize the global MCPClientManager singleton. Call at startup."""
    global _manager
    if _manager is None:
        _manager = MCPClientManager()
    return _manager


def get_mcp_client_manager() -> MCPClientManager:
    """Accessor for the MCPClientManager singleton."""
    if _manager is None:
        raise RuntimeError(
            "MCPClientManager not initialized. Call init_mcp_client_manager() first."
        )
    return _manager


async def close_mcp_client_manager() -> None:
    """Gracefully shut down the manager and its shared HTTP client."""
    global _manager
    if _manager is not None:
        await _manager.close()
        _manager = None
