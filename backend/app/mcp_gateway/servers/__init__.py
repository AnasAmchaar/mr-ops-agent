"""
Internal Multi-Tenant MCP Servers — Google Workspace tool providers.

Each module declares `TOOLS: tuple[ToolDefinition, ...]` and optionally a
`create_mcp_server()` factory for official-SDK FastMCP out-of-process
deployment. The in-process MCPClientManager dispatches to the same handlers.
"""
from __future__ import annotations

from app.mcp_gateway.servers import gmail, google_calendar, google_meet
from app.mcp_gateway.tools.base import ToolDefinition

# Canonical server registry — server name -> its typed tool definitions
SERVER_TOOLS: dict[str, tuple[ToolDefinition, ...]] = {
    google_calendar.SERVER_NAME: google_calendar.TOOLS,
    google_meet.SERVER_NAME: google_meet.TOOLS,
    gmail.SERVER_NAME: gmail.TOOLS,
}

# All tools across every server (unique names enforced by ClientManager init)
ALL_TOOLS: tuple[ToolDefinition, ...] = tuple(
    tool for tools in SERVER_TOOLS.values() for tool in tools
)


def tool_name_to_server(tool_name: str) -> str | None:
    """Resolve which server owns a globally unique tool name."""
    for server_name, tools in SERVER_TOOLS.items():
        for tool in tools:
            if tool.name == tool_name:
                return server_name
    return None


def get_tool(tool_name: str) -> ToolDefinition | None:
    for tool in ALL_TOOLS:
        if tool.name == tool_name:
            return tool
    return None


def create_all_mcp_servers() -> dict[str, object]:
    """Instantiate the FastMCP wrappers for all internal servers."""
    return {
        "google_calendar": google_calendar.create_mcp_server(),
        "google_meet": google_meet.create_mcp_server(),
        "gmail": gmail.create_mcp_server(),
    }
