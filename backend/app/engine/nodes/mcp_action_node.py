"""
MCP Tool Action Node — deterministic single-tool execution.

Unlike the ReactAgentNode (which dynamically selects tools via LLM reasoning),
this node executes a single, pre-configured MCP tool with arguments resolved
from the ExecutionContext variables. It's used for deterministic actions where
the tool and its inputs are known at design time.

Use cases:
- "Always send a confirmation email after booking"
- "Always check calendar availability before routing to booking agent"
- "Create a Google Meet link and store the URL"

The node config specifies:
- `tool_name`: The exact MCP tool to call
- `argument_mapping`: Maps tool parameters to ctx.variables paths
- `static_arguments`: Fixed arguments that don't change per execution
"""
from __future__ import annotations

import json
import time
from typing import Any, ClassVar, Dict, Optional, Type

import structlog
from pydantic import BaseModel, Field

from app.engine.nodes.base import BaseNode
from app.engine.registry import register_node
from app.mcp_gateway.client_manager import ToolResult, get_mcp_client_manager
from app.schemas.execution import ChatMessage, ExecutionContext, NodeOutput
from app.schemas.graph import NodeType

logger = structlog.get_logger(__name__)


# ── Config Schema ────────────────────────────────────────────────────

class MCPActionConfig(BaseModel):
    """Configuration for deterministic MCP tool execution."""
    tool_name: str = Field(
        ...,
        min_length=1,
        description="Fully qualified MCP tool name (e.g., 'calendar_schedule_with_meet')",
    )
    argument_mapping: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Maps tool parameter names to ctx.variables paths. "
            "E.g. {'attendee_emails': 'extracted_emails', 'start_iso': 'appointment_start'}"
        ),
    )
    static_arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Fixed arguments passed directly to the tool",
    )
    output_variable: str = Field(
        default="tool_result",
        description="Variable name to store the tool result in ctx.variables",
    )
    fail_on_error: bool = Field(
        default=True,
        description="If True, routes to 'error' handle on tool failure. If False, continues with 'default'.",
    )
    add_to_conversation: bool = Field(
        default=True,
        description="Whether to append the tool result as a system message to conversation history",
    )


# ── Node Implementation ─────────────────────────────────────────────

@register_node(NodeType.MCP_TOOL_ACTION)
class MCPActionNode(BaseNode):
    """
    Deterministic single-tool execution node.

    Resolves arguments from the execution context and static config,
    executes exactly one MCP tool, and stores the result.
    """

    ConfigSchema: ClassVar[Type[BaseModel]] = MCPActionConfig

    def _resolve_arguments(
        self,
        config: MCPActionConfig,
        ctx: ExecutionContext,
    ) -> Dict[str, Any]:
        """
        Build the final tool arguments by:
        1. Starting with static_arguments
        2. Overlaying argument_mapping (resolved from ctx.variables)
        """
        arguments: Dict[str, Any] = dict(config.static_arguments)

        for tool_param, variable_path in config.argument_mapping.items():
            value = self._resolve_path(variable_path, ctx)
            if value is not None:
                arguments[tool_param] = value
            else:
                logger.warning(
                    "argument_mapping_miss",
                    tool_param=tool_param,
                    variable_path=variable_path,
                    available_vars=list(ctx.variables.keys()),
                )

        return arguments

    @staticmethod
    def _resolve_path(path: str, ctx: ExecutionContext) -> Any:
        """
        Resolve a dot-notation path against ctx.variables or ctx.node_outputs.
        Supports: "email", "variables.email", "node_outputs.node_1.meet_link"
        """
        parts = path.split(".")

        # Direct variable lookup (no prefix)
        if len(parts) == 1:
            return ctx.variables.get(parts[0])

        root = parts[0]
        remainder = parts[1:]

        if root == "variables":
            obj: Any = ctx.variables
        elif root == "node_outputs":
            obj = ctx.node_outputs
        else:
            # Treat the full path as a direct variable key
            return ctx.variables.get(path)

        for part in remainder:
            if isinstance(obj, dict):
                obj = obj.get(part)
            else:
                return None

        return obj

    async def execute(
        self,
        ctx: ExecutionContext,
        config: BaseModel,
    ) -> NodeOutput:
        """
        Execute a single MCP tool deterministically.

        1. Resolve arguments from ctx.variables + static config
        2. Execute the tool via MCPClientManager
        3. Store result in ctx.variables
        4. Route to 'default' on success, 'error' on failure
        """
        cfg = config if isinstance(config, MCPActionConfig) else MCPActionConfig.model_validate(config.model_dump())
        mcp = get_mcp_client_manager()

        # Resolve arguments
        arguments = self._resolve_arguments(cfg, ctx)

        logger.info(
            "mcp_action_executing",
            execution_id=ctx.execution_id,
            tool_name=cfg.tool_name,
            argument_keys=list(arguments.keys()),
        )

        # Execute
        start_time = time.monotonic()
        try:
            result: ToolResult = await mcp.execute_tool(
                tenant_id=ctx.tenant_id,
                tool_name=cfg.tool_name,
                arguments=arguments,
            )
        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "mcp_action_exception",
                tool_name=cfg.tool_name,
                error=str(exc),
            )
            result = ToolResult(
                tool_name=cfg.tool_name,
                success=False,
                error=str(exc),
            )

        elapsed_ms = (time.monotonic() - start_time) * 1000

        # Build output
        updated_vars: Dict[str, Any] = {
            cfg.output_variable: result.output if result.success else None,
            f"{cfg.output_variable}_success": result.success,
        }
        if result.error:
            updated_vars[f"{cfg.output_variable}_error"] = result.error

        # Determine routing handle
        if result.success:
            next_handle = "default"
        elif cfg.fail_on_error:
            next_handle = "error"
        else:
            next_handle = "default"

        # Build conversation message
        messages_to_append = []
        if cfg.add_to_conversation:
            result_summary = (
                f"Tool '{cfg.tool_name}' executed "
                f"{'successfully' if result.success else 'with an error'}."
            )
            if result.success and result.output:
                result_summary += f"\nResult: {json.dumps(result.output, default=str)}"
            elif result.error:
                result_summary += f"\nError: {result.error}"

            messages_to_append.append(
                ChatMessage(role="system", content=result_summary)
            )

        logger.info(
            "mcp_action_completed",
            execution_id=ctx.execution_id,
            tool_name=cfg.tool_name,
            success=result.success,
            next_handle=next_handle,
            latency_ms=f"{elapsed_ms:.0f}ms",
        )

        return NodeOutput(
            updated_variables=updated_vars,
            next_handle=next_handle,
            messages_to_append=messages_to_append,
            telemetry={
                "tool_name": cfg.tool_name,
                "success": result.success,
                "latency_ms": elapsed_ms,
                "error": result.error,
            },
        )
