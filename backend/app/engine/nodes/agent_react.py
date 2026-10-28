"""
ReAct Agent Node — custom tool-calling agent loop built from scratch.

This is the heart of the agentic execution engine. It implements a controlled
ReAct (Reasoning + Acting) loop WITHOUT any agentic framework:

    while iterations < max_iterations:
        1. Build messages with system prompt (temporal grounding + CAG/RAG context)
        2. Fetch MCP tool schemas for tenant's connected integrations
        3. Call LLM via gateway (supports Groq or OpenRouter)
        4. If response contains tool_calls:
           a. Parse all tool calls
           b. Execute them concurrently via asyncio.gather()
           c. Append tool results as role:"tool" messages
           d. Continue loop
        5. If response is final text (no tool_calls):
           a. Break loop
           b. Return synthesized response as NodeOutput

Key design constraints:
- NO langchain, NO langgraph, NO crewai — pure asyncio + openai SDK
- All tool execution goes through MCPClientManager (tenant-scoped OAuth)
- Temporal grounding injected into every system prompt
- CAG context blocks enable provider-level KV prefix caching
- RAG chunks appended as context sections
- FinOps telemetry accumulated across all loop iterations
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, ClassVar, Dict, List, Optional, Type

import structlog
from pydantic import BaseModel, Field

from app.engine.nodes.base import BaseNode
from app.engine.registry import register_node
from app.llm.gateway import LLMGateway
from app.main import get_llm_gateway
from app.mcp_gateway.client_manager import ToolResult, get_mcp_client_manager
from app.schemas.execution import ChatMessage, ExecutionContext, NodeOutput
from app.schemas.graph import LLMProvider, NodeType

logger = structlog.get_logger(__name__)


# ── Config Schema ────────────────────────────────────────────────────

class ReactAgentConfig(BaseModel):
    """Configuration for the ReAct Agent node."""
    system_prompt: str = Field(
        ...,
        min_length=1,
        description="Base system prompt defining the agent's persona and capabilities",
    )
    provider: LLMProvider = Field(
        default=LLMProvider.OPENROUTER,
        description="LLM provider for the agent's reasoning",
    )
    model: str = Field(
        default="anthropic/claude-sonnet-4-20250514",
        description="Primary model for the agent",
    )
    fallback_models: List[str] = Field(
        default_factory=list,
        description="OpenRouter fallback model chain",
    )
    mcp_servers: List[str] = Field(
        default_factory=list,
        description="MCP server names to bind tools from (e.g., ['google_calendar', 'gmail'])",
    )
    max_iterations: int = Field(
        default=8,
        ge=1,
        le=20,
        description="Maximum ReAct loop iterations (tool call rounds)",
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
    )
    max_tokens: int = Field(
        default=4096,
        ge=256,
        le=16384,
    )
    enable_cag: bool = Field(
        default=True,
        description="Whether to inject CAG context blocks for prefix caching",
    )
    output_variable: str = Field(
        default="agent_response",
        description="Variable name to store the final response in ctx.variables",
    )


# ── Node Implementation ─────────────────────────────────────────────

@register_node(NodeType.REACT_AGENT)
class ReactAgentNode(BaseNode):
    """
    Custom ReAct Agent implementing a controlled tool-calling loop.

    The agent dynamically binds MCP tool schemas, reasons about which
    tools to use, executes them concurrently, and synthesizes a final
    response — all without any agentic framework.
    """

    ConfigSchema: ClassVar[Type[BaseModel]] = ReactAgentConfig

    def _build_system_prompt(
        self,
        config: ReactAgentConfig,
        ctx: ExecutionContext,
    ) -> str:
        """
        Construct the full system prompt with:
        1. Base persona/instructions
        2. Temporal grounding (prevents date hallucinations)
        3. RAG context (if any retrieved chunks)
        """
        parts: List[str] = [config.system_prompt]

        # Temporal grounding
        parts.append(f"\n\n## Current Date & Time\n{ctx.get_temporal_grounding()}")

        # RAG context (not part of CAG cache — appended separately)
        if ctx.rag_retrieved_chunks:
            rag_text = "\n\n---\n\n".join(ctx.rag_retrieved_chunks)
            parts.append(
                f"\n\n## Retrieved Knowledge\n"
                f"Use the following retrieved information to answer the user's question. "
                f"If the information doesn't contain the answer, say so.\n\n{rag_text}"
            )

        # Variable context — expose extracted entities for tool use
        if ctx.variables:
            safe_vars = {
                k: v for k, v in ctx.variables.items()
                if isinstance(v, (str, int, float, bool, list))
            }
            if safe_vars:
                parts.append(
                    f"\n\n## Extracted Context Variables\n"
                    f"```json\n{json.dumps(safe_vars, indent=2, default=str)}\n```"
                )

        return "\n".join(parts)

    def _build_messages(
        self,
        system_prompt: str,
        ctx: ExecutionContext,
    ) -> List[ChatMessage]:
        """Build the full message list for the LLM call."""
        messages: List[ChatMessage] = [
            ChatMessage(role="system", content=system_prompt),
        ]

        # Append conversation history (excluding previous system messages)
        for msg in ctx.conversation_history:
            if msg.role != "system":
                messages.append(msg)

        return messages

    async def _execute_tool_calls(
        self,
        tenant_id: str,
        tool_calls: List[Dict[str, Any]],
    ) -> List[ChatMessage]:
        """
        Execute multiple tool calls concurrently via asyncio.gather.

        Each tool call is dispatched to the MCPClientManager, which
        handles OAuth credential injection and MCP server routing.

        Returns:
            List of ChatMessage(role="tool") with results for each call.
        """
        mcp = get_mcp_client_manager()

        async def _execute_single(tc: Dict[str, Any]) -> ChatMessage:
            tool_call_id = tc.get("id", "unknown")
            function_data = tc.get("function", {})
            tool_name = function_data.get("name", "unknown")
            raw_args = function_data.get("arguments", "{}")

            # Parse arguments
            try:
                arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError as exc:
                logger.error(
                    "tool_args_parse_failed",
                    tool_name=tool_name,
                    raw_args=raw_args,
                    error=str(exc),
                )
                return ChatMessage(
                    role="tool",
                    tool_call_id=tool_call_id,
                    name=tool_name,
                    content=json.dumps({
                        "error": f"Failed to parse tool arguments: {exc}",
                        "success": False,
                    }),
                )

            # Execute via MCP
            try:
                result: ToolResult = await mcp.execute_tool(
                    tenant_id=tenant_id,
                    tool_name=tool_name,
                    arguments=arguments,
                )
                content = json.dumps({
                    "success": result.success,
                    "output": result.output,
                    "error": result.error,
                }, default=str)
            except Exception as exc:
                logger.error(
                    "tool_execution_failed",
                    tool_name=tool_name,
                    error=str(exc),
                )
                content = json.dumps({
                    "success": False,
                    "error": f"Tool execution failed: {exc}",
                })

            logger.info(
                "tool_executed",
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                success=result.success if 'result' in dir() else False,
            )

            return ChatMessage(
                role="tool",
                tool_call_id=tool_call_id,
                name=tool_name,
                content=content,
            )

        # Execute all tool calls concurrently
        results = await asyncio.gather(
            *[_execute_single(tc) for tc in tool_calls],
            return_exceptions=False,
        )
        return list(results)

    async def execute(
        self,
        ctx: ExecutionContext,
        config: BaseModel,
    ) -> NodeOutput:
        """
        Run the ReAct loop:

        while iterations < max_iterations:
            -> Call LLM with tools
            -> If tool_calls: execute concurrently, append results, loop
            -> If final text: break and return

        Returns NodeOutput with the agent's final response.
        """
        cfg = config if isinstance(config, ReactAgentConfig) else ReactAgentConfig.model_validate(config.model_dump())
        gateway: LLMGateway = get_llm_gateway()

        # Build system prompt (without CAG — that's injected at gateway level)
        system_prompt = self._build_system_prompt(cfg, ctx)

        # Build initial message list
        messages = self._build_messages(system_prompt, ctx)

        # Fetch MCP tool schemas
        tool_schemas: List[Dict[str, Any]] = []
        if cfg.mcp_servers:
            try:
                mcp = get_mcp_client_manager()
                tool_schemas = await mcp.get_tool_schemas(
                    tenant_id=ctx.tenant_id,
                    server_names=cfg.mcp_servers,
                )
            except Exception as exc:
                logger.warning(
                    "mcp_tool_schema_fetch_failed",
                    error=str(exc),
                    servers=cfg.mcp_servers,
                )

        # Determine CAG blocks for cache injection
        cag_blocks = ctx.cag_context_blocks if cfg.enable_cag else None

        # ── ReAct Loop ───────────────────────────────────────────────
        iteration = 0
        total_telemetry: Dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "cost_usd": 0.0,
            "iterations": 0,
            "tool_calls_total": 0,
        }
        final_response: str = ""
        accumulated_tool_messages: List[ChatMessage] = []

        while iteration < cfg.max_iterations:
            iteration += 1
            total_telemetry["iterations"] = iteration

            logger.info(
                "react_iteration_start",
                execution_id=ctx.execution_id,
                iteration=iteration,
                max_iterations=cfg.max_iterations,
                tool_count=len(tool_schemas),
                message_count=len(messages),
            )

            # Call LLM
            response, tel = await gateway.chat_completion(
                provider=cfg.provider,
                model=cfg.model,
                messages=messages,
                tools=tool_schemas if tool_schemas else None,
                tool_choice="auto" if tool_schemas else None,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                fallback_models=cfg.fallback_models if cfg.provider == LLMProvider.OPENROUTER else None,
                cag_context_blocks=cag_blocks if cfg.provider == LLMProvider.OPENROUTER else None,
            )

            # Accumulate telemetry
            total_telemetry["prompt_tokens"] += tel.prompt_tokens
            total_telemetry["completion_tokens"] += tel.completion_tokens
            total_telemetry["cached_tokens"] += tel.cached_tokens
            total_telemetry["cost_usd"] += tel.cost_usd

            choice = response.choices[0]
            assistant_message = choice.message

            # Check for tool calls
            if assistant_message.tool_calls:
                # Serialize tool_calls for the message history
                tc_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in assistant_message.tool_calls
                ]

                # Append assistant message with tool_calls
                assistant_chat_msg = ChatMessage(
                    role="assistant",
                    content=assistant_message.content,
                    tool_calls=tc_dicts,
                )
                messages.append(assistant_chat_msg)

                total_telemetry["tool_calls_total"] += len(tc_dicts)

                logger.info(
                    "react_tool_calls",
                    iteration=iteration,
                    tools=[tc["function"]["name"] for tc in tc_dicts],
                )

                # Execute tool calls concurrently
                tool_results = await self._execute_tool_calls(
                    tenant_id=ctx.tenant_id,
                    tool_calls=tc_dicts,
                )

                # Append tool results to messages
                messages.extend(tool_results)
                accumulated_tool_messages.append(assistant_chat_msg)
                accumulated_tool_messages.extend(tool_results)

            else:
                # No tool calls — this is the final response
                final_response = assistant_message.content or ""
                logger.info(
                    "react_final_response",
                    iteration=iteration,
                    response_length=len(final_response),
                )
                break

        # If we exhausted iterations without a final response, force one
        if not final_response and iteration >= cfg.max_iterations:
            logger.warning(
                "react_max_iterations_reached",
                execution_id=ctx.execution_id,
                iterations=iteration,
            )
            # The last assistant message content (if any) becomes the response
            for msg in reversed(messages):
                if msg.role == "assistant" and msg.content:
                    final_response = msg.content
                    break
            if not final_response:
                final_response = (
                    "I apologize, but I wasn't able to complete the task within "
                    "the allowed number of steps. Please try again or simplify your request."
                )

        # Build messages to append to conversation history
        messages_to_append: List[ChatMessage] = [
            *accumulated_tool_messages,
            ChatMessage(role="assistant", content=final_response),
        ]

        return NodeOutput(
            updated_variables={
                cfg.output_variable: final_response,
            },
            next_handle="default",
            messages_to_append=messages_to_append,
            telemetry=total_telemetry,
        )

    async def on_error(
        self,
        ctx: ExecutionContext,
        config: BaseModel,
        error: Exception,
    ) -> NodeOutput | None:
        """
        Error recovery: if the LLM call fails, return a graceful error
        message rather than crashing the entire workflow.
        """
        logger.error(
            "react_agent_error",
            execution_id=ctx.execution_id,
            error=str(error),
        )
        return NodeOutput(
            updated_variables={
                "agent_response": (
                    "I'm sorry, I encountered an error while processing your request. "
                    "Please try again in a moment."
                ),
                "agent_error": str(error),
            },
            next_handle="error",
            messages_to_append=[
                ChatMessage(
                    role="assistant",
                    content="I'm sorry, I encountered an error while processing your request. "
                            "Please try again in a moment.",
                ),
            ],
            telemetry={"error": str(error)},
        )
