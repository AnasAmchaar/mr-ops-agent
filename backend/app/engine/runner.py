"""
GraphRunner — the core DAG execution state machine.

This is the frameworkless orchestration engine that:
1. Loads a WorkflowBlueprint and ExecutionContext.
2. Resolves the current node from the NodeRegistry.
3. Executes nodes in a controlled while-loop with circuit breaker.
4. Routes to the next node via edge conditions.
5. Emits real-time TraceEvents to Redis Pub/Sub for SSE frontend tracing.
6. Handles cycle detection, suspension (HITL), and error recovery.

The runner supports both acyclic DAGs and controlled cyclic graphs (e.g.,
ReAct loops that cycle back to the router after tool execution). The
`max_steps` circuit breaker prevents infinite loops.
"""
from __future__ import annotations

import time
import traceback
from typing import Any, Dict, List, Optional

import redis.asyncio as aioredis
import structlog

from app.core.database import get_redis
from app.engine.registry import NodeRegistry
from app.schemas.execution import (
    ExecutionContext,
    ExecutionStatus,
    NodeOutput,
    TraceEvent,
)
from app.schemas.graph import GraphEdge, WorkflowBlueprint

logger = structlog.get_logger(__name__)


class GraphRunner:
    """
    The core DAG execution engine.

    Lifecycle:
        runner = GraphRunner(blueprint, ctx)
        final_ctx = await runner.run()

    The runner modifies `ctx` in-place as nodes execute, and returns
    the final context when the workflow completes, fails, or suspends.
    """

    def __init__(
        self,
        blueprint: WorkflowBlueprint,
        ctx: ExecutionContext,
        *,
        redis_client: Optional[aioredis.Redis] = None,  # type: ignore[type-arg]
    ) -> None:
        self._blueprint = blueprint
        self._ctx = ctx
        self._redis: Optional[aioredis.Redis] = redis_client  # type: ignore[type-arg]

        # Pre-build lookup tables for O(1) access during execution
        self._node_map: Dict[str, Any] = {n.id: n for n in blueprint.nodes}
        self._adjacency: Dict[str, List[GraphEdge]] = {}
        for edge in blueprint.edges:
            self._adjacency.setdefault(edge.source, []).append(edge)

    # ── Public API ───────────────────────────────────────────────────

    async def run(self) -> ExecutionContext:
        """
        Execute the workflow graph from the current node until completion,
        failure, or suspension (HITL).

        Returns:
            The final ExecutionContext with accumulated state.
        """
        try:
            self._redis = self._redis or get_redis()
        except RuntimeError:
            self._redis = None
            logger.warning("redis_unavailable", msg="Trace events will not be emitted")

        logger.info(
            "graph_run_started",
            execution_id=self._ctx.execution_id,
            workflow_id=self._ctx.workflow_id,
            tenant_id=self._ctx.tenant_id,
            entry_node=self._ctx.current_node_id,
            max_steps=self._blueprint.max_steps,
        )

        while (
            self._ctx.status == ExecutionStatus.RUNNING
            and self._ctx.step_count < self._blueprint.max_steps
        ):
            node_id = self._ctx.current_node_id
            graph_node = self._node_map.get(node_id)

            if graph_node is None:
                await self._fail(
                    node_id,
                    RuntimeError(f"Node '{node_id}' not found in blueprint"),
                )
                break

            # Resolve the node implementation from registry
            try:
                node_cls = NodeRegistry.get(graph_node.type)
            except KeyError as exc:
                await self._fail(node_id, exc)
                break

            node_instance = node_cls()

            # Validate and parse the node config
            try:
                config = node_instance.parse_config(graph_node.config)
                node_instance.validate_config(config)
            except Exception as exc:
                await self._fail(node_id, exc)
                break

            # Emit node_started trace
            await self._emit_trace(
                node_id=node_id,
                node_type=graph_node.type.value,
                event="node_started",
            )

            start_time = time.monotonic()

            # Execute the node
            try:
                output: NodeOutput = await node_instance.execute(self._ctx, config)
            except Exception as exc:
                # Attempt error recovery via the node's on_error hook
                try:
                    recovery = await node_instance.on_error(self._ctx, config, exc)
                except Exception:
                    recovery = None

                if recovery is not None:
                    output = recovery
                    logger.warning(
                        "node_error_recovered",
                        node_id=node_id,
                        error=str(exc),
                    )
                else:
                    elapsed_ms = (time.monotonic() - start_time) * 1000
                    await self._emit_trace(
                        node_id=node_id,
                        node_type=graph_node.type.value,
                        event="node_failed",
                        duration_ms=elapsed_ms,
                        data={"error": str(exc), "traceback": traceback.format_exc()},
                    )
                    await self._fail(node_id, exc)
                    break

            elapsed_ms = (time.monotonic() - start_time) * 1000

            # Apply the node's output to the execution context
            self._ctx.merge_node_output(node_id, output)
            self._ctx.step_count += 1

            # Emit node_completed trace
            await self._emit_trace(
                node_id=node_id,
                node_type=graph_node.type.value,
                event="node_completed",
                duration_ms=elapsed_ms,
                data={
                    "next_handle": output.next_handle,
                    "variables_updated": list(output.updated_variables.keys()),
                    "telemetry": output.telemetry,
                },
            )

            logger.info(
                "node_executed",
                execution_id=self._ctx.execution_id,
                node_id=node_id,
                node_type=graph_node.type.value,
                step=self._ctx.step_count,
                next_handle=output.next_handle,
                duration_ms=f"{elapsed_ms:.1f}ms",
            )

            # Check for suspension (HITL)
            if output.suspend_execution:
                self._ctx.status = ExecutionStatus.PAUSED_FOR_HUMAN
                logger.info(
                    "execution_suspended",
                    execution_id=self._ctx.execution_id,
                    node_id=node_id,
                    reason="human_approval_required",
                )
                break

            # Route to the next node
            next_node_id = self._resolve_next_node(node_id, output.next_handle)

            if next_node_id is None:
                # No outgoing edges — workflow is complete
                self._ctx.status = ExecutionStatus.COMPLETED
                await self._emit_trace(
                    node_id=node_id,
                    node_type=graph_node.type.value,
                    event="execution_completed",
                    data={
                        "total_steps": self._ctx.step_count,
                        "total_prompt_tokens": self._ctx.total_prompt_tokens,
                        "total_completion_tokens": self._ctx.total_completion_tokens,
                        "estimated_cost_usd": self._ctx.estimated_cost_usd,
                    },
                )
                logger.info(
                    "graph_run_completed",
                    execution_id=self._ctx.execution_id,
                    total_steps=self._ctx.step_count,
                    cost_usd=f"${self._ctx.estimated_cost_usd:.6f}",
                )
                break

            self._ctx.current_node_id = next_node_id

        # Circuit breaker check
        if (
            self._ctx.status == ExecutionStatus.RUNNING
            and self._ctx.step_count >= self._blueprint.max_steps
        ):
            self._ctx.status = ExecutionStatus.FAILED
            error_msg = (
                f"Circuit breaker triggered: reached max_steps "
                f"({self._blueprint.max_steps}) without completion"
            )
            await self._emit_trace(
                node_id=self._ctx.current_node_id,
                node_type="circuit_breaker",
                event="execution_failed",
                data={"error": error_msg},
            )
            logger.error(
                "circuit_breaker_triggered",
                execution_id=self._ctx.execution_id,
                max_steps=self._blueprint.max_steps,
                step_count=self._ctx.step_count,
            )

        return self._ctx

    # ── Edge Routing ─────────────────────────────────────────────────

    def _resolve_next_node(
        self,
        current_node_id: str,
        output_handle: str,
    ) -> Optional[str]:
        """
        Resolve the next node ID by matching the output handle against
        outgoing edge conditions.

        Routing priority:
        1. Exact source_handle match
        2. 'default' handle fallback
        3. Unconditional edge (no condition)
        4. None (terminal node — no outgoing edges)
        """
        outgoing = self._adjacency.get(current_node_id, [])

        if not outgoing:
            return None

        # Priority 1: Exact handle match
        for edge in outgoing:
            if edge.condition and edge.condition.source_handle == output_handle:
                # Evaluate optional expression if present
                if edge.condition.expression:
                    if self._evaluate_expression(edge.condition.expression):
                        return edge.target
                else:
                    return edge.target

        # Priority 2: Default handle fallback
        for edge in outgoing:
            if edge.condition and edge.condition.source_handle == "default":
                return edge.target

        # Priority 3: Unconditional edge (no condition specified)
        for edge in outgoing:
            if edge.condition is None:
                return edge.target

        logger.warning(
            "no_matching_edge",
            execution_id=self._ctx.execution_id,
            node_id=current_node_id,
            output_handle=output_handle,
            available_handles=[
                e.condition.source_handle for e in outgoing if e.condition
            ],
        )
        return None

    def _evaluate_expression(self, expression: str) -> bool:
        """
        Evaluate a deterministic expression against the execution context
        variables. Supports simple dot-notation variable access:
            "variables.appointment_confirmed == true"
            "variables.intent == 'booking'"

        Uses a safe evaluation approach — NO eval(). Parses simple
        equality/inequality expressions only.
        """
        try:
            expression = expression.strip()

            # Parse "left op right" format
            for op in ("==", "!=", ">=", "<=", ">", "<"):
                if op in expression:
                    left_str, right_str = expression.split(op, 1)
                    left_val = self._resolve_variable_path(left_str.strip())
                    right_val = self._parse_literal(right_str.strip())

                    if op == "==":
                        return left_val == right_val
                    elif op == "!=":
                        return left_val != right_val
                    elif op == ">=":
                        return left_val >= right_val  # type: ignore
                    elif op == "<=":
                        return left_val <= right_val  # type: ignore
                    elif op == ">":
                        return left_val > right_val  # type: ignore
                    elif op == "<":
                        return left_val < right_val  # type: ignore

            # If no operator found, treat as truthy check
            val = self._resolve_variable_path(expression)
            return bool(val)

        except Exception as exc:
            logger.warning(
                "expression_eval_failed",
                expression=expression,
                error=str(exc),
            )
            return False

    def _resolve_variable_path(self, path: str) -> Any:
        """
        Resolve a dot-notation path against execution context.
        Supports: "variables.field", "node_outputs.node_id.field"
        """
        parts = path.split(".")
        if not parts:
            return None

        root = parts[0]
        remainder = parts[1:]

        if root == "variables":
            obj: Any = self._ctx.variables
        elif root == "node_outputs":
            obj = self._ctx.node_outputs
        else:
            return path  # Treat as literal string

        for part in remainder:
            if isinstance(obj, dict):
                obj = obj.get(part)
            else:
                return None

        return obj

    @staticmethod
    def _parse_literal(value: str) -> Any:
        """Parse a literal value from an expression string."""
        # Strip quotes for string literals
        if (value.startswith("'") and value.endswith("'")) or \
           (value.startswith('"') and value.endswith('"')):
            return value[1:-1]

        # Boolean
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False

        # Numeric
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value

    # ── Error Handling ───────────────────────────────────────────────

    async def _fail(self, node_id: str, error: Exception) -> None:
        """Transition the execution to FAILED state."""
        self._ctx.status = ExecutionStatus.FAILED
        self._ctx.node_outputs[f"__error_{node_id}"] = {
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        await self._emit_trace(
            node_id=node_id,
            node_type="error",
            event="execution_failed",
            data={"error": str(error)},
        )
        logger.error(
            "node_execution_failed",
            execution_id=self._ctx.execution_id,
            node_id=node_id,
            error=str(error),
            traceback=traceback.format_exc(),
        )

    # ── Trace Emission ───────────────────────────────────────────────

    async def _emit_trace(
        self,
        node_id: str,
        node_type: str,
        event: str,
        duration_ms: Optional[float] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Emit a TraceEvent to the Redis Pub/Sub channel
        `trace:{execution_id}` for real-time SSE frontend consumption.
        """
        trace = TraceEvent(
            execution_id=self._ctx.execution_id,
            tenant_id=self._ctx.tenant_id,
            node_id=node_id,
            node_type=node_type,
            event=event,  # type: ignore[arg-type]
            step=self._ctx.step_count,
            duration_ms=duration_ms,
            data=data or {},
        )

        if self._redis is not None:
            try:
                channel = f"trace:{self._ctx.execution_id}"
                await self._redis.publish(channel, trace.model_dump_json())
            except Exception as exc:
                logger.warning(
                    "trace_emit_failed",
                    channel=f"trace:{self._ctx.execution_id}",
                    error=str(exc),
                )

        logger.debug(
            "trace_event",
            event=event,
            node_id=node_id,
            step=self._ctx.step_count,
        )
