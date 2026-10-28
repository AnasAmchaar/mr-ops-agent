"""
Runtime Execution State Contracts — the living context that flows through
every node during a workflow run.

ExecutionContext is the single mutable state object threaded through the
DAG engine. It accumulates conversation history, extracted variables,
knowledge context, and FinOps telemetry as nodes execute.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ── Chat Message ─────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    """
    OpenAI-compatible message format used throughout the system.
    Supports system, user, assistant, and tool messages.
    """
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

    def to_api_dict(self) -> Dict[str, Any]:
        """Serialize to OpenAI API-compatible dict, stripping None fields."""
        d: Dict[str, Any] = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.name is not None:
            d["name"] = self.name
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls is not None:
            d["tool_calls"] = self.tool_calls
        return d


# ── Execution Status ─────────────────────────────────────────────────

class ExecutionStatus(str, Enum):
    """Lifecycle states of a workflow execution."""
    RUNNING = "RUNNING"
    PAUSED_FOR_HUMAN = "PAUSED_FOR_HUMAN"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# ── Node Output ──────────────────────────────────────────────────────

class NodeOutput(BaseModel):
    """
    Return value from every BaseNode.execute() call.
    Controls routing, state updates, and execution flow.
    """
    updated_variables: Dict[str, Any] = Field(
        default_factory=dict,
        description="Variables to merge into ExecutionContext.variables",
    )
    next_handle: str = Field(
        default="default",
        description="Output handle key for edge routing (e.g. 'booking_intent')",
    )
    suspend_execution: bool = Field(
        default=False,
        description="If True, pauses execution (HITL approval, async wait)",
    )
    messages_to_append: List[ChatMessage] = Field(
        default_factory=list,
        description="Messages to append to conversation history",
    )
    telemetry: Dict[str, Any] = Field(
        default_factory=dict,
        description="Node-level telemetry (tokens, latency, cost)",
    )


# ── Trace Event ──────────────────────────────────────────────────────

class TraceEvent(BaseModel):
    """
    Real-time execution trace event emitted to Redis Pub/Sub for the
    frontend SSE trace viewer. Each event maps to a node state transition.
    """
    execution_id: str
    tenant_id: str
    node_id: str
    node_type: str
    event: Literal["node_started", "node_completed", "node_failed", "execution_completed", "execution_failed"]
    step: int
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    duration_ms: Optional[float] = None
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Event-specific payload (output handle, error message, token counts)",
    )


# ── Execution Context ───────────────────────────────────────────────

class ExecutionContext(BaseModel):
    """
    The single mutable state object that flows through the DAG engine.
    Accumulates conversation history, extracted variables, knowledge
    context, and FinOps telemetry as nodes execute.

    Strictly scoped by tenant_id — no cross-tenant state leakage.
    """
    execution_id: str = Field(..., min_length=1)
    tenant_id: str = Field(..., min_length=1, description="Multi-tenant scope key")
    workflow_id: str = Field(..., min_length=1)
    session_id: str = Field(
        ...,
        min_length=1,
        description="Channel-specific session key (e.g. WhatsApp phone number)",
    )
    current_node_id: str = Field(..., min_length=1)
    status: ExecutionStatus = ExecutionStatus.RUNNING
    step_count: int = Field(default=0, ge=0)

    # ── Temporal & Tenant Grounding ──────────────────────────────────
    business_timezone: str = Field(default="UTC")
    current_iso_time: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )

    # ── Memory & State ───────────────────────────────────────────────
    conversation_history: List[ChatMessage] = Field(default_factory=list)
    variables: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted entities (email, date, meet_link, etc.)",
    )
    cag_context_blocks: List[str] = Field(
        default_factory=list,
        description="Cached static markdown blocks for provider-level KV prefix caching",
    )
    rag_retrieved_chunks: List[str] = Field(
        default_factory=list,
        description="Dynamically retrieved RAG chunks from Qdrant",
    )
    node_outputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Accumulated outputs keyed by node_id",
    )

    # ── FinOps & Telemetry ───────────────────────────────────────────
    total_prompt_tokens: int = Field(default=0, ge=0)
    total_completion_tokens: int = Field(default=0, ge=0)
    cached_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)

    @field_validator("business_timezone")
    @classmethod
    def _validate_timezone(cls, v: str) -> str:
        # Basic validation — accept IANA timezone strings
        if not v or len(v) > 64:
            raise ValueError(f"Invalid timezone: {v}")
        return v

    def merge_node_output(self, node_id: str, output: NodeOutput) -> None:
        """
        Apply a NodeOutput to the execution context.
        Merges variables, appends messages, records node output, and
        accumulates telemetry counters.
        """
        # Merge extracted variables
        self.variables.update(output.updated_variables)

        # Append conversation messages
        self.conversation_history.extend(output.messages_to_append)

        # Record node output for downstream reference
        self.node_outputs[node_id] = {
            "next_handle": output.next_handle,
            "variables": output.updated_variables,
            "telemetry": output.telemetry,
        }

        # Accumulate token telemetry
        tel = output.telemetry
        self.total_prompt_tokens += tel.get("prompt_tokens", 0)
        self.total_completion_tokens += tel.get("completion_tokens", 0)
        self.cached_tokens += tel.get("cached_tokens", 0)
        self.estimated_cost_usd += tel.get("cost_usd", 0.0)

        # Handle suspension
        if output.suspend_execution:
            self.status = ExecutionStatus.PAUSED_FOR_HUMAN

    def get_temporal_grounding(self) -> str:
        """
        Return a formatted temporal context string for LLM system prompts.
        Prevents relative-date hallucinations (e.g., 'next Monday').
        """
        try:
            dt = datetime.fromisoformat(self.current_iso_time)
            day_name = dt.strftime("%A")
            date_str = dt.strftime("%Y-%m-%d")
            time_str = dt.strftime("%H:%M:%S")
            return (
                f"Current date: {date_str} ({day_name}). "
                f"Current time: {time_str} ({self.business_timezone}). "
                f"Use these exact values for any date/time reasoning."
            )
        except (ValueError, TypeError):
            return f"Current ISO time: {self.current_iso_time} ({self.business_timezone})"
