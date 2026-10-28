"""
Graph Serialization Contracts — the canonical JSON schema that the React Flow
frontend compiles into and the DAG engine consumes.

Every field is strictly typed. The `config` dict on each GraphNode is validated
against node-type-specific Pydantic schemas at runtime via the NodeRegistry.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enumerations ─────────────────────────────────────────────────────

class NodeType(str, Enum):
    """All supported node types in the visual workflow builder."""
    WHATSAPP_TRIGGER = "whatsapp_trigger"
    INTENT_ROUTER = "intent_router"
    REACT_AGENT = "react_agent"
    KNOWLEDGE_RETRIEVER = "knowledge_retriever"
    MCP_TOOL_ACTION = "mcp_tool_action"
    HITL_APPROVAL = "hitl_approval"
    WHATSAPP_REPLY = "whatsapp_reply"


class LLMProvider(str, Enum):
    """Supported LLM inference providers."""
    GROQ = "groq"
    OPENROUTER = "openrouter"


# ── Edge Condition ───────────────────────────────────────────────────

class EdgeCondition(BaseModel):
    """
    Conditional routing on an edge.
    `source_handle` maps to the output branch key emitted by a node
    (e.g. 'booking_intent', 'faq_intent', 'default').
    `expression` allows optional deterministic evaluation rules for
    variable-based branching (e.g. "variables.appointment_confirmed == true").
    """
    source_handle: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Output branch key from the source node",
    )
    expression: Optional[str] = Field(
        None,
        max_length=512,
        description="Optional deterministic evaluation rule",
    )


# ── Graph Edge ───────────────────────────────────────────────────────

class GraphEdge(BaseModel):
    """Directed edge connecting two nodes, optionally with a condition."""
    id: str = Field(..., min_length=1, max_length=64)
    source: str = Field(..., min_length=1, description="Source node ID")
    target: str = Field(..., min_length=1, description="Target node ID")
    condition: Optional[EdgeCondition] = None


# ── Graph Node ───────────────────────────────────────────────────────

class GraphNode(BaseModel):
    """
    A single node in the visual workflow graph.
    `config` is intentionally a loose dict here — it is validated against
    the node-type-specific ConfigSchema when the engine loads the blueprint.
    """
    id: str = Field(..., min_length=1, max_length=64)
    type: NodeType
    label: str = Field(..., min_length=1, max_length=128)
    position: Dict[str, float] = Field(
        ...,
        description="React Flow position: {'x': float, 'y': float}",
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Validated against node-specific Pydantic schema at runtime",
    )

    @field_validator("position")
    @classmethod
    def _validate_position(cls, v: Dict[str, float]) -> Dict[str, float]:
        if "x" not in v or "y" not in v:
            raise ValueError("position must contain 'x' and 'y' keys")
        return v


# ── Workflow Blueprint ───────────────────────────────────────────────

class WorkflowBlueprint(BaseModel):
    """
    Complete serialized workflow graph — stored as JSONB in PostgreSQL.
    This is the execution contract between the React Flow canvas and
    the backend DAG engine.
    """
    workflow_id: str = Field(..., min_length=1)
    tenant_id: str = Field(..., min_length=1, description="Multi-tenant scope key (UUID)")
    name: str = Field(..., min_length=1, max_length=256)
    version: int = Field(default=1, ge=1)
    entry_node_id: str = Field(..., min_length=1)
    nodes: List[GraphNode] = Field(..., min_length=1)
    edges: List[GraphEdge] = Field(default_factory=list)
    max_steps: int = Field(
        default=25,
        ge=1,
        le=100,
        description="Circuit breaker for cyclic graphs",
    )

    @model_validator(mode="after")
    def _validate_graph_integrity(self) -> "WorkflowBlueprint":
        """Ensure structural integrity of the graph definition."""
        node_ids = {n.id for n in self.nodes}

        # Entry node must exist
        if self.entry_node_id not in node_ids:
            raise ValueError(
                f"entry_node_id '{self.entry_node_id}' does not reference "
                f"any node in the graph. Available: {node_ids}"
            )

        # All edge endpoints must reference existing nodes
        for edge in self.edges:
            if edge.source not in node_ids:
                raise ValueError(
                    f"Edge '{edge.id}' source '{edge.source}' not found in nodes"
                )
            if edge.target not in node_ids:
                raise ValueError(
                    f"Edge '{edge.id}' target '{edge.target}' not found in nodes"
                )

        # Detect duplicate node IDs
        if len(node_ids) != len(self.nodes):
            raise ValueError("Duplicate node IDs detected in graph")

        # Detect duplicate edge IDs
        edge_ids = [e.id for e in self.edges]
        if len(set(edge_ids)) != len(edge_ids):
            raise ValueError("Duplicate edge IDs detected in graph")

        return self

    def get_node(self, node_id: str) -> GraphNode:
        """Look up a node by ID. Raises KeyError if not found."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(f"Node '{node_id}' not found in workflow '{self.workflow_id}'")

    def get_outgoing_edges(self, node_id: str) -> List[GraphEdge]:
        """Return all edges originating from the given node."""
        return [e for e in self.edges if e.source == node_id]
