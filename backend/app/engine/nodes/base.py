"""
Abstract Base Node — the universal contract for all executable node types
in the visual workflow engine.

Every concrete node implementation (ReactAgentNode, RouterNode, etc.) must:
1. Declare a `ConfigSchema` class attribute (a Pydantic BaseModel) that
   validates the `config` dict from GraphNode.
2. Implement `async def execute(ctx, config) -> NodeOutput`.

The ConfigSchema is resolved at runtime by the NodeRegistry when the
GraphRunner loads a WorkflowBlueprint, ensuring that misconfigured nodes
fail fast at blueprint validation time rather than mid-execution.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Type

from pydantic import BaseModel

from app.schemas.execution import ExecutionContext, NodeOutput


class EmptyConfig(BaseModel):
    """Default config schema for nodes that require no configuration."""
    pass


class BaseNode(ABC):
    """
    Abstract base class for all workflow node types.

    Subclasses MUST:
    - Set `ConfigSchema` to a Pydantic BaseModel subclass
    - Implement `execute()` returning a `NodeOutput`

    Subclasses MAY override:
    - `validate_config()` for additional cross-field validation
    - `on_error()` for custom error recovery logic
    """

    ConfigSchema: ClassVar[Type[BaseModel]] = EmptyConfig

    @abstractmethod
    async def execute(
        self,
        ctx: ExecutionContext,
        config: BaseModel,
    ) -> NodeOutput:
        """
        Execute this node's logic within the given execution context.

        Args:
            ctx: The mutable execution context (conversation history,
                 variables, knowledge context, telemetry).
            config: Node-specific configuration, validated against
                    this node's ConfigSchema.

        Returns:
            NodeOutput containing updated variables, routing handle,
            messages to append, and telemetry data.
        """
        ...

    def validate_config(self, config: BaseModel) -> None:
        """
        Optional hook for cross-field config validation beyond what
        the Pydantic schema enforces. Raise ValueError on invalid config.
        """
        pass

    async def on_error(
        self,
        ctx: ExecutionContext,
        config: BaseModel,
        error: Exception,
    ) -> NodeOutput | None:
        """
        Optional error recovery hook. If this returns a NodeOutput,
        the engine uses it instead of failing the execution.
        Return None to propagate the error.
        """
        return None

    @classmethod
    def parse_config(cls, raw_config: dict) -> BaseModel:
        """
        Parse and validate a raw config dict against this node's schema.
        Raises ValidationError on invalid config.
        """
        return cls.ConfigSchema.model_validate(raw_config)
