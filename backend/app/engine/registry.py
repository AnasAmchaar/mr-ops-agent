"""
Decorator-based Node Type Registry — maps NodeType enum values to their
concrete BaseNode subclass implementations.

Usage:
    @register_node(NodeType.INTENT_ROUTER)
    class RouterNode(BaseNode):
        ...

    # Later, at runtime:
    node_cls = NodeRegistry.get(NodeType.INTENT_ROUTER)
    node = node_cls()
    config = node.parse_config(raw_config)
    output = await node.execute(ctx, config)

The registry is populated at import time via the decorator, and the
GraphRunner uses it to instantiate nodes dynamically from the blueprint.
"""
from __future__ import annotations

from typing import Dict, Type

import structlog

from app.engine.nodes.base import BaseNode
from app.schemas.graph import NodeType

logger = structlog.get_logger(__name__)


class NodeRegistry:
    """
    Global registry mapping NodeType -> BaseNode subclass.
    Thread-safe for reads (populated at module import time).
    """

    _registry: Dict[NodeType, Type[BaseNode]] = {}

    @classmethod
    def register(cls, node_type: NodeType, node_cls: Type[BaseNode]) -> None:
        """Register a node class for a given NodeType."""
        if node_type in cls._registry:
            existing = cls._registry[node_type].__name__
            raise ValueError(
                f"NodeType '{node_type.value}' is already registered to "
                f"'{existing}'. Duplicate registration from '{node_cls.__name__}'."
            )
        cls._registry[node_type] = node_cls
        logger.debug(
            "node_registered",
            node_type=node_type.value,
            node_class=node_cls.__name__,
        )

    @classmethod
    def get(cls, node_type: NodeType) -> Type[BaseNode]:
        """
        Retrieve the BaseNode subclass for a given NodeType.
        Raises KeyError if the node type has not been registered.
        """
        if node_type not in cls._registry:
            registered = [nt.value for nt in cls._registry]
            raise KeyError(
                f"No node implementation registered for '{node_type.value}'. "
                f"Registered types: {registered}"
            )
        return cls._registry[node_type]

    @classmethod
    def list_registered(cls) -> Dict[NodeType, str]:
        """Return a snapshot of all registered node types and their class names."""
        return {nt: ncls.__name__ for nt, ncls in cls._registry.items()}

    @classmethod
    def is_registered(cls, node_type: NodeType) -> bool:
        """Check if a node type has a registered implementation."""
        return node_type in cls._registry

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations. Used only in testing."""
        cls._registry.clear()


def register_node(node_type: NodeType):
    """
    Class decorator that registers a BaseNode subclass in the NodeRegistry.

    Usage:
        @register_node(NodeType.INTENT_ROUTER)
        class RouterNode(BaseNode):
            ...
    """
    def decorator(cls: Type[BaseNode]) -> Type[BaseNode]:
        if not issubclass(cls, BaseNode):
            raise TypeError(
                f"@register_node can only decorate BaseNode subclasses. "
                f"'{cls.__name__}' does not inherit from BaseNode."
            )
        NodeRegistry.register(node_type, cls)
        return cls
    return decorator
