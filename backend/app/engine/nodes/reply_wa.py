"""
WhatsApp Reply Node — sends the agent's response back to the user
via the WhatsApp Cloud API.

This is typically the terminal node in a WhatsApp-triggered workflow.
It reads the response from ctx.variables (set by the ReactAgentNode
or other upstream nodes) and dispatches it through the WhatsApp
Cloud API outbound sender.

Full implementation with interactive buttons, template messages,
and media support is delivered in Phase 5. This module provides the
node registration and config schema.
"""
from __future__ import annotations

import time
from typing import ClassVar, Dict, List, Optional, Type

import structlog
from pydantic import BaseModel, Field

from app.engine.nodes.base import BaseNode
from app.engine.registry import register_node
from app.schemas.execution import ChatMessage, ExecutionContext, NodeOutput
from app.schemas.graph import NodeType

logger = structlog.get_logger(__name__)


class WhatsAppReplyConfig(BaseModel):
    """Configuration for the WhatsApp reply node."""
    response_variable: str = Field(
        default="agent_response",
        description="ctx.variables key containing the response text to send",
    )
    fallback_message: str = Field(
        default="I'm sorry, I wasn't able to generate a response. Please try again.",
        description="Sent if the response_variable is empty or missing",
    )
    phone_number_id: Optional[str] = Field(
        default=None,
        description="WhatsApp Business Phone Number ID (overrides tenant default)",
    )
    max_message_length: int = Field(
        default=4096,
        ge=1,
        le=65536,
        description="Maximum message length before splitting",
    )


@register_node(NodeType.WHATSAPP_REPLY)
class WhatsAppReplyNode(BaseNode):
    """
    Sends the agent's response back via WhatsApp Cloud API.

    In Phase 2, this node prepares the outbound payload and stores it
    in ctx.variables. Phase 5 replaces this with actual API dispatch
    via the WhatsApp client module.
    """

    ConfigSchema: ClassVar[Type[BaseModel]] = WhatsAppReplyConfig

    async def execute(
        self,
        ctx: ExecutionContext,
        config: BaseModel,
    ) -> NodeOutput:
        cfg = config if isinstance(config, WhatsAppReplyConfig) else WhatsAppReplyConfig.model_validate(config.model_dump())
        start_time = time.monotonic()

        # Resolve the response text
        response_text = ctx.variables.get(cfg.response_variable)
        if not response_text or not isinstance(response_text, str):
            response_text = cfg.fallback_message
            logger.warning(
                "wa_reply_using_fallback",
                execution_id=ctx.execution_id,
                variable=cfg.response_variable,
            )

        # Truncate if needed
        if len(response_text) > cfg.max_message_length:
            response_text = response_text[:cfg.max_message_length - 3] + "..."

        recipient_phone = ctx.variables.get("wa_sender_phone", ctx.session_id)

        # In Phase 5, this dispatches via the WhatsApp client.
        # For now, we store the outbound payload for the webhook handler to send.
        outbound_payload = {
            "recipient_phone": recipient_phone,
            "message_text": response_text,
            "phone_number_id": cfg.phone_number_id,
        }

        elapsed_ms = (time.monotonic() - start_time) * 1000

        logger.info(
            "wa_reply_prepared",
            execution_id=ctx.execution_id,
            recipient=recipient_phone,
            text_length=len(response_text),
        )

        return NodeOutput(
            updated_variables={
                "wa_outbound_payload": outbound_payload,
                "wa_reply_sent": True,
                "wa_reply_text": response_text,
            },
            next_handle="default",
            messages_to_append=[
                ChatMessage(role="assistant", content=response_text),
            ],
            telemetry={
                "latency_ms": elapsed_ms,
                "message_length": len(response_text),
                "channel": "whatsapp",
            },
        )
