"""
WhatsApp Trigger Node — entry point for WhatsApp-initiated workflows.

This node is the first node in any WhatsApp-triggered workflow graph.
It doesn't perform its own execution logic at runtime — instead, it
serves as the entry point that the WhatsApp webhook handler targets
when dispatching an incoming message to the GraphRunner.

The webhook handler:
1. Receives the WhatsApp message
2. Debounces burst messages (Redis sliding-window)
3. Transcribes audio if needed (Groq Whisper)
4. Creates an ExecutionContext with the user message in conversation_history
5. Sets current_node_id = this trigger node's ID
6. Invokes GraphRunner.run()

At that point, the trigger node simply passes through to the next node
via the "default" handle, carrying any WhatsApp-specific metadata
(sender phone, message type, media URLs) in ctx.variables.

Full implementation with audio transcription wiring is in Phase 5.
"""
from __future__ import annotations

from typing import ClassVar, Dict, List, Optional, Type

import structlog
from pydantic import BaseModel, Field

from app.engine.nodes.base import BaseNode
from app.engine.registry import register_node
from app.schemas.execution import ExecutionContext, NodeOutput
from app.schemas.graph import NodeType

logger = structlog.get_logger(__name__)


class WhatsAppTriggerConfig(BaseModel):
    """Configuration for the WhatsApp trigger node."""
    welcome_message: Optional[str] = Field(
        default=None,
        description="Optional welcome message for first-time conversations",
    )
    supported_message_types: List[str] = Field(
        default=["text", "audio", "image"],
        description="Message types this trigger accepts",
    )
    phone_number_id: Optional[str] = Field(
        default=None,
        description="WhatsApp Business Phone Number ID (from Meta Business Manager)",
    )


@register_node(NodeType.WHATSAPP_TRIGGER)
class WhatsAppTriggerNode(BaseNode):
    """
    WhatsApp workflow entry point.

    At runtime, the incoming message data is already populated in
    ctx.variables by the webhook handler. This node simply validates
    and passes through.
    """

    ConfigSchema: ClassVar[Type[BaseModel]] = WhatsAppTriggerConfig

    async def execute(
        self,
        ctx: ExecutionContext,
        config: BaseModel,
    ) -> NodeOutput:
        cfg = config if isinstance(config, WhatsAppTriggerConfig) else WhatsAppTriggerConfig.model_validate(config.model_dump())

        # Extract WhatsApp metadata that the webhook handler placed in variables
        sender_phone = ctx.variables.get("wa_sender_phone", ctx.session_id)
        message_type = ctx.variables.get("wa_message_type", "text")
        user_text = ctx.variables.get("wa_user_text", "")

        logger.info(
            "wa_trigger_activated",
            execution_id=ctx.execution_id,
            sender_phone=sender_phone,
            message_type=message_type,
            text_length=len(user_text),
        )

        updated_vars: Dict[str, object] = {
            "wa_sender_phone": sender_phone,
            "wa_message_type": message_type,
            "channel": "whatsapp",
        }

        return NodeOutput(
            updated_variables=updated_vars,
            next_handle="default",
            telemetry={
                "message_type": message_type,
                "channel": "whatsapp",
            },
        )
