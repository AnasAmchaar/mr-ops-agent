"""
Human-in-the-Loop (HITL) Approval Node — suspends workflow execution
and waits for human review before proceeding.

Use cases:
- High-stakes tool executions (e.g., sending emails, scheduling meetings)
- Content approval before customer-facing replies
- Escalation gates for sensitive operations

Design:
- Serializes the full ExecutionContext to Redis for persistence
- Sets status to PAUSED_FOR_HUMAN
- Sends a notification to the business owner (via WhatsApp or dashboard)
- The `resume_execution()` method on ExecutionContextManager transitions
  back to RUNNING and continues the DAG from the next node

The node outputs `suspend_execution=True`, which the GraphRunner interprets
as a break signal in the execution loop.
"""
from __future__ import annotations

import json
import time
from typing import Any, ClassVar, Dict, List, Optional, Type

import structlog
from pydantic import BaseModel, Field

from app.core.database import get_redis
from app.engine.context import ExecutionContextManager
from app.engine.nodes.base import BaseNode
from app.engine.registry import register_node
from app.schemas.execution import ChatMessage, ExecutionContext, NodeOutput
from app.schemas.graph import NodeType

logger = structlog.get_logger(__name__)


# ── Config Schema ────────────────────────────────────────────────────

class HITLApprovalConfig(BaseModel):
    """Configuration for the HITL Approval node."""
    approval_message: str = Field(
        default="An action requires your approval before proceeding.",
        description="Message shown to the human reviewer",
    )
    review_fields: List[str] = Field(
        default_factory=list,
        description=(
            "Variable names from ctx.variables to include in the review payload. "
            "If empty, all current variables are included."
        ),
    )
    timeout_seconds: int = Field(
        default=86400,
        ge=60,
        le=604800,
        description="Maximum time to wait for approval (default: 24h, max: 7 days)",
    )
    notification_channel: str = Field(
        default="dashboard",
        description="Where to send the approval notification: 'dashboard', 'whatsapp', 'email'",
    )
    notification_recipient: Optional[str] = Field(
        default=None,
        description="Recipient for notifications (phone number for WhatsApp, email for email)",
    )
    auto_approve_after: Optional[int] = Field(
        default=None,
        ge=60,
        description="Optional: auto-approve after N seconds if no response",
    )
    approval_buttons: List[str] = Field(
        default=["Approve", "Reject"],
        min_length=2,
        description="Button labels for the approval prompt",
    )
    rejection_handle: str = Field(
        default="rejected",
        description="Edge handle to route to if the reviewer rejects",
    )
    approval_handle: str = Field(
        default="approved",
        description="Edge handle to route to upon approval",
    )


# ── Node Implementation ─────────────────────────────────────────────

@register_node(NodeType.HITL_APPROVAL)
class HITLApprovalNode(BaseNode):
    """
    Human-in-the-Loop approval gate.

    When executed, this node:
    1. Builds a review payload from the current execution state
    2. Persists the ExecutionContext to Redis
    3. Publishes a notification to the review channel
    4. Returns `suspend_execution=True` to pause the DAG

    The workflow resumes when a human calls the resume API endpoint
    with an approval decision, which triggers ExecutionContextManager.resume().
    """

    ConfigSchema: ClassVar[Type[BaseModel]] = HITLApprovalConfig

    def _build_review_payload(
        self,
        config: HITLApprovalConfig,
        ctx: ExecutionContext,
    ) -> Dict[str, Any]:
        """
        Build the review payload that the human reviewer will see.
        Includes selected variables, conversation context, and metadata.
        """
        # Select variables to include in review
        if config.review_fields:
            review_vars = {
                k: v for k, v in ctx.variables.items()
                if k in config.review_fields
            }
        else:
            review_vars = dict(ctx.variables)

        # Extract last few conversation messages for context
        recent_messages = [
            {"role": m.role, "content": m.content}
            for m in ctx.conversation_history[-5:]
            if m.content
        ]

        return {
            "execution_id": ctx.execution_id,
            "workflow_id": ctx.workflow_id,
            "tenant_id": ctx.tenant_id,
            "session_id": ctx.session_id,
            "approval_message": config.approval_message,
            "variables": review_vars,
            "recent_conversation": recent_messages,
            "buttons": config.approval_buttons,
            "timeout_seconds": config.timeout_seconds,
            "step_count": ctx.step_count,
            "current_node_id": ctx.current_node_id,
        }

    async def _publish_notification(
        self,
        config: HITLApprovalConfig,
        review_payload: Dict[str, Any],
        ctx: ExecutionContext,
    ) -> None:
        """
        Publish the approval notification to the configured channel.

        - 'dashboard': Publish to Redis Pub/Sub for real-time dashboard
        - 'whatsapp': Queue a WhatsApp message to the business owner
        - 'email': Queue an email notification

        WhatsApp and email dispatch are handled by the respective channel
        modules (Phase 5). Here we publish the event and let the channel
        workers pick it up.
        """
        redis = get_redis()

        # Always publish to the dashboard channel for real-time UI
        dashboard_channel = f"hitl:{ctx.tenant_id}"
        await redis.publish(
            dashboard_channel,
            json.dumps(review_payload, default=str),
        )

        # Store the pending approval for API retrieval
        approval_key = f"hitl:pending:{ctx.tenant_id}:{ctx.execution_id}"
        await redis.setex(
            approval_key,
            config.timeout_seconds,
            json.dumps(review_payload, default=str),
        )

        # Queue channel-specific notifications
        if config.notification_channel == "whatsapp" and config.notification_recipient:
            notification = {
                "type": "hitl_approval_request",
                "channel": "whatsapp",
                "recipient": config.notification_recipient,
                "message": config.approval_message,
                "execution_id": ctx.execution_id,
                "tenant_id": ctx.tenant_id,
                "buttons": config.approval_buttons,
            }
            await redis.lpush(
                f"notifications:{ctx.tenant_id}",
                json.dumps(notification, default=str),
            )
        elif config.notification_channel == "email" and config.notification_recipient:
            notification = {
                "type": "hitl_approval_request",
                "channel": "email",
                "recipient": config.notification_recipient,
                "subject": f"Approval Required: {config.approval_message}",
                "body": json.dumps(review_payload, indent=2, default=str),
                "execution_id": ctx.execution_id,
                "tenant_id": ctx.tenant_id,
            }
            await redis.lpush(
                f"notifications:{ctx.tenant_id}",
                json.dumps(notification, default=str),
            )

        logger.info(
            "hitl_notification_published",
            channel=config.notification_channel,
            tenant_id=ctx.tenant_id,
            execution_id=ctx.execution_id,
        )

    async def execute(
        self,
        ctx: ExecutionContext,
        config: BaseModel,
    ) -> NodeOutput:
        """
        Suspend execution and await human approval.

        1. Build review payload
        2. Persist context to Redis
        3. Publish notification
        4. Return with suspend_execution=True
        """
        cfg = config if isinstance(config, HITLApprovalConfig) else HITLApprovalConfig.model_validate(config.model_dump())
        start_time = time.monotonic()

        # Build the review payload
        review_payload = self._build_review_payload(cfg, ctx)

        # Persist the execution context for later resumption
        redis = get_redis()
        ctx_manager = ExecutionContextManager(redis)
        await ctx_manager.save(ctx, ttl_seconds=cfg.timeout_seconds)
        await ctx_manager.index_session(ctx, ttl_seconds=cfg.timeout_seconds)

        # Publish the notification
        await self._publish_notification(cfg, review_payload, ctx)

        elapsed_ms = (time.monotonic() - start_time) * 1000

        logger.info(
            "hitl_suspended",
            execution_id=ctx.execution_id,
            tenant_id=ctx.tenant_id,
            timeout_seconds=cfg.timeout_seconds,
            notification_channel=cfg.notification_channel,
        )

        return NodeOutput(
            updated_variables={
                "hitl_pending": True,
                "hitl_approval_message": cfg.approval_message,
                "hitl_timeout_seconds": cfg.timeout_seconds,
            },
            next_handle=cfg.approval_handle,  # Pre-set for post-resume routing
            suspend_execution=True,
            messages_to_append=[
                ChatMessage(
                    role="assistant",
                    content=(
                        "I need to pause here for a moment. Your request requires "
                        "human approval before I can proceed. A reviewer has been "
                        "notified and will respond shortly."
                    ),
                ),
            ],
            telemetry={
                "latency_ms": elapsed_ms,
                "notification_channel": cfg.notification_channel,
                "timeout_seconds": cfg.timeout_seconds,
            },
        )
