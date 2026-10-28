"""
Intent Router Node — ultra-fast Groq-powered intent classifier.

Uses Groq's structured JSON output mode with `llama-3.3-70b-versatile`
to classify user intent in <200ms. The classified intent maps directly
to a `source_handle` on the node's outgoing edges, enabling deterministic
DAG routing without brittle regex or keyword matching.

Design:
- System prompt defines available intents with descriptions
- Groq's JSON schema mode guarantees valid structured output
- The `intent` field from the response becomes `NodeOutput.next_handle`
- Supports confidence-based fallback to a "default" or "escalate" handle
"""
from __future__ import annotations

import json
import time
from typing import Any, ClassVar, Dict, List, Optional, Type

import structlog
from pydantic import BaseModel, Field

from app.engine.nodes.base import BaseNode
from app.engine.registry import register_node
from app.llm.gateway import LLMGateway
from app.main import get_llm_gateway
from app.schemas.execution import ChatMessage, ExecutionContext, NodeOutput
from app.schemas.graph import LLMProvider, NodeType

logger = structlog.get_logger(__name__)


# ── Config Schema ────────────────────────────────────────────────────

class IntentDefinition(BaseModel):
    """A single intent that the router can classify."""
    name: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Intent key — must match an outgoing edge's source_handle",
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Natural language description of when this intent applies",
    )
    examples: List[str] = Field(
        default_factory=list,
        description="Example user messages that should trigger this intent",
    )


class RouterNodeConfig(BaseModel):
    """Configuration for the Intent Router node."""
    intents: List[IntentDefinition] = Field(
        ...,
        min_length=2,
        description="Available intents to classify. Must have at least 2.",
    )
    default_intent: str = Field(
        default="default",
        description="Fallback intent when confidence is below threshold",
    )
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to accept a classification",
    )
    model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model for intent classification",
    )
    system_prompt_override: Optional[str] = Field(
        default=None,
        description="Optional custom system prompt (replaces default template)",
    )


# ── Structured Output Schema ────────────────────────────────────────

class RouterResponse(BaseModel):
    """Expected structured JSON response from the router LLM."""
    intent: str = Field(..., description="Classified intent name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    reasoning: str = Field(..., description="Brief reasoning for the classification")


# ── Node Implementation ─────────────────────────────────────────────

@register_node(NodeType.INTENT_ROUTER)
class RouterNode(BaseNode):
    """
    Ultra-fast intent classification node powered by Groq LPU inference.

    Takes the latest user message from conversation history, classifies
    it against the configured intent definitions, and routes to the
    matching outgoing edge via `next_handle`.
    """

    ConfigSchema: ClassVar[Type[BaseModel]] = RouterNodeConfig

    def _build_system_prompt(
        self,
        config: RouterNodeConfig,
        ctx: ExecutionContext,
    ) -> str:
        """Build the intent classification system prompt."""
        if config.system_prompt_override:
            return config.system_prompt_override

        intent_descriptions = "\n".join(
            f"- **{intent.name}**: {intent.description}"
            + (f"\n  Examples: {', '.join(repr(e) for e in intent.examples[:3])}"
               if intent.examples else "")
            for intent in config.intents
        )

        return (
            "You are an ultra-precise intent classifier. Your ONLY job is to "
            "classify the user's latest message into exactly one of the following intents.\n\n"
            f"## Available Intents\n{intent_descriptions}\n\n"
            f"## Temporal Context\n{ctx.get_temporal_grounding()}\n\n"
            "## Rules\n"
            "1. Classify based on the user's LATEST message, considering conversation context.\n"
            "2. If the message is ambiguous or doesn't clearly match any intent, "
            f"use '{config.default_intent}'.\n"
            "3. Set confidence between 0.0 and 1.0 based on how certain you are.\n"
            "4. Keep reasoning to one concise sentence.\n"
            "5. NEVER invent intent names not listed above.\n"
        )

    def _build_json_schema(self, config: RouterNodeConfig) -> Dict[str, Any]:
        """Build the Groq JSON schema for structured output."""
        valid_intents = [i.name for i in config.intents]
        if config.default_intent not in valid_intents:
            valid_intents.append(config.default_intent)

        return {
            "name": "router_classification",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": valid_intents,
                        "description": "The classified intent",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Classification confidence 0.0-1.0",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief reasoning for this classification",
                    },
                },
                "required": ["intent", "confidence", "reasoning"],
                "additionalProperties": False,
            },
        }

    def _extract_latest_user_message(self, ctx: ExecutionContext) -> str:
        """Extract the most recent user message from conversation history."""
        for msg in reversed(ctx.conversation_history):
            if msg.role == "user" and msg.content:
                return msg.content
        return ""

    async def execute(
        self,
        ctx: ExecutionContext,
        config: BaseModel,
    ) -> NodeOutput:
        """
        Classify intent using Groq structured JSON output.

        Flow:
        1. Build system prompt with intent definitions
        2. Send to Groq with JSON schema enforcement
        3. Parse structured response
        4. Apply confidence threshold
        5. Return intent as next_handle for edge routing
        """
        cfg = config if isinstance(config, RouterNodeConfig) else RouterNodeConfig.model_validate(config.model_dump())
        gateway: LLMGateway = get_llm_gateway()

        # Build classification prompt
        system_prompt = self._build_system_prompt(cfg, ctx)
        user_message = self._extract_latest_user_message(ctx)

        if not user_message:
            logger.warning(
                "router_no_user_message",
                execution_id=ctx.execution_id,
            )
            return NodeOutput(
                next_handle=cfg.default_intent,
                telemetry={"skipped": True, "reason": "no_user_message"},
            )

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_message),
        ]

        # Groq structured output call
        start_time = time.monotonic()
        response, telemetry = await gateway.chat_completion(
            provider=LLMProvider.GROQ,
            model=cfg.model,
            messages=messages,
            temperature=0.0,  # Deterministic classification
            max_tokens=256,
            json_schema=self._build_json_schema(cfg),
        )
        elapsed_ms = (time.monotonic() - start_time) * 1000

        # Parse structured response
        raw_content = response.choices[0].message.content or "{}"
        try:
            parsed = RouterResponse.model_validate_json(raw_content)
        except Exception as exc:
            logger.error(
                "router_parse_failed",
                raw_content=raw_content,
                error=str(exc),
            )
            return NodeOutput(
                next_handle=cfg.default_intent,
                telemetry={
                    **telemetry.to_dict(),
                    "parse_error": str(exc),
                    "latency_ms": elapsed_ms,
                },
            )

        # Apply confidence threshold
        final_intent = parsed.intent
        if parsed.confidence < cfg.confidence_threshold:
            logger.info(
                "router_low_confidence_fallback",
                intent=parsed.intent,
                confidence=parsed.confidence,
                threshold=cfg.confidence_threshold,
                fallback=cfg.default_intent,
            )
            final_intent = cfg.default_intent

        # Validate intent is a known handle
        valid_intents = {i.name for i in cfg.intents}
        valid_intents.add(cfg.default_intent)
        if final_intent not in valid_intents:
            logger.warning(
                "router_unknown_intent",
                intent=final_intent,
                valid=list(valid_intents),
            )
            final_intent = cfg.default_intent

        logger.info(
            "router_classified",
            execution_id=ctx.execution_id,
            intent=final_intent,
            confidence=parsed.confidence,
            reasoning=parsed.reasoning,
            latency_ms=f"{elapsed_ms:.0f}ms",
        )

        return NodeOutput(
            updated_variables={
                "classified_intent": final_intent,
                "intent_confidence": parsed.confidence,
                "intent_reasoning": parsed.reasoning,
            },
            next_handle=final_intent,
            telemetry={
                **telemetry.to_dict(),
                "classified_intent": final_intent,
                "confidence": parsed.confidence,
                "reasoning": parsed.reasoning,
                "latency_ms": elapsed_ms,
            },
        )
