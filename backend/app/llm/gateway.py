"""
Unified LLM Gateway — single interface routing between Groq Console and
OpenRouter with automatic retries, token/cost telemetry, CAG cache_control
injection, and Groq Whisper audio transcription.

Design decisions:
- Two persistent AsyncOpenAI clients (Groq + OpenRouter) initialized once.
- CAG injection modifies system messages in-place with cache_control markers
  for OpenRouter/Anthropic provider-level KV prefix caching.
- Whisper transcription is Groq-exclusive (LPU-accelerated inference).
- All methods return structured telemetry dicts for FinOps accumulation.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import structlog
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import get_settings
from app.llm.telemetry import CostCalculator, LLMUsageTelemetry
from app.schemas.execution import ChatMessage
from app.schemas.graph import LLMProvider

logger = structlog.get_logger(__name__)

# Retryable exceptions from the OpenAI SDK
_RETRYABLE_EXCEPTIONS = (
    Exception,  # Narrowed below; we catch and filter in the retry predicate
)


def _is_retryable(exc: BaseException) -> bool:
    """Only retry on transient server/rate-limit errors."""
    from openai import APIStatusError, APITimeoutError, APIConnectionError

    if isinstance(exc, (APITimeoutError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code in (429, 500, 502, 503):
        return True
    return False


class LLMGateway:
    """
    Unified async LLM client supporting Groq Console and OpenRouter.

    Usage:
        gateway = LLMGateway()
        response, telemetry = await gateway.chat_completion(
            provider=LLMProvider.GROQ,
            model="llama-3.3-70b-versatile",
            messages=[...],
        )
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._groq_client = AsyncOpenAI(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=30.0,
            max_retries=0,  # We handle retries ourselves
        )
        self._openrouter_client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=120.0,
            max_retries=0,
        )
        self._cost_calculator = CostCalculator()

    def _get_client(self, provider: LLMProvider) -> AsyncOpenAI:
        if provider == LLMProvider.GROQ:
            return self._groq_client
        return self._openrouter_client

    # ── CAG Cache Control Injection ──────────────────────────────────

    @staticmethod
    def _inject_cag_cache_control(
        messages: List[Dict[str, Any]],
        cag_context_blocks: Sequence[str],
    ) -> List[Dict[str, Any]]:
        """
        Inject cache_control markers on system messages containing CAG
        context for OpenRouter/Anthropic provider-level KV prefix caching.

        The system message content is restructured into the Anthropic
        multi-block format with cache_control: {"type": "ephemeral"}
        on the static knowledge block.
        """
        if not cag_context_blocks:
            return messages

        cag_text = "\n\n---\n\n".join(cag_context_blocks)
        injected: List[Dict[str, Any]] = []

        for msg in messages:
            if msg.get("role") == "system":
                # Restructure into multi-block format for Anthropic caching
                original_content = msg.get("content", "")
                injected.append({
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": original_content,
                        },
                        {
                            "type": "text",
                            "text": f"\n\n<knowledge_base>\n{cag_text}\n</knowledge_base>",
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                })
            else:
                injected.append(msg)

        return injected

    # ── Core Chat Completion ─────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=lambda retry_state: logger.warning(
            "llm_retry",
            attempt=retry_state.attempt_number,
            error=str(retry_state.outcome.exception()) if retry_state.outcome else "unknown",
        ),
    )
    async def _raw_chat_completion(
        self,
        client: AsyncOpenAI,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Execute raw chat completion with retry logic."""
        response = await client.chat.completions.create(**kwargs)
        return response

    async def chat_completion(
        self,
        provider: LLMProvider,
        model: str,
        messages: List[ChatMessage],
        *,
        # Optional parameters
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str | Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        # OpenRouter-specific
        fallback_models: Optional[List[str]] = None,
        # CAG injection
        cag_context_blocks: Optional[List[str]] = None,
        # Structured output (Groq)
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> tuple[ChatCompletion, LLMUsageTelemetry]:
        """
        Unified chat completion across providers.

        Returns:
            Tuple of (ChatCompletion, LLMUsageTelemetry)
        """
        client = self._get_client(provider)
        start_time = time.monotonic()

        # Serialize messages to API format
        api_messages: List[Dict[str, Any]] = [m.to_api_dict() for m in messages]

        # Inject CAG cache_control for OpenRouter
        if provider == LLMProvider.OPENROUTER and cag_context_blocks:
            api_messages = self._inject_cag_cache_control(
                api_messages, cag_context_blocks
            )

        # Build kwargs
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        # Response format — structured JSON outputs
        if json_schema and provider == LLMProvider.GROQ:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema,
            }
        elif response_format:
            kwargs["response_format"] = response_format

        # OpenRouter fallback models & provider preferences
        if provider == LLMProvider.OPENROUTER:
            extra_body: Dict[str, Any] = {}
            if fallback_models:
                extra_body["models"] = [model] + fallback_models
            if cag_context_blocks:
                extra_body["provider"] = {
                    "allow_fallbacks": True,
                    "require_parameters": True,
                }
            if extra_body:
                kwargs["extra_body"] = extra_body

        # Execute
        try:
            response = await self._raw_chat_completion(client, **kwargs)
        except Exception as exc:
            # Check if retryable before re-raising
            if not _is_retryable(exc):
                raise
            raise

        elapsed_ms = (time.monotonic() - start_time) * 1000

        # Build telemetry
        usage = response.usage
        telemetry = self._cost_calculator.compute(
            provider=provider,
            model=response.model or model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            cached_tokens=getattr(usage, "prompt_tokens_details", None)
            and getattr(usage.prompt_tokens_details, "cached_tokens", 0)
            or 0,
            latency_ms=elapsed_ms,
        )

        logger.info(
            "llm_completion",
            provider=provider.value,
            model=response.model,
            prompt_tokens=telemetry.prompt_tokens,
            completion_tokens=telemetry.completion_tokens,
            cached_tokens=telemetry.cached_tokens,
            cost_usd=f"${telemetry.cost_usd:.6f}",
            latency_ms=f"{telemetry.latency_ms:.0f}ms",
        )

        return response, telemetry

    # ── Groq Whisper Audio Transcription ─────────────────────────────

    async def transcribe_audio(
        self,
        file_bytes: bytes,
        *,
        filename: str = "audio.ogg",
        model: str = "whisper-large-v3-turbo",
        language: Optional[str] = None,
    ) -> str:
        """
        Transcribe audio using Groq's Whisper API.
        Groq's LPU delivers sub-second transcription for typical voice notes.

        Args:
            file_bytes: Raw audio file bytes (.ogg, .mp3, .wav, etc.)
            filename: Filename hint for the API
            model: Whisper model variant
            language: Optional ISO 639-1 language code

        Returns:
            Transcribed text string
        """
        start_time = time.monotonic()

        kwargs: Dict[str, Any] = {
            "model": model,
            "file": (filename, file_bytes),
        }
        if language:
            kwargs["language"] = language

        transcription = await self._groq_client.audio.transcriptions.create(**kwargs)
        elapsed_ms = (time.monotonic() - start_time) * 1000

        text = transcription.text.strip()

        logger.info(
            "whisper_transcription",
            model=model,
            text_length=len(text),
            latency_ms=f"{elapsed_ms:.0f}ms",
        )

        return text

    # ── Cleanup ──────────────────────────────────────────────────────

    async def close(self) -> None:
        """Close underlying HTTP clients."""
        await self._groq_client.close()
        await self._openrouter_client.close()
