"""
LLM Token Metering, Latency Tracking & FinOps Cost Calculator.

Pricing is approximate and based on published per-1M-token rates.
Updated pricing periodically — these serve as baseline FinOps estimates.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.graph import LLMProvider


class LLMUsageTelemetry(BaseModel):
    """Structured telemetry returned from every LLM call."""
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0

    def to_dict(self) -> dict[str, float | int | str]:
        return self.model_dump()


# ── Pricing Table (per 1M tokens) ───────────────────────────────────
# Format: model_prefix -> (input_cost_per_1M, output_cost_per_1M, cached_input_per_1M)

_PRICING: dict[str, tuple[float, float, float]] = {
    # Groq Console (very competitive pricing)
    "llama-3.3-70b": (0.59, 0.79, 0.59),
    "llama-3.1-8b": (0.05, 0.08, 0.05),
    "whisper-large-v3": (0.0, 0.0, 0.0),  # Audio pricing is per-second, handled separately

    # OpenRouter — Anthropic
    "anthropic/claude-sonnet-4": (3.0, 15.0, 0.30),
    "anthropic/claude-3.7-sonnet": (3.0, 15.0, 0.30),
    "anthropic/claude-3.5-sonnet": (3.0, 15.0, 0.30),

    # OpenRouter — OpenAI
    "openai/gpt-4o": (2.50, 10.0, 1.25),
    "openai/gpt-4o-mini": (0.15, 0.60, 0.075),

    # OpenRouter — DeepSeek
    "deepseek/deepseek-chat-v3": (0.27, 1.10, 0.07),
    "deepseek/deepseek-r1": (0.55, 2.19, 0.14),

    # OpenRouter — Google
    "google/gemini-2.5-pro": (1.25, 10.0, 0.3125),
    "google/gemini-2.5-flash": (0.15, 0.60, 0.0375),
}

# Fallback pricing for unknown models
_DEFAULT_PRICING = (1.0, 3.0, 0.5)


class CostCalculator:
    """Compute estimated USD cost from token counts and model identity."""

    def _find_pricing(self, model: str) -> tuple[float, float, float]:
        """Match model string against pricing table prefixes."""
        model_lower = model.lower()
        for prefix, pricing in _PRICING.items():
            if model_lower.startswith(prefix.lower()):
                return pricing
        return _DEFAULT_PRICING

    def compute(
        self,
        provider: LLMProvider,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int,
        latency_ms: float,
    ) -> LLMUsageTelemetry:
        """Compute cost and return structured telemetry."""
        input_per_1m, output_per_1m, cached_per_1m = self._find_pricing(model)

        # Non-cached input tokens
        uncached_input = max(0, prompt_tokens - cached_tokens)

        cost = (
            (uncached_input / 1_000_000) * input_per_1m
            + (cached_tokens / 1_000_000) * cached_per_1m
            + (completion_tokens / 1_000_000) * output_per_1m
        )

        return LLMUsageTelemetry(
            provider=provider.value,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            latency_ms=latency_ms,
            cost_usd=round(cost, 8),
        )
