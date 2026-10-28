"""
OpenRouter / Anthropic CAG (Cache-Augmented Generation) cache_control
injection utilities.

This module provides helpers for constructing system prompts that leverage
provider-level KV prefix caching. The primary injection logic lives in
LLMGateway._inject_cag_cache_control; this module offers higher-level
composition helpers for building cache-optimized message sequences.
"""
from __future__ import annotations

from typing import Any, Dict, List


def build_cag_system_prompt(
    base_system_prompt: str,
    cag_blocks: List[str],
    temporal_grounding: str,
) -> Dict[str, Any]:
    """
    Build an Anthropic-compatible multi-block system message with
    cache_control markers for static knowledge injection.

    The knowledge block is marked as ephemeral for prefix caching.
    The base instructions and temporal grounding are not cached
    (they change per-request).

    Args:
        base_system_prompt: The core agent instructions
        cag_blocks: Static knowledge markdown blocks
        temporal_grounding: Current time/date string

    Returns:
        System message dict with multi-block content and cache_control
    """
    content_blocks: List[Dict[str, Any]] = []

    # Block 1: Static knowledge (cacheable)
    if cag_blocks:
        knowledge_text = "\n\n---\n\n".join(cag_blocks)
        content_blocks.append({
            "type": "text",
            "text": f"<knowledge_base>\n{knowledge_text}\n</knowledge_base>",
            "cache_control": {"type": "ephemeral"},
        })

    # Block 2: Core instructions + temporal grounding (not cached)
    instructions_text = f"{base_system_prompt}\n\n{temporal_grounding}"
    content_blocks.append({
        "type": "text",
        "text": instructions_text,
    })

    return {
        "role": "system",
        "content": content_blocks,
    }


def estimate_cache_savings(
    total_prompt_tokens: int,
    cached_tokens: int,
    input_cost_per_1m: float,
    cached_cost_per_1m: float,
) -> Dict[str, float]:
    """
    Calculate estimated cache savings for FinOps dashboards.

    Returns dict with:
        - full_cost: What it would have cost without caching
        - actual_cost: What it actually cost with caching
        - savings: Dollar amount saved
        - cache_hit_ratio: Fraction of prompt tokens that were cached
    """
    full_cost = (total_prompt_tokens / 1_000_000) * input_cost_per_1m

    uncached = max(0, total_prompt_tokens - cached_tokens)
    actual_cost = (
        (uncached / 1_000_000) * input_cost_per_1m
        + (cached_tokens / 1_000_000) * cached_cost_per_1m
    )

    return {
        "full_cost_usd": round(full_cost, 8),
        "actual_cost_usd": round(actual_cost, 8),
        "savings_usd": round(full_cost - actual_cost, 8),
        "cache_hit_ratio": round(cached_tokens / max(total_prompt_tokens, 1), 4),
    }
