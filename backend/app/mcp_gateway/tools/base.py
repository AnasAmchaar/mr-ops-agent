"""
MCP Tool Definitions — typed contract shared by all internal MCP servers.

Each server module declares `TOOLS: tuple[ToolDefinition, ...]`. A
ToolDefinition binds:
- its OpenAI function-calling name / description,
- a strictly typed Pydantic v2 input model (arguments are validated before
  any handler runs — LLM output is NEVER trusted raw),
- an async handler receiving the validated input, the tenant's access token.

The same definitions generate:
1. OpenAI tool schemas for the LLM (via ClientManager.get_tool_schemas).
2. Official-SDK MCP servers (FastMCP) for out-of-process deployment —
   the handlers are the single source of execution truth for both paths.

HTTP layer: one shared httpx.AsyncClient per event loop with tenacity-backed
retries for transient failures (429 / 5xx / network).
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, Type

import httpx
import structlog
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = structlog.get_logger(__name__)

GOOGLE_CALENDAR_API = "https://www.googleapis.com/calendar/v3"
GOOGLE_GMAIL_API = "https://gmail.googleapis.com/gmail/v1"

# ── Tool Definition Contract ─────────────────────────────────────────

class ToolDefinition(BaseModel):
    """
    Declarative, strictly typed MCP tool registration.

    handler signature:
        async def handler(input: <PydanticModel>, access_token: str,
                          **kwargs) -> dict[str, Any]
    """
    model_config = {"arbitrary_types_allowed": True}

    name: str
    server: str                # owning server key, e.g. "google_calendar"
    description: str
    input_model: Type[BaseModel]
    handler: Callable[..., Awaitable[Dict[str, Any]]]

    def to_openai_tool(self) -> Dict[str, Any]:
        """OpenAI function-calling schema derived from the Pydantic model."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": _pydantic_to_json_schema(self.input_model),
            },
        }


def _pydantic_to_json_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    """JSON Schema with LLM-friendly annotations and strict field requirements."""
    schema: Dict[str, Any] = model.model_json_schema()
    schema.pop("title", None)
    props = schema.get("properties", {})
    defs = schema.pop("$defs", None)
    if defs:
        schema["definitions"] = defs
    required = schema.get("required", [])
    for prop in props.values():  # give the LLM deterministic markup for enums
        any_of = prop.pop("anyOf", None)
        if any_of and len(any_of) == 2:
            for variant in any_of:
                if variant.get("type") == "string" and variant.get("enum"):
                    prop["type"] = "string"
                    prop["enum"] = variant["enum"]
    schema["required"] = required  # keep required fields strict
    return schema


# ── Shared Google API HTTP Layer ─────────────────────────────────────

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=30.0)
    return _client


async def close_http_client() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, (httpx.ConnectError, httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError)):
        return True
    if isinstance(exc, GoogleAPIError):
        return exc.status_code in (408, 429, 500, 502, 503, 504)
    return False


class GoogleAPIError(Exception):
    """Typed Google REST API failure with HTTP status and structured detail."""

    def __init__(self, status_code: int, message: str, detail: Any = None) -> None:
        super().__init__(f"Google API HTTP {status_code}: {message}")
        self.status_code = status_code
        self.message = message
        self.detail = detail


async def google_request(
    method: str,
    url: str,
    *,
    access_token: str,
    json_body: Dict[str, Any] | None = None,
    params: Dict[str, Any] | None = None,
    ok_statuses: tuple[int, ...] = (200,),
) -> Dict[str, Any]:
    """
    Execute one retry-hardened authenticated Google REST call.

    Retries transient failures with exponential backoff (3 attempts).
    Non-success statuses raise GoogleAPIError carrying status + body detail.
    """
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    async for attempt in AsyncRetrying(
        retry=retry_if_exception(_is_transient),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        reraise=True,
    ):
        with attempt:
            client = _get_client()
            try:
                resp = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=json_body,
                    params=params,
                )
            except httpx.TimeoutException:
                raise
            except httpx.HTTPError as exc:
                logger.warning("google_request_network_error", url=url, error=str(exc))
                raise GoogleAPIError(503, f"network error: {exc}") from exc

            if resp.status_code not in ok_statuses:
                detail: Any = None
                try:
                    detail = resp.json()
                except Exception:
                    detail = resp.text[:500]
                message = _extract_human_message(detail)
                err = GoogleAPIError(resp.status_code, message, detail)
                if not _is_transient(err):
                    raise err
                attempt_retryable = _is_transient(err)
                if attempt_retryable:
                    logger.warning(
                        "google_request_retry",
                        url=url,
                        status=resp.status_code,
                        attempt=attempt.retry_state.attempt_number,
                    )
                raise err

            if resp.status_code == 204 or not resp.content:
                return {}
            return resp.json()

    raise GoogleAPIError(500, "unreachable: retry loop exhausted")  # pragma: no cover


def _extract_human_message(detail: Any) -> str:
    """Pull Google's human-readable error message out of the error envelope."""
    if isinstance(detail, dict):
        err = detail.get("error")
        if isinstance(err, dict):
            return str(err.get("message", err))
        if isinstance(err, str):
            return err
        if "message" in detail:
            return str(detail["message"])
    return str(detail)


def google_auth_failed(exc: GoogleAPIError) -> bool:
    """True when the failure indicates a client-side auth problem."""
    return exc.status_code in (401, 403)
