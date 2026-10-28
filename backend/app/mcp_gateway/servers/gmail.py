"""
Gmail MCP Server — thread-aware send/reply.

Tools:
- gmail_send_message: send a MIME-encoded message via Gmail REST
  (users.messages.send), optionally threading a reply onto an existing
  thread_id so replies appear inside the original Gmail conversation.
- gmail_read_thread: fetch full thread contents (messages + headers +
  decoded body text) for context before answering or replying.

Wire format notes:
- Gmail accepts only base64url-encoded raw RFC-2822 messages.
- Threading requires: In-Reply-To + References headers + threadId + matching
  subject prefix ("Re: ...") for reliability clients-side.
"""
from __future__ import annotations

import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, Optional

import structlog
from pydantic import BaseModel, Field, field_validator

from app.mcp_gateway.tools.base import GOOGLE_GMAIL_API, ToolDefinition, google_request

logger = structlog.get_logger(__name__)

SERVER_NAME = "gmail"


# ── Input Models ─────────────────────────────────────────────────────

class SendMessageInput(BaseModel):
    """Outbound email payload for Gmail users.messages.send."""
    to: str = Field(..., min_length=3, max_length=320, description="Recipient email address")
    subject: str = Field(..., min_length=1, max_length=998)
    body_html: str = Field(..., min_length=1, description="Email body as HTML")
    thread_id: Optional[str] = Field(
        None,
        description=(
            "Gmail threadId to send this as a threaded reply. When provided, "
            "In-Reply-To/References headers are derived from the thread's "
            "latest message so it lands inside the same Gmail conversation."
        ),
    )
    in_reply_to_message_id: Optional[str] = Field(
        None,
        description="Message-ID header of the message being replied to (when known)",
    )

    @field_validator("to")
    @classmethod
    def _valid_email(cls, v: str) -> str:
        import re
        cleaned = v.strip()
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", cleaned):
            raise ValueError(f"Invalid recipient email: {v!r}")
        return cleaned


class ReadThreadInput(BaseModel):
    """Thread fetch parameters."""
    thread_id: str = Field(..., min_length=10, description="Gmail threadId to fetch")
    max_messages: int = Field(20, ge=1, le=100, description="Cap on messages returned")


# ── Handlers ─────────────────────────────────────────────────────────

def _build_raw_message(
    args: SendMessageInput,
    reply_headers: Dict[str, str],
    from_header: str,
) -> str:
    """Build a RFC-2822 base64url raw message for the Gmail send endpoint."""
    msg = MIMEMultipart("alternative")
    msg["To"] = args.to
    msg["Subject"] = args.subject
    msg["From"] = from_header

    for header_name, value in reply_headers.items():
        msg[header_name] = value

    # Plain-text fallback improves deliverability and screen-reader UX
    import html as html_mod
    import re as re_mod
    text_fallback = re_mod.sub(r"<[^>]+>", " ", html_mod.unescape(args.body_html))
    text_fallback = re_mod.sub(r"\s+", " ", text_fallback).strip()

    msg.attach(MIMEText(text_fallback, "plain", "utf-8"))
    msg.attach(MIMEText(args.body_html, "html", "utf-8"))

    return base64.urlsafe_b64encode(msg.as_bytes()).decode("ascii")


def _decode_message_text(message: Dict[str, Any]) -> str:
    """Walk parts and decode text/plain or text/html body to text."""
    payload = message.get("payload", {})

    def walk(node: Dict[str, Any]) -> Optional[str]:
        mime = node.get("mimeType", "")
        if mime in ("text/plain", "text/html"):
            data = node.get("body", {}).get("data")
            if data:
                raw = base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
                import re
                if mime == "text/html":
                    raw = re.sub(r"<[^>]+>", " ", raw)
                return raw
        for part in node.get("parts", []):
            found = walk(part)
            if found:
                return found
        return None

    body = walk(payload)
    return (body or message.get("snippet", "") or "").strip()


async def send_message(args: SendMessageInput, access_token: str) -> Dict[str, Any]:
    """
    Send (or thread-reply) an email via Gmail REST.

    When thread_id is set, the thread is fetched first so In-Reply-To and
    References headers point at the real conversation chain — required for
    Gmail clients to render the reply inside the original thread.
    """
    from_header: str = ""
    reply_headers: Dict[str, str] = {}

    if args.thread_id:
        thread = await google_request(
            "GET",
            f"{GOOGLE_GMAIL_API}/users/me/threads/{args.thread_id}",
            access_token=access_token,
            params={"format": "full"},
        )
        messages = thread.get("messages", [])
        if messages:
            last_headers = {
                h["name"].lower(): h["value"]
                for h in messages[-1].get("payload", {}).get("headers", [])
            }
            last_message_id = last_headers.get("message-id")
            references_parts: list[str] = []
            for m in messages:
                m_headers = {
                    h["name"].lower(): h["value"]
                    for h in m.get("payload", {}).get("headers", [])
                }
                mid = m_headers.get("message-id")
                if mid:
                    references_parts.append(mid)
            references = " ".join(references_parts[-10:])
            if last_message_id:
                reply_headers["In-Reply-To"] = last_message_id
                reply_headers["References"] = references[:2000]
            if not args.subject.lower().startswith("re:"):
                args = args.model_copy(update={"subject": f"Re: {args.subject}"})

    raw = _build_raw_message(args, reply_headers, from_header or "Mr.Ops Agent <me>")

    send_payload: Dict[str, Any] = {"raw": raw}
    if args.thread_id:
        send_payload["threadId"] = args.thread_id

    result = await google_request(
        "POST",
        f"{GOOGLE_GMAIL_API}/users/me/messages/send",
        access_token=access_token,
        json_body=send_payload,
    )

    message_id = result.get("id", "")
    logger.info("gmail_message_sent", message_id=message_id, thread_id=args.thread_id or "new")

    return {
        "message_id": message_id,
        "thread_id": result.get("threadId", args.thread_id),
        "to": args.to,
        "subject": args.subject,
        "sent_as_reply": bool(args.thread_id),
    }


async def read_thread(args: ReadThreadInput, access_token: str) -> Dict[str, Any]:
    """
    Fetch a thread's messages with decoded bodies for agent context.
    """
    thread = await google_request(
        "GET",
        f"{GOOGLE_GMAIL_API}/users/me/threads/{args.thread_id}",
        access_token=access_token,
        params={"format": "full"},
    )

    messages_out: list[Dict[str, Any]] = []
    for m in thread.get("messages", [])[: args.max_messages]:
        headers = {h["name"].lower(): h["value"] for h in m.get("payload", {}).get("headers", [])}
        messages_out.append(
            {
                "message_id": m.get("id", ""),
                "thread_id": thread.get("id", args.thread_id),
                # Only derived metadata is exposed to the LLM — no raw secrets
                "from": headers.get("from", ""),
                "to": headers.get("to", ""),
                "date": headers.get("date", ""),
                "subject": headers.get("subject", ""),
                "snippet": m.get("snippet", ""),
            }
        )

    return {
        "thread_id": thread.get("id", args.thread_id),
        "message_count": len(messages_out),
        "messages": messages_out,
    }


# ── Registry ─────────────────────────────────────────────────────────

TOOLS = (
    ToolDefinition(
        name="gmail_send_message",
        server=SERVER_NAME,
        description=(
            "Send an email via Gmail, optionally as a threaded reply when a "
            "thread_id is supplied. HTML body supported. Returns the new "
            "message and thread IDs."
        ),
        input_model=SendMessageInput,
        handler=send_message,
    ),
    ToolDefinition(
        name="gmail_read_thread",
        server=SERVER_NAME,
        description=(
            "Read a Gmail thread by threadId; returns the messages, senders, "
            "dates and decoded body snippets inside the conversation."
        ),
        input_model=ReadThreadInput,
        handler=read_thread,
    ),
)


def create_mcp_server() -> Any:
    """Official-SDK FastMCP wrapper for out-of-process deployment."""
    from mcp.server.fastmcp import FastMCP

    from app.mcp_gateway.servers.google_calendar import attached_access_token as token_ctx

    mcp = FastMCP(SERVER_NAME)

    @mcp.tool()
    async def gmail_send_message(
        to: str, subject: str, body_html: str,
        thread_id: str | None = None, in_reply_to_message_id: str | None = None,
    ) -> Dict[str, Any]:
        args = SendMessageInput(
            to=to, subject=subject, body_html=body_html,
            thread_id=thread_id, in_reply_to_message_id=in_reply_to_message_id,
        )
        return await send_message(args, token_ctx.get())

    @mcp.tool()
    async def gmail_read_thread(thread_id: str, max_messages: int = 20) -> Dict[str, Any]:
        return await read_thread(ReadThreadInput(thread_id=thread_id, max_messages=max_messages), token_ctx.get())

    return mcp
