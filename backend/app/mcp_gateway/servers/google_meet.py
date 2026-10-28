"""
Google Meet MCP Server — standalone Meet link provisioning.

Tool:
- meet_provision_link: mints a Google Meet URL via the Calendar API by
  creating a short placeholder event carrying conferenceData (No attendees
  by default, so the link can be embedded in any outbound reply/email).

The Calendar API is the only official bulk way to mint hangoutsMeet links
outside the Meet REST Spaces API; a five-minute hold event is created on
the tenant's primary calendar and can be deleted afterwards
(`delete_placeholder=True`) once the true meeting event exists.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import structlog
from pydantic import BaseModel, Field, field_validator

from app.mcp_gateway.servers.google_calendar import _extract_meet_link
from app.mcp_gateway.tools.base import GOOGLE_CALENDAR_API, ToolDefinition, google_request

logger = structlog.get_logger(__name__)

SERVER_NAME = "google_meet"


# ── Input Models ─────────────────────────────────────────────────────

class ProvisionLinkInput(BaseModel):
    """Minimal input to mint a standalone Meet link."""
    summary: str = Field(..., min_length=1, max_length=300, description="Meeting topic / title")
    timezone: str = Field("UTC", description="IANA timezone for the hold event")
    duration_minutes: int = Field(15, ge=1, le=120, description="Hold event length")
    attendee_emails: List[str] = Field(
        default_factory=list,
        description="Optional guests to embed on the hold event",
    )
    delete_placeholder: bool = Field(
        False,
        description=(
            "True: delete the hold event right after minting the link. "
            "Leave False if an agent will later schedule the real event on this calendar."
        ),
    )

    @field_validator("timezone")
    @classmethod
    def _valid_timezone(cls, v: str) -> str:
        try:
            from zoneinfo import ZoneInfo
            ZoneInfo(v)
        except Exception as exc:
            raise ValueError(f"Unknown IANA timezone: {v!r}") from exc
        return v

    @field_validator("attendee_emails")
    @classmethod
    def _valid_emails(cls, v: List[str]) -> List[str]:
        import re
        email_re = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
        for item in v:
            if not email_re.match(item):
                raise ValueError(f"Invalid attendee email: {item!r}")
        return v


# ── Handlers ─────────────────────────────────────────────────────────

async def provision_link(args: ProvisionLinkInput, access_token: str) -> Dict[str, Any]:
    """
    Mint a Meet URL via a conferenceData-equipped Calendar event.

    The hold event starts 1 minute in the future (server-side UTC clock) and
    lasts `duration_minutes`. The returned payload carries the event IDs so
    the calling agent can reference or delete the placeholder later.
    """
    now_utc = datetime.now(timezone.utc)
    start = now_utc + timedelta(minutes=1)
    end = start + timedelta(minutes=args.duration_minutes)

    event_body: Dict[str, Any] = {
        "summary": args.summary,
        "description": (
            "Google Meet hold event provisioned by Mr.Ops. "
            "Safe to delete once the real meeting exists."
        ),
        "start": {"dateTime": start.isoformat(), "timeZone": args.timezone},
        "end": {"dateTime": end.isoformat(), "timeZone": args.timezone},
        "attendees": [{"email": e} for e in args.attendee_emails],
        "conferenceData": {
            "createRequest": {
                "requestId": f"mrops-meet-{now_utc.strftime('%Y%m%d%H%M%S%f')}",
                "conferenceSolutionKey": {"type": "hangoutsMeet"},
            }
        },
        "reminders": {"useDefault": False, "overrides": []},
    }

    created = await google_request(
        "POST",
        f"{GOOGLE_CALENDAR_API}/calendars/primary/events",
        access_token=access_token,
        params={"conferenceDataVersion": 1, "sendUpdates": "none"},
        json_body=event_body,
    )

    event_id = created.get("id", "")
    meet_link = _extract_meet_link(created) or ""

    if delete_requested := args.delete_placeholder:
        try:
            await google_request(
                "DELETE",
                f"{GOOGLE_CALENDAR_API}/calendars/primary/events/{event_id}",
                access_token=access_token,
                ok_statuses=(200, 204, 410),
            )
        except Exception as exc:
            logger.warning("meet_placeholder_delete_failed", event_id=event_id, error=str(exc))

    logger.info("meet_link_provisioned", event_id=event_id, link_present=bool(meet_link))

    return {
        "meet_link": meet_link,
        "placeholder_event_id": event_id,
        "placeholder_deleted": delete_requested,
        "summary": args.summary,
    }


# ── Registry ─────────────────────────────────────────────────────────

TOOLS = (
    ToolDefinition(
        name="meet_provision_link",
        server=SERVER_NAME,
        description=(
            "Provision a standalone Google Meet video link (https://meet.google.com/...). "
            "Creates an optional placeholder calendar event to mint the link; returns "
            "the Meet URL and the placeholder event ID."
        ),
        input_model=ProvisionLinkInput,
        handler=provision_link,
    ),
)


def create_mcp_server() -> Any:
    """Official-SDK FastMCP wrapper for out-of-process deployment."""
    from mcp.server.fastmcp import FastMCP

    from app.mcp_gateway.servers.google_calendar import attached_access_token as token_ctx

    mcp = FastMCP(SERVER_NAME)

    @mcp.tool()
    async def meet_provision_link(
        summary: str,
        timezone: str = "UTC",
        duration_minutes: int = 15,
        attendee_emails: List[str] | None = None,
        delete_placeholder: bool = False,
    ) -> Dict[str, Any]:
        args = ProvisionLinkInput(
            summary=summary,
            timezone=timezone,
            duration_minutes=duration_minutes,
            attendee_emails=attendee_emails or [],
            delete_placeholder=delete_placeholder,
        )
        return await provision_link(args, token_ctx.get())

    return mcp
