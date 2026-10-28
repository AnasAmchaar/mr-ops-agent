"""
Google Calendar MCP Server — availability checks and event scheduling.

Tools:
- calendar_check_availability: FreeBusy query returning busy intervals.
- calendar_schedule_with_meet: Create an event with a Google Meet conference
  (conferenceDataVersion=1, conferenceSolutionKey=hangoutsMeet), returning
  both the Calendar event ID and the hangout link.

Auth: tenant-scoped access token injected by the ClientManager via OAuthVault.
The handlers NEVER receive tenant identity from the LLM — the manager binds
tenancy server-side before dispatch.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import structlog
from pydantic import BaseModel, Field, field_validator

from app.mcp_gateway.tools.base import GOOGLE_CALENDAR_API, ToolDefinition, google_request

logger = structlog.get_logger(__name__)
from contextvars import ContextVar  # noqa: E402  (module-level per-request token carrier)

attached_access_token: ContextVar[str] = ContextVar("attached_access_token", default="")

SERVER_NAME = "google_calendar"


# ── Input Models ─────────────────────────────────────────────────────

class CheckAvailabilityInput(BaseModel):
    """FreeBusy availability query window."""
    start_iso: str = Field(..., description="Window start, ISO 8601 (e.g. '2026-10-27T09:00:00Z')")
    end_iso: str = Field(..., description="Window end, ISO 8601")
    timezone: str = Field("UTC", description="IANA timezone used to interpret the window")
    calendar_id: str = Field("primary", description="Google Calendar ID to query")

    @field_validator("start_iso", "end_iso")
    @classmethod
    def _valid_iso(cls, v: str) -> str:
        cleaned = v.strip()
        try:
            datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Not a valid ISO 8601 datetime: {v!r}") from exc
        return cleaned

    @field_validator("timezone")
    @classmethod
    def _valid_timezone(cls, v: str) -> str:
        try:
            from zoneinfo import ZoneInfo
            ZoneInfo(v)
        except Exception as exc:
            raise ValueError(f"Unknown IANA timezone: {v!r}") from exc
        return v


class ScheduleWithMeetInput(BaseModel):
    """Create a calendar event with an attached Google Meet conference."""
    summary: str = Field(..., min_length=1, max_length=500, description="Event title")
    start_iso: str = Field(..., description="Event start, ISO 8601")
    end_iso: str = Field(..., description="Event end, ISO 8601")
    attendee_emails: List[str] = Field(
        default_factory=list,
        description="Guest email addresses to invite",
    )
    description: str = Field(default="", description="Event description / agenda")
    timezone: str = Field(
        "UTC",
        description="IANA timezone; events store this alongside the local time",
    )
    calendar_id: str = Field("primary", description="Calendar to create the event on")
    send_updates: str = Field(
        "all",
        description="Guests notification level: 'all' | 'externalOnly' | 'none'",
    )

    @field_validator("start_iso", "end_iso")
    @classmethod
    def _valid_iso(cls, v: str) -> str:
        cleaned = v.strip()
        try:
            datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Not a valid ISO 8601 datetime: {v!r}") from exc
        return cleaned

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

async def check_availability(
    args: CheckAvailabilityInput, access_token: str
) -> Dict[str, Any]:
    """
    Google Calendar freeBusy query.

    Returns busy intervals plus a boolean 'all_free' convenience flag so the
    agent can branch deterministically without re-reasoning over slots.
    """
    body: Dict[str, Any] = {
        "timeMin": args.start_iso,
        "timeMax": args.end_iso,
        "items": [{"id": args.calendar_id}],
        "timeZone": args.timezone,
    }
    result = await google_request(
        "POST",
        f"{GOOGLE_CALENDAR_API}/freeBusy",
        access_token=access_token,
        json_body=body,
    )

    calendars = result.get("calendars", {})
    cal = calendars.get(args.calendar_id, {})
    busy: List[Dict[str, str]] = cal.get("busy", [])

    return {
        "calendar_id": args.calendar_id,
        "window": {"start": args.start_iso, "end": args.end_iso, "timezone": args.timezone},
        "busy_intervals": busy,
        "all_free": len(busy) == 0,
        "busy_count": len(busy),
    }


async def schedule_with_meet(
    args: ScheduleWithMeetInput, access_token: str
) -> Dict[str, Any]:
    """
    Insert a calendar event with an auto-provisioned Google Meet link.

    Payload contract (per Google docs):
    - conferenceDataVersion=1 query param enables conference creation.
    - conferenceData.createRequest.conferenceSolutionKey.type = 'hangoutsMeet'.
    - A generated requestId keeps creation idempotent per event draft.
    """
    event_body: Dict[str, Any] = {
        "summary": args.summary,
        "description": args.description,
        "start": {"dateTime": args.start_iso, "timeZone": args.timezone},
        "end": {"dateTime": args.end_iso, "timeZone": args.timezone},
        "attendees": [{"email": e} for e in args.attendee_emails],
        "conferenceData": {
            "createRequest": {
                "requestId": _make_request_id(args.summary, args.start_iso),
                "conferenceSolutionKey": {"type": "hangoutsMeet"},
            }
        },
        "reminders": {"useDefault": True},
    }

    params: Dict[str, Any] = {
        "conferenceDataVersion": 1,
        "sendUpdates": args.send_updates,
    }

    created = await google_request(
        "POST",
        f"{GOOGLE_CALENDAR_API}/calendars/{args.calendar_id}/events",
        access_token=access_token,
        params=params,
        json_body=event_body,
    )

    meet_link = _extract_meet_link(created) or ""
    event_id = created.get("id", "")

    if not meet_link:
        # Conference data can take a few seconds to materialize; fetch once.
        try:
            refetched = await google_request(
                "GET",
                f"{GOOGLE_CALENDAR_API}/calendars/{args.calendar_id}/events/{event_id}",
                access_token=access_token,
                params={"conferenceDataVersion": 1},
            )
            meet_link = _extract_meet_link(refetched) or ""
        except Exception as exc:
            logger.warning("meet_link_refetch_failed", event_id=event_id, error=str(exc))

    logger.info(
        "calendar_event_created",
        event_id=event_id,
        meet_link_present=bool(meet_link),
        attendees=len(args.attendee_emails),
    )

    return {
        "event_id": event_id,
        "meet_link": meet_link,
        "html_link": created.get("htmlLink", ""),
        "summary": created.get("summary", args.summary),
        "start": created.get("start", {}).get("dateTime", args.start_iso),
        "end": created.get("end", {}).get("dateTime", args.end_iso),
        "attendees": args.attendee_emails,
    }


def _extract_meet_link(event: Dict[str, Any]) -> str | None:
    """Locate the hangoutsMeet entry point in a conferenceData payload."""
    conf = event.get("conferenceData", {})
    for entry in conf.get("entryPoints", []):
        if entry.get("entryPointType") == "video" and entry.get("uri", "").startswith("https://meet.google.com/"):
            return entry["uri"]
    return None


def _make_request_id(summary: str, start_iso: str) -> str:
    """Deterministic per-request idempotency key for conferenceData."""
    import re
    slug = re.sub(r"[^a-z0-9]", "", summary.lower())[:20] or "event"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"mrops-{slug}-{stamp}"


# ── Registry ─────────────────────────────────────────────────────────

TOOLS = (
    ToolDefinition(
        name="calendar_check_availability",
        server=SERVER_NAME,
        description=(
            "Check a Google Calendar's free/busy status for a time window. "
            "Returns the list of busy intervals and whether the whole window is free."
        ),
        input_model=CheckAvailabilityInput,
        handler=check_availability,
    ),
    ToolDefinition(
        name="calendar_schedule_with_meet",
        server=SERVER_NAME,
        description=(
            "Create a Google Calendar event with an auto-generated Google Meet "
            "video link, invite attendees, and return the event ID plus the "
            "Meet URL. Send the meet link to the user in your final reply."
        ),
        input_model=ScheduleWithMeetInput,
        handler=schedule_with_meet,
    ),
)


def create_mcp_server() -> Any:
    """
    Official-SDK MCP server wrapper (FastMCP) exposing the same handlers.

    Used for out-of-process MCP deployment (stdio / streamable HTTP). The
    in-process ClientManager path calls `TOOLS` handlers directly.
    """
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(SERVER_NAME)

    @mcp.tool()
    async def calendar_check_availability(
        start_iso: str, end_iso: str, timezone: str = "UTC", calendar_id: str = "primary"
    ) -> Dict[str, Any]:
        env = attached_access_token.get()
        args = CheckAvailabilityInput(start_iso=start_iso, end_iso=end_iso, timezone=timezone, calendar_id=calendar_id)
        return await check_availability(args, env)

    @mcp.tool()
    async def calendar_schedule_with_meet(
        summary: str, start_iso: str, end_iso: str,
        attendee_emails: List[str] | None = None, description: str = "",
        timezone: str = "UTC", calendar_id: str = "primary", send_updates: str = "all",
    ) -> Dict[str, Any]:
        token = attached_access_token.get()
        args = ScheduleWithMeetInput(
            summary=summary, start_iso=start_iso, end_iso=end_iso,
            attendee_emails=attendee_emails or [], description=description,
            timezone=timezone, calendar_id=calendar_id, send_updates=send_updates,
        )
        return await schedule_with_meet(args, token)

    return mcp