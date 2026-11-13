"""Single source of H/Q task text for manual use and Chat runners."""

from datetime import datetime
from pathlib import Path
from string import Template
from urllib.parse import urlencode, urlsplit

CASES = ("mail", "timesheet", "schedule")
PHASES = ("H", "Q")
TASK_ROOT = Path(__file__).resolve().parent
DEFAULT_ANCHORS = {
    "mail": {"H": "2026-11-06T10:20:00+08:00", "Q": "2026-11-06T10:20:00+08:00"},
    "timesheet": {"H": "2026-11-03T10:00:00+08:00", "Q": "2026-11-04T10:00:00+08:00"},
    "schedule": {"H": "2026-10-31T09:00:00+08:00", "Q": "2026-10-31T09:00:00+08:00"},
}


def task_path(case: str, phase: str) -> Path:
    if case not in CASES or phase not in PHASES:
        raise ValueError("Choose case mail/timesheet/schedule and phase H/Q")
    return TASK_ROOT / case / f"{phase}.md"


def render_task(
    case: str,
    phase: str,
    *,
    workspace: str,
    page_url: str = "http://127.0.0.1:19482",
    anchor: str | None = None,
) -> str:
    """Fill environment fields; business goals and guidance stay in Markdown."""
    path = task_path(case, phase)
    parts = urlsplit(page_url)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.netloc
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
    ):
        raise ValueError("page_url must be an HTTP(S) base URL without credentials or query")
    if not workspace:
        raise ValueError("workspace must not be empty")
    instant = datetime.fromisoformat(anchor or DEFAULT_ANCHORS[case][phase])
    if instant.utcoffset() is None:
        raise ValueError("anchor must include a timezone offset")
    entry = page_url.rstrip("/") + "/mock/login?" + urlencode({"workspace": workspace})
    return (
        Template(path.read_text(encoding="utf-8"))
        .substitute(entry=entry, anchor=instant.isoformat(timespec="seconds"))
        .strip()
    )
