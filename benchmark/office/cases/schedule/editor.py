"""Accessible date/time fields and scoped calendar return locations."""

from html import escape
from urllib.parse import parse_qs, urlencode, urlsplit


def datetime_field(key, label, value, kind):
    field_id = key.replace("_", "-")
    return (
        f'<div class="datetime-field" data-key="{key}" data-kind="{kind}">'
        f'<label for="{field_id}">{label}</label>'
        f'<input id="{field_id}" name="{key}" type="text" value="{escape(value)}" '
        f'required autocomplete="off" inputmode="numeric" '
        f'placeholder="{"YYYY-MM-DD" if kind == "date" else "HH:mm"}" '
        f"data-datetime-input></div>"
    )


def calendar_return_url(base, value):
    """Keep view filters within this workspace; discard external return targets."""
    target = urlsplit(value)
    if target.scheme or target.netloc or target.path != base:
        return base
    values = parse_qs(target.query, keep_blank_values=True)
    allowed = {
        k: v
        for k, v in values.items()
        if k in {"month", "view", "q", "calendar_filter", "calendar", "selected"}
    }
    return base + ("?" + urlencode(allowed, doseq=True) if allowed else "")


def with_query(url, **updates):
    target = urlsplit(url)
    values = parse_qs(target.query, keep_blank_values=True)
    values.update(updates)
    return target.path + "?" + urlencode(values, doseq=True)
