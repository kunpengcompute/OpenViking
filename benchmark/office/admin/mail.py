"""Import synthetic mail fixtures and obtain browser login state for office cases."""

import imaplib
import json
import os
from datetime import datetime
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from email.utils import format_datetime, formataddr
from html.parser import HTMLParser

import httpx


def message_bytes(row, recipient, fixture_id="mail-pilot-v2"):
    message = EmailMessage()
    message["From"] = formataddr((row["from_name"], row["from"]), charset="utf-8")
    message["To"] = recipient
    message["Subject"] = row["subject"]
    message["Date"] = format_datetime(datetime.fromisoformat(row["date"]))
    message["Message-ID"] = f"<{fixture_id}-{row['id']}@example.test>"
    message.set_content(row["body"], charset="utf-8")
    for attachment in row.get("attachments", []):
        message.add_attachment(
            attachment["content"], subtype="plain", filename=attachment["filename"], charset="utf-8"
        )
    return message.as_bytes(policy=policy.SMTP)


def seed(settings, snapshot, fixture):
    """Append only missing fixtures. Reject unrelated mail and any future mail in history."""
    fixture_id = fixture["fixture_id"]
    desired = {f"<{fixture_id}-{mid}@example.test>" for mid in fixture["snapshots"][snapshot]}
    with imaplib.IMAP4(settings["imap_host"], settings["imap_port"]) as conn:
        conn.login(settings["username"], settings["password"])
        folder = settings.get("mailbox", "INBOX")
        status, _ = conn.select(folder)
        if status != "OK":
            status, _ = conn.create(folder)
            assert status == "OK", "Cannot create isolated fixture folder"
            conn.subscribe(folder)
            status, _ = conn.select(folder)
        assert status == "OK"
        status, numbers = conn.search(None, "ALL")
        assert status == "OK"
        existing = {}
        for number in numbers[0].split():
            status, raw = conn.fetch(number, "(BODY.PEEK[])")
            assert status == "OK"
            message = BytesParser(policy=policy.default).parsebytes(raw[0][1])
            # RFC header folding can leave leading whitespace on a long ID.
            existing[str(message["Message-ID"]).strip()] = message
        if not set(existing) <= desired:
            raise ValueError("Mailbox has mail outside the requested snapshot; use a fresh fixture")
        rows = {
            f"<{fixture_id}-{row['id']}@example.test>": row
            for row in fixture["messages"]
            if row["id"] in fixture["snapshots"][snapshot]
        }
        for mid, row in rows.items():
            if mid not in existing:
                stamp = datetime.fromisoformat(row["date"]).timestamp()
                status, _ = conn.append(
                    folder,
                    None,
                    imaplib.Time2Internaldate(stamp),
                    message_bytes(row, fixture["recipient"], fixture_id),
                )
                assert status == "OK"
            else:
                actual = existing[mid]
                body = (
                    actual.get_body(preferencelist=("plain",)) if actual.is_multipart() else actual
                )
                attachments = [
                    {
                        "filename": part.get_filename(),
                        "content": part.get_content().replace("\r\n", "\n").rstrip("\n"),
                    }
                    for part in actual.iter_attachments()
                ]
                expected = [
                    {
                        "filename": part["filename"],
                        "content": part["content"].replace("\r\n", "\n").rstrip("\n"),
                    }
                    for part in row.get("attachments", [])
                ]
                if (
                    str(actual["Subject"]) != row["subject"]
                    or body is None
                    or body.get_content().strip() != row["body"]
                    or attachments != expected
                ):
                    raise ValueError("Existing fixture differs from source")
        status, numbers = conn.search(None, "ALL")
        assert status == "OK" and len(numbers[0].split()) == len(desired)
        # Each phase starts unread, retaining the original message dates.
        if numbers[0]:
            conn.store(b",".join(numbers[0].split()), "-FLAGS", r"(\Seen)")
    return {
        "snapshot": snapshot,
        "verified": True,
        "message_count": len(desired),
        "message_ids": sorted(desired),
    }


class Inputs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.values = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "input" and attrs.get("name"):
            self.values[attrs["name"]] = attrs.get("value", "")


def login_state(settings, output):
    """Create a private browser cookie state without passing login credentials to the model."""
    with httpx.Client(follow_redirects=True, trust_env=False, timeout=30) as client:
        response = client.get(settings["webmail_url"])
        response.raise_for_status()
        form = Inputs()
        form.feed(response.text)
        if "_token" not in form.values:
            raise ValueError("Roundcube login form not found")
        payload = {
            **form.values,
            "_task": "login",
            "_action": "login",
            "_user": settings["username"],
            "_pass": settings["password"],
        }
        response = client.post(settings["webmail_url"], data=payload)
        response.raise_for_status()
        if 'id="messagelist"' not in response.text:
            raise ValueError("Roundcube login did not reach the inbox")
        cookies = [
            {
                "name": c.name,
                "value": c.value,
                "domain": c.domain,
                "path": c.path,
                "expires": c.expires or -1,
                "httpOnly": "HttpOnly" in c._rest,
                "secure": c.secure,
                "sameSite": "Lax",
            }
            for c in client.cookies.jar
        ]
        fd = os.open(output, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump({"cookies": cookies, "origins": []}, handle)
    return {"authenticated": True, "state_file": str(output)}
