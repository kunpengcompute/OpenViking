"""Synthetic employee selection with workspace cookies and native-app SSO bridge."""

import hashlib
import json
import secrets
import tempfile
import time
from http.cookies import SimpleCookie
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .store import connect


class OfficeAuth:
    def __init__(self, registry):
        self.registry = Path(registry)

    def cookie_name(self, ns, purpose="session"):
        return "office_" + purpose + "_" + hashlib.sha256(ns.encode()).hexdigest()[:16]

    def token(self, h, ns, purpose="session"):
        try:
            jar = SimpleCookie()
            jar.load(h.headers.get("Cookie", ""))
            return jar[self.cookie_name(ns, purpose)].value
        except (KeyError, ValueError):
            return ""

    def session(self, h, ns):
        token = self.token(h, ns)
        if not token:
            return None
        db = connect(self.registry, ns)
        try:
            row = db.execute(
                "SELECT * FROM sessions WHERE token_hash=? AND created>?",
                (hashlib.sha256(token.encode()).hexdigest(), time.time() - 3600),
            ).fetchone()
            return dict(row) if row else None
        finally:
            db.close()

    def challenge(self, ns):
        token = secrets.token_urlsafe(32)
        csrf = secrets.token_urlsafe(24)
        db = connect(self.registry, ns)
        try:
            with db:
                db.execute("DELETE FROM challenges WHERE created<?", (time.time() - 600,))
                db.execute(
                    "INSERT INTO challenges VALUES(?,?,?)",
                    (hashlib.sha256(token.encode()).hexdigest(), csrf, time.time()),
                )
        finally:
            db.close()
        return token, csrf

    def consume_challenge(self, h, ns, csrf):
        token = self.token(h, ns, "login")
        if not token:
            return False
        db = connect(self.registry, ns)
        try:
            with db:
                result = db.execute(
                    "DELETE FROM challenges WHERE token_hash=? AND csrf=? AND created>?",
                    (hashlib.sha256(token.encode()).hexdigest(), csrf, time.time() - 600),
                )
            return result.rowcount == 1
        finally:
            db.close()

    def native_cookies(self, ns):
        # Credentials remain in a host-only 0600 file; browser receives session cookies only.
        private = self.registry / "office-private" / (ns + ".local.json")
        if not private.exists():
            return []
        if private.stat().st_mode & 0o077:
            raise ValueError("Private login configuration must be mode 0600")
        data = json.loads(private.read_text())
        from benchmark.office.admin.mail import login_state

        state = {"cookies": []}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.local.json"
            login_state(data["mail"], path)
            state["cookies"].extend(json.loads(path.read_text())["cookies"])
        return state["cookies"]

    def issue(self, ns, cookies):
        token = secrets.token_urlsafe(32)
        csrf = secrets.token_urlsafe(24)
        names = sorted({c["name"] for c in cookies})
        db = connect(self.registry, ns)
        try:
            with db:
                db.execute("DELETE FROM sessions WHERE created<?", (time.time() - 3600,))
                db.execute(
                    "INSERT INTO sessions VALUES(?,?,?,?)",
                    (
                        hashlib.sha256(token.encode()).hexdigest(),
                        csrf,
                        time.time(),
                        json.dumps(names),
                    ),
                )
        finally:
            db.close()
        return token

    def cookie(self, name, value, max_age=3600):
        jar = SimpleCookie()
        jar[name] = value
        jar[name]["path"] = "/"
        jar[name]["httponly"] = True
        jar[name]["samesite"] = "Lax"
        jar[name]["max-age"] = max_age
        return jar[name].OutputString()

    def form(self, h):
        origin = h.headers.get("Origin")
        if origin and urlsplit(origin).netloc != h.headers.get("Host"):
            raise ValueError("Cross-origin form")
        size = int(h.headers.get("Content-Length", "0"))
        if not 0 < size <= 12000:
            raise ValueError("Invalid form size")
        return {
            k: v[0]
            for k, v in parse_qs(h.rfile.read(size).decode(), keep_blank_values=True).items()
        }

    def logout(self, h, ns, current, csrf):
        if not secrets.compare_digest(current["csrf"], csrf):
            return None
        db = connect(self.registry, ns)
        try:
            with db:
                db.execute("DELETE FROM sessions WHERE token_hash=?", (current["token_hash"],))
        finally:
            db.close()
        return json.loads(current["cookie_names"])
