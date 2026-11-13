"""Loopback-only synthetic workbench and scoped HTML enhancement proxy.

No credentials, scoring references, expected outcomes or application database access.
The upstream Redmine owns issue persistence. The optional synthetic calendar
adapter owns its isolated calendar persistence; neither exposes scoring references.
"""

import argparse
import html
import http.client
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

NAMESPACE = re.compile(r"[a-z0-9][a-z0-9-]{0,79}")
UPSTREAM = "http://127.0.0.1:19481"
PAGE_APP = "http://127.0.0.1:19482"
HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "content-length",
}


def load_registry(folder, namespace):
    if not NAMESPACE.fullmatch(namespace):
        raise ValueError("Invalid namespace")
    data = json.loads((folder / (namespace + ".json")).read_text())
    if data["namespace"] != namespace or (
        data.get("case") not in ("schedule", "office") and len(data["projects"]) != 2
    ):
        raise ValueError("Invalid page registry")
    for project in data["projects"]:
        if not re.fullmatch(
            re.escape(namespace) + r"-(northstar-platform|northstar-data)", project["identifier"]
        ):
            raise ValueError("Unscoped project")
    return data


def workbench(data):
    from .portal.catalog import webmail_url

    cards = []
    if data["case"] == "cross":
        cards.append(("办公邮箱", "申请与确认邮件", webmail_url()))
    if data["case"] == "schedule":
        cards.append(("日程管理", "个人日历、部门日历与日程安排", "/calendar/" + data["namespace"]))
        for title, key in [
            ("办公邮箱", "mail"),
            ("知识库", "knowledge"),
            ("通讯录", "contacts"),
            ("公告通知", "news"),
            ("工单中心", "tickets"),
            ("文件空间", "files"),
            ("审批中心", "approvals"),
        ]:
            cards.append((title, "办公应用", "/office-info/" + data["namespace"] + "/" + key))
    for project in data["projects"]:
        path = "/projects/" + project["identifier"]
        cards.append((project["name"], "工单与项目资料", path))
    links = "".join(
        f"<article><h2>{html.escape(name)}</h2><p>{html.escape(description)}</p>"
        f'<a target="_blank" rel="noopener" href="{html.escape(url, quote=True)}">进入{name}</a></article>'
        for name, description, url in cards
    )
    return (
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>办公工作台</title>'
        "<style>body{font:16px sans-serif;margin:40px;background:#f3f6fb}nav{padding:18px;background:#183b61;color:white}"
        "section{display:flex;gap:20px;flex-wrap:wrap}article{padding:24px;background:white;border:1px solid #ddd;min-width:220px}"
        "input{margin:24px 0;padding:10px;width:340px}</style><nav>办公工作台 · 应用中心</nav>"
        '<h1>我的应用</h1><label for="app-search">搜索应用</label> '
        '<input id="app-search" type="search" placeholder="应用名称">'
        "<section>" + links + '</section><p id="empty" hidden>没有匹配应用</p>'
        '<script>document.querySelector("input").oninput=e=>{let visible=0;'
        'document.querySelectorAll("article").forEach(x=>{x.hidden=!x.textContent.includes(e.target.value.trim());'
        'if(!x.hidden)visible++});document.querySelector("#empty").hidden=!!visible}</script></html>'
    ).encode()


def registered_html(body, folder):
    # Only exact project identifiers from this layer's registry opt into enhancement.
    text = body.decode("utf-8", errors="replace")
    identifiers = set(re.findall(r'/projects/([a-z0-9-]+)(?=[/?"\'#])', text))
    for identifier in identifiers:
        for suffix in ["-northstar-platform", "-northstar-data"]:
            if not identifier.endswith(suffix):
                continue
            try:
                data = load_registry(folder, identifier[: -len(suffix)])
            except (OSError, ValueError, KeyError):
                continue
            if identifier in {p["identifier"] for p in data["projects"]}:
                return True
    return False


class PageHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # Never log cookies, credentials, URLs or form bodies.

    def respond(self, status, content, mime="text/html; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        route = urlsplit(self.path).path
        portal = getattr(self.server, "portal", None)
        if portal and portal.handle(self, "GET", route):
            return
        if route == "/office-ui.js":
            return self.respond(
                200,
                Path(__file__).with_name("page_ui.js").read_bytes(),
                "application/javascript; charset=utf-8",
            )
        if route == "/healthz":
            return self.respond(200, b'{"version":"office-pages-v2"}', "application/json")
        if route.startswith("/workbench/"):
            try:
                data = load_registry(self.server.registry, route.removeprefix("/workbench/"))
            except (ValueError, OSError, KeyError):
                return self.respond(404, b"Workbench not found")
            if data.get("case") == "office":
                return self.server.portal.workbench(self, data)
            if data.get("case") == "schedule":
                return self.server.calendar.bootstrap(self, data["namespace"], workbench(data))
            return self.respond(200, workbench(data))
        if route.startswith("/office-info/"):
            pieces = route.strip("/").split("/")
            try:
                data = load_registry(self.server.registry, pieces[1])
                if data.get("case") != "schedule":
                    raise ValueError("Unknown office")
            except (OSError, ValueError, KeyError, IndexError):
                return self.respond(404, b"Not found")
            from .schedule.app import layout

            return self.respond(
                200,
                layout(
                    "办公应用",
                    '<h1>办公应用</h1><p>本应用没有本轮待办。</p><a href="/workbench/'
                    + data["namespace"]
                    + '">返回工作台</a>',
                ),
            )
        if self.server.calendar.handle(self, "GET", route):
            return
        form = re.fullmatch(
            r"/projects/([a-z0-9-]+)-(northstar-platform|northstar-data)/issues/new", route
        )
        if form:
            try:
                data = load_registry(self.server.registry, form[1])
                project = next(
                    p for p in data["projects"] if p["identifier"] == form[1] + "-" + form[2]
                )
            except (OSError, ValueError, KeyError, StopIteration):
                pass
            else:
                # The native global new form exposes project switching; project-scoped
                # sibling projects do not. Redmine still owns the form and AJAX reload.
                self.send_response(302)
                self.send_header(
                    "Location", "/issues/new?issue%5Bproject_id%5D=" + str(int(project["id"]))
                )
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
        self.proxy()

    def do_POST(self):
        portal = getattr(self.server, "portal", None)
        if portal and portal.handle(self, "POST", urlsplit(self.path).path):
            return
        if self.server.calendar.handle(self, "POST", urlsplit(self.path).path):
            return
        self.proxy()

    def proxy(self):
        if not self.path.startswith("/") or self.path.startswith("//"):
            return self.respond(400, b"Invalid path")
        headers = {
            k: v
            for k, v in self.headers.items()
            if k.lower() not in HOP_HEADERS | {"host", "accept-encoding"}
        }
        headers["Host"] = urlsplit(UPSTREAM).netloc
        headers["Accept-Encoding"] = "identity"
        if headers.get("Origin") == PAGE_APP:
            headers["Origin"] = UPSTREAM
        if headers.get("Referer", "").startswith(PAGE_APP + "/"):
            headers["Referer"] = UPSTREAM + headers["Referer"][len(PAGE_APP) :]
        length = int(self.headers.get("Content-Length", "0"))
        if length > 2_000_000:
            return self.respond(413, b"Form too large")
        body = self.rfile.read(length) if length else None
        connection = http.client.HTTPConnection("127.0.0.1", 19481, timeout=30)
        try:
            connection.request(self.command, self.path, body=body, headers=headers)
            response = connection.getresponse()
            payload = response.read()
            mime = response.getheader("Content-Type", "")
            enhanced = "text/html" in mime and registered_html(payload, self.server.registry)
            if enhanced:
                payload = payload.replace(
                    b"</body>", b'<script src="/office-ui.js"></script></body>'
                )
            self.send_response(response.status)
            for key, value in response.getheaders():
                if key.lower() in HOP_HEADERS:
                    continue
                if enhanced and key.lower() in {"etag", "content-md5"}:
                    continue
                if key.lower() == "location":
                    value = value.replace(UPSTREAM, PAGE_APP)
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (OSError, http.client.HTTPException):
            self.respond(502, b"Upstream unavailable")
        finally:
            connection.close()


def main():
    global PAGE_APP

    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--port", type=int, default=19482)
    args = parser.parse_args()
    args.registry.mkdir(parents=True, exist_ok=True)
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535")
    PAGE_APP = f"http://127.0.0.1:{args.port}"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), PageHandler)
    server.registry = args.registry
    from .schedule.app import CalendarApp

    server.calendar = CalendarApp(args.registry)
    from .portal.app import OfficePortal

    server.portal = OfficePortal(args.registry)
    server.serve_forever()


if __name__ == "__main__":
    main()
