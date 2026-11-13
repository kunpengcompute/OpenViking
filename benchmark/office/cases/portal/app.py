"""Shared login/workbench for three synthetic office cases and benign distractors."""

import html
import json
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

from ..schedule.data import NAMESPACE
from .auth import OfficeAuth
from .catalog import DISTRACTORS, applications, webmail_url


def esc(value):
    return html.escape(str(value), quote=True)


def layout(title, body):
    return (
        """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>"""
        + esc(title)
        + """</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f6f9;color:#263247;font:15px/1.65 system-ui,"Microsoft YaHei",sans-serif}header{background:#fff;border-bottom:1px solid #dce2eb;padding:15px 30px;display:flex;align-items:center;justify-content:space-between;gap:20px}header strong{color:#a92737;font-size:20px}header span{color:#68758a;font-size:13px}main{max-width:1200px;margin:25px auto;padding:0 22px}h1{font-size:26px}h2{font-size:19px}h3{font-size:16px;margin:0}a{color:#245d9a}button,input,textarea,select{font:inherit}button{cursor:pointer;background:white;color:#34465e;border:1px solid #c5cfdc;border-radius:6px;padding:8px 13px}button.primary{background:#a92737;color:white;border-color:#a92737}button:disabled{opacity:.45;cursor:default}input,textarea,select{border:1px solid #bdc8d7;border-radius:5px;padding:10px;background:#fff}a:focus-visible,button:focus-visible,input:focus-visible,summary:focus-visible{outline:3px solid #8cb2dc;outline-offset:2px}.muted{color:#68758a}.hint{font-size:13px;color:#68758a}.pill{background:#fce9ec;color:#a92737;border-radius:5px;padding:3px 8px;font-size:12px}nav{display:flex;align-items:center;gap:24px}nav a{text-decoration:none}section{margin-top:25px}article,.panel{background:#fff;border:1px solid #dce2eb;border-radius:8px;padding:20px}.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.cards article p{font-size:13px;min-height:43px;color:#68758a}.actions{display:flex;gap:12px;align-items:center;margin-top:15px}.enter{border-radius:5px;background:#a92737;color:#fff;padding:6px 11px;text-decoration:none;font-size:13px}.search{width:480px;max-width:100%}.login{max-width:470px;margin:65px auto}.login label{display:block;margin:20px 0 7px}.login input[type=text]{width:100%}.network{border:0;padding:0;margin:15px 0}.network label{display:inline-flex;align-items:center;gap:5px;margin:5px 10px 5px 0}.network input{padding:0}#app-catalog h3{margin:12px 0}nav{flex-wrap:wrap}.tabs{display:flex;gap:20px;padding:12px 0;border-bottom:1px solid #dce2eb}.tabs [aria-selected=true]{color:#a92737;border-bottom:2px solid #a92737}.login button.primary{width:100%;margin-top:22px}.candidates{border:1px solid #c5cfdc;background:white;border-radius:5px;margin-top:5px}.candidates button{border:0;border-radius:0;width:100%;text-align:left;padding:12px}.candidates button:hover{background:#f2f5fa}.error{color:#ac2333}dialog{border:1px solid #bdc8d7;border-radius:9px;max-width:550px}table{border-collapse:collapse;width:100%;background:white}th,td{border-bottom:1px solid #dce2eb;text-align:left;padding:11px}.formgrid{display:grid;grid-template-columns:1fr 1fr;gap:15px}.formgrid label{display:block;margin:10px 0 5px}textarea{width:100%;min-height:75px}[hidden]{display:none!important}@media(max-width:800px){.cards{grid-template-columns:1fr 1fr}header{padding:13px 20px;flex-wrap:wrap}}@media(max-width:520px){.cards,.formgrid{grid-template-columns:1fr}.login{margin:25px auto}}
</style></head><body><header><strong>企业办公测试平台</strong><span>合成测试环境 · 模拟账号登录</span></header><main>"""
        + body
        + '<script src="/office-assets/ui.js"></script></main></body></html>'
    ).encode()


class OfficePortal:
    def __init__(self, registry):
        self.registry = Path(registry)
        self.auth = OfficeAuth(registry)

    def config(self, ns):
        if not NAMESPACE.fullmatch(ns):
            raise ValueError("Invalid workspace")
        data = json.loads((self.registry / (ns + ".json")).read_text())
        if data.get("namespace") != ns or data.get("case") != "office":
            raise ValueError("Unknown workspace")
        return data

    def redirect(self, h, target, cookies=()):
        h.send_response(303)
        h.send_header("Location", target)
        for value in cookies:
            h.send_header("Set-Cookie", value)
        h.send_header("Cache-Control", "no-store")
        h.send_header("Content-Length", "0")
        h.end_headers()

    def login_path(self, ns):
        return "/mock/login?" + urlencode({"workspace": ns})

    def require(self, h, ns, method):
        current = self.auth.session(h, ns)
        if current:
            return current
        if method == "GET":
            self.redirect(h, self.login_path(ns))
        else:
            h.respond(403, b"Office login required")
        return None

    def workbench(self, h, data):
        ns = data["namespace"]
        current = self.require(h, ns, "GET")
        if not current:
            return
        employee = data["employee"]
        apps = applications(ns)

        def card(app, subscribed=False):
            name = esc(app["name"])
            desc = esc(app["description"])
            opening = (
                (
                    '<a class="enter" target="_blank" rel="noopener" href="'
                    + esc(app["url"])
                    + '">进入应用</a>'
                )
                if subscribed
                else '<span class="hint">'
                + ("已订阅" if app["subscribed"] else "未订阅 · 可申请订阅")
                + "</span>"
            )
            return (
                '<article data-keywords="'
                + esc(app["keywords"])
                + '"><h3>'
                + name
                + "</h3><p>"
                + desc
                + "</p>"
                '<p class="hint">'
                + esc(app["category"] + " · " + app["publisher"])
                + '</p><div class="actions">'
                '<button type="button" data-detail="'
                + name
                + '" data-description="'
                + desc
                + '">应用详情</button>'
                + opening
                + "</div></article>"
            )

        subscribed = "".join(card(app, True) for app in apps if app["subscribed"])
        catalog = ""
        for category in dict.fromkeys(app["category"] for app in apps):
            catalog += (
                '<section aria-label="'
                + esc(category)
                + '"><h3>'
                + esc(category)
                + '</h3><div class="cards">'
            )
            catalog += (
                "".join(card(app) for app in apps if app["category"] == category)
                + "</div></section>"
            )
        body = (
            '<nav><a target="_blank" rel="noopener" href="'
            + esc(webmail_url())
            + '">邮件</a><a href="/workbench/'
            + ns
            + '" aria-current="page">工作台</a><a href="#subscribed">我的应用</a>'
            '<a href="#app-catalog">应用市场</a><button type="button" data-info="manage">管理</button>'
            "<span>"
            + esc(employee["name"])
            + " · "
            + esc(employee["department"])
            + " · "
            + esc(employee["number"])
            + "</span>"
            '<form method="post" action="/mock/logout?workspace='
            + ns
            + '"><input type="hidden" name="csrf" value="'
            + esc(current["csrf"])
            + '"><button>退出登录</button></form></nav>'
            '<h1>办公工作台</h1><p class="muted">从已订阅的应用进入办理事项，可按应用名称或拼音搜索。</p>'
            '<label for="app-search">应用搜索</label><p><input class="search" id="app-search" type="search" placeholder="请输入应用名称或拼音查找"></p>'
            '<section aria-label="已订阅"><h2>已订阅 <span class="pill">'
            + str(sum(a["subscribed"] for a in apps))
            + ' 个应用</span></h2><div id="subscribed" class="cards">'
            + subscribed
            + "</div>"
            '<p id="app-empty" role="status" hidden>没有匹配的已订阅应用。</p></section>'
            '<section id="app-catalog" aria-label="应用市场"><h2>应用市场 <span class="pill">'
            + str(len(apps))
            + " 个应用</span></h2>"
            '<p class="hint">按分类浏览应用目录，已订阅应用可从上方进入。</p>'
            + catalog
            + "</section>"
            '<dialog id="app-detail"><h2 id="app-detail-title"></h2><p id="app-detail-body"></p><p class="hint">应用详情展示功能介绍。通过工作台“进入应用”打开业务页面。</p>'
            '<button type="button" data-close>关闭</button></dialog>'
            '<dialog id="info-dialog"><h2 id="info-title"></h2><p id="info-body"></p><button type="button" data-close>关闭</button></dialog>'
        )
        # One common portal authorizes the legacy calendar without exposing its cookie in scripts.
        h.server.calendar.bootstrap(h, ns, layout("办公工作台", body))

    def handle(self, h, method, path):
        if path == "/office-assets/ui.js" and method == "GET":
            h.respond(
                200,
                Path(__file__).with_name("ui.js").read_bytes(),
                "application/javascript; charset=utf-8",
            )
            return True
        if path in ("/", "/mock/login", "/mock/logout"):
            params = parse_qs(urlsplit(h.path).query)
            ns = params.get("workspace", [""])[0]
            if not ns:
                default = self.registry / "office-default.txt"
                if default.exists():
                    ns = default.read_text().strip()
            try:
                data = self.config(ns)
            except (ValueError, OSError, KeyError):
                h.respond(404, b"Office workspace not found")
                return True
            if path == "/":
                self.redirect(h, self.login_path(ns))
                return True
            if path == "/mock/logout":
                if method != "POST":
                    h.respond(405, b"Use POST to logout")
                    return True
                current = self.auth.session(h, ns)
                try:
                    fields = self.auth.form(h)
                except ValueError:
                    h.respond(403, b"Invalid logout form")
                    return True
                names = (
                    self.auth.logout(h, ns, current, fields.get("csrf", "")) if current else None
                )
                if names is None:
                    h.respond(403, b"Invalid logout")
                    return True
                names += [self.auth.cookie_name(ns), h.server.calendar.cookie_name(ns)]
                self.redirect(
                    h, self.login_path(ns), [self.auth.cookie(name, "", 0) for name in names]
                )
                return True
            if method == "GET":
                token, csrf = self.auth.challenge(ns)
                employee = data["employee"]
                body = (
                    '<section class="panel login"><span class="pill">模拟测试登录</span><h1>统一认证 测试登录</h1>'
                    '<p class="hint">使用本测试工作区的合成工号选择账号。</p><form method="post" action="'
                    + self.login_path(ns)
                    + '">'
                    '<fieldset class="network"><legend>模拟网络环境：</legend><label><input type="radio" name="network" value="office" checked>办公网</label><label><input type="radio" name="network" value="internet">互联网</label><label><input type="radio" name="network" value="business">业务网</label></fieldset>'
                    '<input type="hidden" name="csrf" value="'
                    + esc(csrf)
                    + '"><input type="hidden" id="employee" name="employee" value="">'
                    '<label for="employee-input">工号 / 姓名</label><input id="employee-input" name="employee_input" type="text" autocomplete="off" role="combobox" '
                    'aria-autocomplete="list" aria-expanded="false" aria-controls="employee-options" placeholder="请输入手机号、姓名、sapID、一事通ID搜索账号"><button type="button" id="employee-choose" aria-controls="employee-options">选择</button>'
                    '<div id="employee-options" class="candidates" role="listbox" aria-label="测试账号候选" hidden><button type="button" role="option" data-number="'
                    + esc(employee["number"])
                    + '">'
                    + esc(
                        employee["number"]
                        + " · "
                        + employee["name"]
                        + " · "
                        + employee["department"]
                    )
                    + "</button></div>"
                    '<p id="employee-empty" class="error" hidden>没有匹配的测试账号。</p><p class="hint">输入后选择下拉候选，按 Enter 可确认选择。</p>'
                    '<button id="login-submit" class="primary" disabled>登录</button></form>'
                    '<p class="hint">支持测试账号模拟认证。正式认证方式由账号所在网络环境决定。</p>'
                    '<p>未找到合适的账号？ <button type="button" data-info="create">点击创建</button></p>'
                    '<p><button type="button" data-info="help" aria-label="登录帮助">帮助</button> <a href="'
                    + self.login_path(ns)
                    + '">返回登录页</a></p>'
                    '<dialog id="info-dialog"><h2 id="info-title"></h2><p id="info-body"></p><button type="button" data-close>关闭</button></dialog>'
                    '<p class="hint">此页面仅用于合成案例测试。</p></section>'
                )
                content = layout("模拟测试登录", body)
                h.send_response(200)
                h.send_header("Content-Type", "text/html; charset=utf-8")
                h.send_header(
                    "Set-Cookie", self.auth.cookie(self.auth.cookie_name(ns, "login"), token, 600)
                )
                h.send_header("Cache-Control", "no-store")
                h.send_header("Content-Length", str(len(content)))
                h.end_headers()
                h.wfile.write(content)
                return True
            if method != "POST":
                h.respond(405, b"Method not allowed")
                return True
            try:
                fields = self.auth.form(h)
            except ValueError:
                h.respond(403, b"Invalid login form")
                return True
            if fields.get("network", "office") not in ("office", "internet", "business"):
                h.respond(422, b"Invalid network environment")
                return True
            if fields.get("employee") != data["employee"]["number"]:
                h.respond(
                    422,
                    layout(
                        "请选择账号",
                        '<h1>请选择有效的测试账号</h1><a href="'
                        + self.login_path(ns)
                        + '">返回登录</a>',
                    ),
                )
                return True
            if not self.auth.consume_challenge(h, ns, fields.get("csrf", "")):
                h.respond(403, b"Login form expired or invalid")
                return True
            try:
                cookies = self.auth.native_cookies(ns)
            except Exception:
                # Private native login errors never expose account secrets or response bodies.
                h.respond(
                    503,
                    layout(
                        "应用连接暂不可用",
                        "<h1>应用登录暂不可用</h1><p>请稍后从测试登录页重试。</p>",
                    ),
                )
                return True
            token = self.auth.issue(ns, cookies)
            values = [
                self.auth.cookie(self.auth.cookie_name(ns), token),
                self.auth.cookie(self.auth.cookie_name(ns, "login"), "", 0),
            ]
            values += [self.auth.cookie(c["name"], c["value"]) for c in cookies]
            self.redirect(h, "/workbench/" + ns, values)
            return True
        parts = path.strip("/").split("/")
        if len(parts) >= 2 and parts[0] in ("calendar", "timesheet", "office"):
            ns = parts[1]
            try:
                self.config(ns)
            except (ValueError, OSError, KeyError):
                return False
            current = self.require(h, ns, method)
            if not current:
                return True
            if parts[0] == "timesheet":
                from ..timesheet.app import handle

                return handle(h, self, ns, current, method, path)
            if parts[0] == "office":
                if method != "GET":
                    h.respond(405, b"Method not allowed")
                    return True
                key = parts[2] if len(parts) == 3 else ""
                if key == "tickets":
                    self.redirect(h, "/workbench/" + ns)
                    return True
                elif key in DISTRACTORS:
                    title, _, content = DISTRACTORS[key]
                    body = (
                        "<h1>"
                        + esc(title)
                        + '</h1><section class="panel"><h2>应用内容</h2><p>'
                        + esc(content)
                        + "</p></section>"
                    )
                else:
                    h.respond(404, b"Unknown office application")
                    return True
                h.respond(
                    200,
                    layout(
                        "办公应用", body + '<p><a href="/workbench/' + ns + '">返回工作台</a></p>'
                    ),
                )
                return True
        return False
