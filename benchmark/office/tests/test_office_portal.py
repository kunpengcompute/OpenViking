"""Unified workbench login, scope, navigation and persisted timesheet contracts."""

import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from http.cookiejar import CookieJar
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener

from benchmark.office.cases.page_layer import (
    PageHandler,
)
from benchmark.office.cases.portal.app import (
    OfficePortal,
)
from benchmark.office.cases.portal.data import provision
from benchmark.office.cases.schedule.app import (
    CalendarApp,
)
from benchmark.office.cases.schedule.data import (
    inventory as calendar_inventory,
)
from benchmark.office.cases.timesheet.data import (
    inventory,
    timesheet_checks,
    upgrade_timesheet,
)


class Inputs(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.fields = {}
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "input" and a.get("name") and a.get("type") == "hidden":
            self.fields[a["name"]] = a.get("value", "")


class PortalStructure(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.region = None
        self.articles = {}
        self.depth = 0
        self.region_depth = None
        self.radios = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "input" and attrs.get("type") == "radio":
            self.radios.append(attrs)
        if tag not in ("input", "meta", "link", "br", "hr", "img"):
            self.depth += 1
        if attrs.get("id") in ("subscribed", "app-catalog"):
            self.region = attrs["id"]
            self.region_depth = self.depth
            self.articles[self.region] = 0
        if tag == "article" and self.region:
            self.articles[self.region] += 1

    def handle_endtag(self, tag):
        if self.region and self.depth == self.region_depth:
            self.region = None
        self.depth -= 1


class OfficePortalTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.ns = "office-test"
        self.spec = self.seed(self.ns)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), PageHandler)
        self.server.registry = self.folder
        self.server.calendar = CalendarApp(self.folder)
        self.server.portal = OfficePortal(self.folder)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = "http://127.0.0.1:" + str(self.server.server_port)
        self.client = self.browser()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def browser(self):
        return build_opener(HTTPCookieProcessor(CookieJar()))

    def seed(self, ns):
        return provision(
            self.folder,
            ns,
            [
                {"identifier": ns + "-northstar-" + k, "name": name, "id": i}
                for i, (k, name) in enumerate([("platform", "北辰平台"), ("data", "北辰数据")], 1)
            ],
            "2026-10-28T10:00:00+08:00",
        )

    def get(self, path, client=None):
        return (client or self.client).open(self.base + path)

    def post(self, path, fields, client=None, origin=None):
        return (client or self.client).open(
            Request(
                self.base + path,
                data=urlencode(fields).encode(),
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    **({"Origin": origin} if origin else {}),
                },
            )
        )

    def login(self, client=None, ns=None):
        ns = ns or self.ns
        client = client or self.client
        path = "/mock/login?workspace=" + ns
        text = self.get(path, client).read().decode()
        return (
            self.post(
                path,
                {**Inputs(text).fields, "employee": "100001", "employee_input": "100001"},
                client,
            )
            .read()
            .decode()
        )

    def form(self):
        self.login()
        text = self.get("/timesheet/" + self.ns + "/cards/demand-ops").read().decode()
        return {
            **Inputs(text).fields,
            "hours": "0.5",
            "content": "办公平台的验证测试",
            "notes": "办公平台的验证测试",
        }

    def submit(self, data, **kw):
        return self.post("/timesheet/" + self.ns + "/cards/demand-ops/entries", data, **kw)

    def test_login_redirect_and_all_apps_share_workbench(self):
        response = self.get("/workbench/" + self.ns)
        self.assertIn("/mock/login?", response.geturl())
        page = self.login()
        for text in [
            "办公邮箱",
            "日程管理",
            "电子看板（ST）",
            "已订阅",
            "未订阅",
            "知识库",
            "通讯录",
            "公告通知",
            "审批中心",
            "员工服务",
            "文件空间",
            "应用详情",
        ]:
            self.assertIn(text, page)
        self.assertIn('target="_blank"', page)
        for path in [
            "/calendar/" + self.ns,
            "/timesheet/" + self.ns,
            "/office/" + self.ns + "/knowledge",
        ]:
            self.assertEqual(200, self.get(path).status)

    def test_login_requires_valid_candidate_and_one_time_challenge(self):
        path = "/mock/login?workspace=" + self.ns
        text = self.get(path).read().decode()
        data = {**Inputs(text).fields, "employee": "999999"}
        with self.assertRaises(HTTPError) as e:
            self.post(path, data)
        self.assertEqual(422, e.exception.code)
        data["employee"] = "100001"
        self.post(path, data).read()
        with self.assertRaises(HTTPError) as e:
            self.post(path, data)
        self.assertEqual(403, e.exception.code)

    def test_login_network_and_account_picker_structure(self):
        path = "/mock/login?workspace=" + self.ns
        text = self.get(path).read().decode()
        structure = PortalStructure(text)
        self.assertEqual(["office", "internet", "business"], [r["value"] for r in structure.radios])
        self.assertIn("checked", structure.radios[0])
        for label in [
            "模拟网络环境",
            "请输入手机号、姓名、sapID、一事通ID搜索账号",
            "employee-choose",
            "测试账号候选",
            "点击创建",
            "返回登录页",
        ]:
            self.assertIn(label, text)
        with self.assertRaises(HTTPError) as e:
            self.post(path, {**Inputs(text).fields, "employee": "100001", "network": "other"})
        self.assertEqual(422, e.exception.code)
        self.assertIn("/mock/login?", self.get("/workbench/" + self.ns).geturl())

    def test_catalog_separate_from_subscribed_and_ambiguous_apps_readonly(self):
        page = self.login()
        structure = PortalStructure(page)
        self.assertEqual({"subscribed": 9, "app-catalog": 23}, structure.articles)
        self.assertIn("电子看板（历史查询）", page)
        self.assertIn("电子看板（ST）", page)
        self.assertIn("日程查询", page)
        self.assertIn("会议室预约", page)
        # Catalogue-only entries have no launch route; they cannot mutate work cards.
        before = inventory(self.folder, self.ns)
        with self.assertRaises(HTTPError) as e:
            self.get("/office/" + self.ns + "/board-history")
        self.assertEqual(404, e.exception.code)
        self.assertEqual(before, inventory(self.folder, self.ns))

    def test_timesheet_records_tab_and_saved_view(self):
        data = self.form()
        page = self.get("/timesheet/" + self.ns + "/cards/demand-ops").read().decode()
        self.assertRegex(page, r'id="work-records"[^>]* hidden')
        self.assertIn('aria-controls="work-records">报工记录', page)
        saved = self.submit(data).read().decode()
        self.assertRegex(saved, r'id="work-overview"[^>]* hidden')
        self.assertIn('id="total-hours">11</strong>', saved)
        self.assertIn("预计工时（小时）", saved)
        self.assertIn("44.5", saved)

    def test_board_columns_filters_and_similar_task_names(self):
        from benchmark.office.datasets.timesheet.catalog import (
            card_specs,
        )

        self.assertTrue(
            all(s["start"] <= s["due"] for s in card_specs("2026-10-28T10:00:00+08:00"))
        )
        self.login()
        base = "/timesheet/" + self.ns
        page = self.get(base).read().decode()
        self.assertEqual(24, page.count('<article class="work-card">'))
        target_column = page.split('aria-label="开发自测-todo"', 1)[1].split("</section>", 1)[0]
        self.assertIn("/cards/demand-ops", target_column)
        self.assertIn("需求与运营工作（归档）", page)
        filtered = (
            self.get(
                base
                + "?"
                + urlencode({"q": "需求与运营工作", "status": "开发自测-todo", "owner": "林澈"})
            )
            .read()
            .decode()
        )
        self.assertEqual(1, filtered.count('<article class="work-card">'))
        self.assertIn("/cards/demand-ops", filtered)
        self.assertNotIn("需求与运营工作（归档）", filtered)
        listing = self.get(base + "?view=list").read().decode()
        self.assertIn("累计已报工时", listing)

    def test_record_search_pagination_keep_full_total(self):
        self.login()
        base = "/timesheet/" + self.ns + "/cards/demand-ops"
        first = self.get(base + "?tab=records").read().decode()
        second = self.get(base + "?tab=records&page=2").read().decode()
        found = (
            self.get(base + "?" + urlencode({"tab": "records", "entry_q": "办公平台的验证测试"}))
            .read()
            .decode()
        )
        self.assertIn("第 1 / 2 页", first)
        self.assertIn("第 2 / 2 页", second)
        self.assertIn("办公平台的验证测试", second)
        for page in [first, second, found]:
            self.assertIn('id="total-hours">10.5</strong>', page)
        self.assertIn("匹配 1 条", found)

    def test_fixture_upgrade_is_idempotent_and_keeps_user_records(self):
        data = self.form()
        self.submit(data).read()
        before = inventory(self.folder, self.ns)
        self.assertEqual(57, len(before["entries"]))
        self.assertFalse(upgrade_timesheet(self.folder, self.ns))
        self.assertEqual(before, inventory(self.folder, self.ns))

    def test_work_date_conflicting_replay_and_activity(self):
        data = self.form()
        before = inventory(self.folder, self.ns)
        with self.assertRaises(HTTPError) as e:
            self.submit({**data, "work_date": "2026-02-30"})
        self.assertEqual(422, e.exception.code)
        self.assertEqual(before, inventory(self.folder, self.ns))
        data["work_date"] = "2026-10-27"
        self.submit(data).read()
        with self.assertRaises(HTTPError) as e:
            self.submit({**data, "hours": "1"})
        self.assertEqual(409, e.exception.code)
        after = inventory(self.folder, self.ns)
        self.assertEqual(30, after["demand_minutes"] - before["demand_minutes"])
        page = self.get("/timesheet/" + self.ns + "/cards/demand-ops?tab=activity").read().decode()
        self.assertIn("新增报工 TS-", page)
        created = after["entries"][-1]
        self.assertEqual("2026-10-27", created["work_date"])

    def test_timesheet_bundle_and_filtered_ledger(self):
        self.login()
        base = "/timesheet/" + self.ns
        self.assertIn("javascript", self.get(base + "/ui.js").headers["Content-Type"])
        self.assertIn("text/css", self.get(base + "/ui.css").headers["Content-Type"])
        ledger = (
            self.get(
                base
                + "?"
                + urlencode({"view": "ledger", "q": "需求与运营工作", "status": "开发自测-todo"})
            )
            .read()
            .decode()
        )
        self.assertIn("10.5 小时 · 10 条", ledger)
        self.assertNotIn("版本核对", ledger)

    def test_cross_origin_login_is_rejected(self):
        path = "/mock/login?workspace=" + self.ns
        text = self.get(path).read().decode()
        with self.assertRaises(HTTPError) as e:
            self.post(
                path,
                {**Inputs(text).fields, "employee": "100001"},
                origin="https://example.invalid",
            )
        self.assertEqual(403, e.exception.code)

    def test_missing_and_cross_workspace_auth_cannot_write(self):
        data = self.form()
        self.seed("office-other")
        for path in [
            "/timesheet/office-other/cards/demand-ops/entries",
            "/calendar/office-other/events",
        ]:
            with self.assertRaises(HTTPError) as e:
                self.post(path, data)
            self.assertEqual(403, e.exception.code)
        with self.assertRaises(HTTPError) as e:
            self.submit(data, client=self.browser())
        self.assertEqual(403, e.exception.code)

    def test_same_content_new_form_adds_but_replay_does_not(self):
        before = inventory(self.folder, self.ns)
        data = self.form()
        self.submit(data).read()
        self.submit(data).read()
        proof = timesheet_checks(before, inventory(self.folder, self.ns))
        self.assertEqual(1, proof["created_count"])
        self.assertEqual(30, proof["delta_minutes"])
        self.assertTrue(proof["correct_entry"])
        self.assertTrue(proof["seed_unchanged"])
        text = self.get("/timesheet/" + self.ns + "/cards/demand-ops").read().decode()
        data.update(Inputs(text).fields)
        self.submit(data).read()
        self.assertEqual(
            60, inventory(self.folder, self.ns)["demand_minutes"] - before["demand_minutes"]
        )

    def test_invalid_timesheet_and_csrf_do_not_mutate(self):
        data = self.form()
        before = inventory(self.folder, self.ns)
        for change, status in [
            ({"csrf": "invalid"}, 403),
            ({"hours": "NaN"}, 422),
            ({"hours": "-1"}, 422),
            ({"hours": "0.001"}, 422),
            ({"content": ""}, 422),
        ]:
            with self.assertRaises(HTTPError) as e:
                self.submit({**data, **change})
            self.assertEqual(status, e.exception.code)
        self.assertEqual(before, inventory(self.folder, self.ns))

    def test_concurrent_replay_only_creates_once(self):
        data = self.form()
        before = inventory(self.folder, self.ns)
        # Read the same cookie jar; each worker gets its own opener.
        cookie = "; ".join(
            c.name + "=" + c.value
            for h in self.client.handlers
            if isinstance(h, HTTPCookieProcessor)
            for c in h.cookiejar
        )

        def send(_):
            request = Request(
                self.base + "/timesheet/" + self.ns + "/cards/demand-ops/entries",
                data=urlencode(data).encode(),
                headers={"Cookie": cookie},
            )
            return self.browser().open(request).read()

        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(send, range(3)))
        self.assertEqual(
            30, inventory(self.folder, self.ns)["demand_minutes"] - before["demand_minutes"]
        )

    def test_logout_revokes_access_and_fresh_login_keeps_data(self):
        data = self.form()
        self.submit(data).read()
        before = inventory(self.folder, self.ns)
        text = self.get("/workbench/" + self.ns).read().decode()
        result = self.post("/mock/logout?workspace=" + self.ns, Inputs(text).fields)
        self.assertIn("/mock/login?", result.geturl())
        self.assertIn("/mock/login?", self.get("/calendar/" + self.ns).geturl())
        self.login()
        self.assertEqual(before, inventory(self.folder, self.ns))

    def test_distractors_have_real_readonly_pages(self):
        self.login()
        before = inventory(self.folder, self.ns)
        for key in ["knowledge", "contacts", "news", "files", "approvals", "employee"]:
            page = self.get("/office/" + self.ns + "/" + key).read().decode()
            self.assertIn("应用内容", page)
        self.assertEqual(before, inventory(self.folder, self.ns))

    def test_calendar_stays_persistent_under_common_login(self):
        self.login()
        text = self.get("/calendar/" + self.ns).read().decode()
        data = {
            **Inputs(text).fields,
            "title": "产品评审",
            "calendar_id": "personal",
            "start_date": "2026-10-29",
            "start_time": "10:00",
            "end_date": "2026-10-29",
            "end_time": "10:05",
        }
        result = self.post("/calendar/" + self.ns + "/events", data).read().decode()
        self.assertIn("日程已保存", result)
        self.assertIn("查看目标日期", result)
        events = calendar_inventory(self.folder, self.ns)["events"]
        self.assertEqual(43, len(events))
        created = next(e for e in events if e["source"] == "agent")
        self.assertEqual("2026-10-29T10:05+08:00", created["end"])
        self.assertIn(
            "5分钟",
            self.get("/calendar/" + self.ns + "/events/" + str(created["id"])).read().decode(),
        )

    def test_calendar_template_and_asset_bundle_match(self):
        self.login()
        base = "/calendar/" + self.ns
        page = self.get(base).read().decode()
        self.assertIn('id="calendar-data"', page)
        self.assertIn("calendar-outlook-v2", page)
        self.assertIn('id="editor"', page)
        self.assertIn("/office/" + self.ns + "/contacts", page)
        self.assertNotIn("/office-info/" + self.ns, page)
        script = self.get(base + "/ui.js")
        self.assertIn("javascript", script.headers["Content-Type"])
        self.assertIn("querySelector('#calendar-data')", script.read().decode())
        style = self.get(base + "/ui.css")
        self.assertIn("text/css", style.headers["Content-Type"])
        self.assertIn(".month-board", style.read().decode())
        # All calendar filters can be deselected; no stale default calendar is applied.
        empty = self.get(base + "?calendar_filter=1").read().decode()
        self.assertIn("0项日程", empty)

    def test_native_login_error_does_not_grant_portal_access(self):
        path = "/mock/login?workspace=" + self.ns
        text = self.get(path).read().decode()
        with patch.object(
            self.server.portal.auth, "native_cookies", side_effect=RuntimeError("private detail")
        ):
            with self.assertRaises(HTTPError) as e:
                self.post(path, {**Inputs(text).fields, "employee": "100001"})
        self.assertEqual(503, e.exception.code)
        self.assertNotIn("private detail", e.exception.read().decode())
        self.assertIn("/mock/login?", self.get("/workbench/" + self.ns).geturl())

    def test_invalid_or_existing_workspace_rejected(self):
        with self.assertRaises(ValueError):
            self.seed("../escape")
        with self.assertRaises(FileExistsError):
            self.seed(self.ns)


if __name__ == "__main__":
    unittest.main()
