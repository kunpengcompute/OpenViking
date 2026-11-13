"""Exercise calendar persistence, scope, CSRF, relative dates and duplicate saves."""

import json
import tempfile
import threading
import unittest
from html.parser import HTMLParser
from http.cookiejar import CookieJar
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener

from benchmark.office.cases.page_layer import (
    PageHandler,
    load_registry,
    workbench,
)
from benchmark.office.cases.schedule.app import (
    CalendarApp,
)
from benchmark.office.cases.schedule.data import (
    expected,
    inventory,
    objective_checks,
    provision,
)


class Hidden(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.fields = {}
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "input" and a.get("type") == "hidden":
            self.fields[a["name"]] = a["value"]


class CalendarTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.spec = provision(self.folder, "calendar-test", "2026-10-26T11:28:00+08:00")
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), PageHandler)
        self.server.registry = self.folder
        self.server.calendar = CalendarApp(self.folder)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = "http://127.0.0.1:" + str(self.server.server_port)
        self.client = build_opener(HTTPCookieProcessor(CookieJar()))

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def form(self):
        self.client.open(self.base + "/workbench/calendar-test").read()
        text = self.client.open(self.base + "/calendar/calendar-test").read().decode()
        data = Hidden(text).fields
        data.update(
            title="产品评审",
            calendar_id="personal",
            start_date="2026-10-27",
            start_time="11:28",
            end_date="2026-10-27",
            end_time="11:33",
        )
        self.assertIn("月视图", text)
        self.assertIn('data-day="2026-10-27"', text)
        return data

    def post(self, data, origin=None, client=None):
        return (client or self.client).open(
            Request(
                self.base + "/calendar/calendar-test/events",
                data=urlencode(data).encode(),
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    **({"Origin": origin} if origin else {}),
                },
            )
        )

    def test_valid_form_persists_and_verifies(self):
        before = inventory(self.folder, "calendar-test")
        text = self.post(self.form()).read().decode()
        self.assertIn("已保存", text)
        self.assertIn("查看目标日期", text)
        after = inventory(self.folder, "calendar-test")
        checks = objective_checks(self.spec["expected"], before, after)
        self.assertTrue(checks["backend_task_completed"])
        self.assertFalse(checks["detail_page_loaded"])
        event = next(e for e in after["events"] if e["source"] == "agent")
        detail = (
            self.client.open(self.base + "/calendar/calendar-test/events/" + str(event["id"]))
            .read()
            .decode()
        )
        self.assertIn("5分钟", detail)
        self.assertIsNone(checks["task_score"])

    def test_double_submit_same_nonce_is_idempotent(self):
        data = self.form()
        self.post(data).read()
        self.post(data).read()
        self.assertEqual(
            1,
            len(
                [
                    r
                    for r in inventory(self.folder, "calendar-test")["events"]
                    if r["source"] == "agent"
                ]
            ),
        )

    def test_bad_csrf_and_cross_origin_rejected(self):
        data = self.form()
        bad = {**data, "csrf": "invalid"}
        for body, origin in [(bad, None), (data, "https://example.invalid")]:
            with self.assertRaises(HTTPError) as exc:
                self.post(body, origin)
            self.assertEqual(403, exc.exception.code)
        self.assertFalse(
            any(r["source"] == "agent" for r in inventory(self.folder, "calendar-test")["events"])
        )

    def test_unauthenticated_write_rejected(self):
        data = self.form()
        with self.assertRaises(HTTPError) as exc:
            self.post(data, client=build_opener())
        self.assertEqual(403, exc.exception.code)

    def test_invalid_dates_and_backwards_time_rejected(self):
        data = self.form()
        for change in [
            {"start_date": "2026-02-30"},
            {"end_time": "11:20"},
            {"calendar_id": "foreign"},
        ]:
            with self.assertRaises(HTTPError) as exc:
                self.post({**data, **change})
            self.assertEqual(422, exc.exception.code)
        self.assertEqual(42, len(inventory(self.folder, "calendar-test")["events"]))

    def test_wrong_calendar_does_not_pass_completion(self):
        before = inventory(self.folder, "calendar-test")
        self.post({**self.form(), "calendar_id": "department"}).read()
        self.assertFalse(
            objective_checks(
                self.spec["expected"], before, inventory(self.folder, "calendar-test")
            )["backend_task_completed"]
        )

    def test_existing_registry_not_overwritten(self):
        with self.assertRaises(FileExistsError):
            provision(self.folder, "calendar-test")

    def test_workbench_keeps_old_case_links(self):
        old = {
            "namespace": "old-cross",
            "case": "cross",
            "projects": [
                {"identifier": "old-cross-northstar-platform", "name": "北辰平台"},
                {"identifier": "old-cross-northstar-data", "name": "北辰数据"},
            ],
        }
        (self.folder / "old-cross.json").write_text(json.dumps(old))
        text = workbench(load_registry(self.folder, "old-cross")).decode()
        self.assertIn("办公邮箱", text)
        self.assertNotIn("/calendar/", text)
        self.assertIn("/projects/old-cross-northstar-platform", text)

    def test_namespace_path_escape_rejected(self):
        with self.assertRaises(ValueError):
            provision(self.folder, "../escape")

    def test_relative_time_crosses_month_year_and_midnight(self):
        for anchor, start, end in [
            ("2026-10-31T23:58:00+08:00", "2026-11-01T23:58+08:00", "2026-11-02T00:03+08:00"),
            ("2026-12-31T10:30:00+08:00", "2027-01-01T10:30+08:00", "2027-01-01T10:35+08:00"),
        ]:
            value = expected(anchor)
            self.assertEqual(start, value["start"])
            self.assertEqual(end, value["end"])

    def test_seed_has_no_precreated_answer(self):
        spec = provision(self.folder, "seed-overlap", "2026-10-06T09:00:00+08:00")
        rows = inventory(self.folder, "seed-overlap")["events"]
        self.assertFalse(
            any(
                all(
                    r[k] == spec["expected"][k]
                    for k in ["owner", "calendar_id", "title", "start", "end"]
                )
                for r in rows
            )
        )


if __name__ == "__main__":
    unittest.main()
