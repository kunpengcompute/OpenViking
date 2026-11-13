"""Mail scope: exact subject/sender selection, complete body export, no work orders."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark.office.cases.mail.data import (
    body,
    check_export,
    fixture,
    metadata,
    task,
)
from benchmark.office.cases.page_layer import (
    load_registry,
)
from benchmark.office.cases.portal.auth import (
    OfficeAuth,
)
from benchmark.office.cases.portal.data import (
    provision,
)


class MailReadTest(unittest.TestCase):
    def test_readonly_task_and_metadata_exclude_reference_body(self):
        config = {"mail_read": metadata()}
        text = task(config)
        self.assertIn("iframe", text)
        self.assertIn("content.txt", text)
        self.assertNotIn("工单", text)
        self.assertNotIn("对应项目", text)
        self.assertNotIn(body("H"), json.dumps(config, ensure_ascii=False))

    def test_phase_isolation_and_subject_sender_disambiguation(self):
        data = fixture("synthetic@example.test")
        rows = {r["id"]: r for r in data["messages"]}
        self.assertFalse(set(data["snapshots"]["history"]) & set(data["snapshots"]["target"]))
        for phase, snapshot in [("H", "history"), ("Q", "target")]:
            rule = metadata(phase)
            visible = [rows[mid] for mid in data["snapshots"][snapshot]]
            self.assertEqual(32, len(visible))
            same = [r for r in visible if r["subject"] == rule["subject"]]
            exact = [r for r in same if r["from_name"] == rule["sender"]]
            self.assertEqual(2, len(same))
            self.assertEqual(1, len(exact))
            self.assertEqual(body(phase), exact[0]["body"])
        self.assertNotEqual(body("H"), body("Q"))

    def test_export_requires_complete_body(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "content.txt"
            self.assertFalse(check_export(p)["file_exists"])
            p.write_text(body("H"))
            self.assertTrue(check_export(p)["body_complete"])
            p.write_text(body("H").replace("\n", "\r\n"))
            self.assertTrue(check_export(p)["body_complete"])
            for text in [body("H")[:100], body("Q"), "邮件主题：test\n" + body("H")]:
                p.write_text(text)
                self.assertFalse(check_export(p)["body_complete"])

    def test_fresh_office_workspace_needs_no_redmine_project(self):
        with tempfile.TemporaryDirectory() as folder:
            provision(folder, "mail-only", anchor="2026-10-28T10:00:00+08:00")
            data = load_registry(Path(folder), "mail-only")
            self.assertEqual([], data["projects"])
            self.assertEqual(metadata(), data["mail_read"])

    def test_native_login_uses_mail_without_redmine(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            private = root / "office-private"
            private.mkdir()
            p = private / "mail-only.local.json"
            p.write_text(json.dumps({"mail": {"username": "test"}}))
            p.chmod(0o600)

            def login(settings, path):
                self.assertEqual("test", settings["username"])
                Path(path).write_text(
                    json.dumps({"cookies": [{"name": "roundcube_sessid", "value": "test-cookie"}]})
                )

            with patch(
                "benchmark.office.admin.mail.login_state",
                side_effect=login,
            ):
                cookies = OfficeAuth(root).native_cookies("mail-only")
            self.assertEqual(["roundcube_sessid"], [c["name"] for c in cookies])


if __name__ == "__main__":
    unittest.main()
