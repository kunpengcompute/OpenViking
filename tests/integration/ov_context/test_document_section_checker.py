"""Exercise the document Skill's task-specific Markdown checker through its CLI."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SKILL = ROOT / "examples/deepagents_ov_context/chat/skills/office-documents"
CHECKER = SKILL / "references/check_document_sections.py"
SECTIONS = ("范围", "目标", "规则", "数据", "验收", "来源")


class DocumentSectionCheckerTests(unittest.TestCase):
    def run_checker(self, content, sections=SECTIONS):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "requirements.md"
            path.write_bytes(content)
            command = [sys.executable, str(CHECKER), str(path)]
            if sections:
                command.extend(["--sections", *sections])
            return subprocess.run(command, capture_output=True, text=True, check=False)

    def test_skill_routes_task_sections_to_configurable_checker(self):
        text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
        self.assertIn("references/check_document_sections.py", text)
        self.assertIn("--sections", text)
        self.assertNotIn("check_requirements_structure.py 做通用结构检查", text)

    def test_six_task_sections_pass_without_template_headings(self):
        content = "\n".join(f"## {name}\n本章正文。" for name in SECTIONS)
        process = self.run_checker(content.encode("utf-8"))
        result = json.loads(process.stdout)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertTrue(result["passed"])
        self.assertTrue(all(result["checks"].values()))

    def test_missing_section_reports_its_name(self):
        content = "\n".join(f"## {name}\n本章正文。" for name in SECTIONS[:-1])
        process = self.run_checker(content.encode("utf-8"))
        result = json.loads(process.stdout)
        self.assertEqual(process.returncode, 1)
        self.assertFalse(result["passed"])
        failed = [name for name, passed in result["checks"].items() if not passed]
        self.assertEqual(failed, ["section:来源"])

    def test_headings_without_body_fail(self):
        content = "\n".join(f"## {name}" for name in SECTIONS)
        process = self.run_checker(content.encode("utf-8"))
        result = json.loads(process.stdout)
        self.assertEqual(process.returncode, 1)
        self.assertFalse(result["checks"]["readable_nonempty_body"])

    def test_invalid_utf8_returns_a_read_error(self):
        process = self.run_checker(b"\xff")
        result = json.loads(process.stdout)
        self.assertEqual(process.returncode, 1)
        self.assertEqual(result["checks"], {"readable_file": False})

    def test_directory_is_explicit_instead_of_a_default_template(self):
        process = self.run_checker(b"# Document\nBody", sections=())
        self.assertEqual(process.returncode, 2)
        self.assertIn("--sections", process.stderr)


if __name__ == "__main__":
    unittest.main()
