# SPDX-License-Identifier: AGPL-3.0-only
"""Exercise patch delivery with real temporary Git repositories, without network."""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import patch_manager as pm  # noqa: E402 -- load the adjacent standalone CLI module


class PatchManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ov-tools-test-")
        self.root = Path(self.temp.name).resolve()
        self.repo = self.root / "dev"
        self.repo.mkdir()
        pm.git(self.repo, "init", "--quiet")
        pm.git(self.repo, "config", "user.name", "Patch Test")
        pm.git(self.repo, "config", "user.email", "test@example.invalid")
        (self.repo / "a.txt").write_text("base\n")
        (self.repo / "remove.txt").write_text("remove me\n")
        (self.repo / "binary.bin").write_bytes(b"\x00old\xff")
        (self.repo / "run.sh").write_text("#!/bin/sh\ntrue\n")
        pm.git(self.repo, "add", ".")
        pm.git(self.repo, "commit", "--quiet", "-m", "baseline")
        self.base = pm.object_id(self.repo, "HEAD")
        pm.git(self.repo, "tag", "v0.4.1")
        (self.repo / "a.txt").write_text("changed\n")
        (self.repo / "remove.txt").rename(self.repo / "renamed.txt")
        (self.repo / "binary.bin").write_bytes(b"\x00new\xff\x01")
        (self.repo / "run.sh").chmod(0o755)
        (self.repo / "新增.txt").write_text("新文件\n")
        os.symlink("a.txt", self.repo / "shortcut")
        pm.git(self.repo, "add", "-A")
        pm.git(self.repo, "commit", "--quiet", "-m", "feature")
        self.source = pm.object_id(self.repo, "HEAD")
        self.tree = pm.git(self.repo, "rev-parse", "HEAD^{tree}").decode().strip()
        self.folder = self.root / "template"
        self.folder.mkdir()
        for name in pm.METADATA:
            shutil.copyfile(TOOLS / name, self.folder / name)
        manifest = json.loads((self.folder / "manifest.json").read_text())
        manifest.update(
            base_ref=self.base,
            base_repository=self.repo.as_uri(),
            excluded_paths=[],
            status="template",
        )
        for key in ("source_commit", "source_tree", "development_tree"):
            manifest.pop(key, None)
        manifest["groups"][0]["patterns"] = ["**"]
        (self.folder / "manifest.json").write_text(json.dumps(manifest))
        pm.checksums(self.folder)
        self.release = self.root / "release"

    def tearDown(self):
        self.temp.cleanup()

    def generate(self):
        pm.generate(self.folder, self.repo, "HEAD", self.release)
        return pm.load_manifest(self.release)

    def test_template_integrity_and_no_false_release(self):
        self.assertEqual(pm.verify(self.folder)["status"], "template")
        with self.assertRaisesRegex(pm.PatchError, "no released"):
            pm.verify(self.folder, self.repo)
        with self.assertRaisesRegex(pm.PatchError, "no released"):
            pm.materialize(self.folder, self.repo, "full-default", self.root / "ready")

    def test_roundtrip_preserves_text_binary_rename_mode_symlink_and_source(self):
        manifest = self.generate()
        self.assertEqual(manifest["source_commit"], self.source)
        self.assertEqual(manifest["source_tree"], self.tree)
        pm.verify(self.release, self.repo)
        checkout = self.root / "ready"
        pm.materialize(self.release, self.repo, "full-default", checkout)
        self.assertEqual(pm.git(checkout, "write-tree").decode().strip(), self.tree)
        self.assertEqual((checkout / "binary.bin").read_bytes(), b"\x00new\xff\x01")
        self.assertEqual((checkout / "新增.txt").read_text(), "新文件\n")
        self.assertFalse((checkout / "remove.txt").exists())
        self.assertTrue((checkout / "renamed.txt").exists())
        self.assertTrue((checkout / "run.sh").stat().st_mode & 0o111)
        self.assertEqual(os.readlink(checkout / "shortcut"), "a.txt")
        self.assertEqual(pm.object_id(checkout, "HEAD"), self.base)
        receipt = json.loads((checkout / ".git/openviking-patches.json").read_text())
        self.assertEqual(receipt["source_commit"], self.source)
        self.assertEqual(pm.object_id(self.repo, "HEAD"), self.source)
        self.assertEqual(pm.git(self.repo, "status", "--porcelain"), b"")

    def test_prepare_wrapper_clones_pinned_tag_without_existing_checkout(self):
        self.generate()
        output = self.root / "downloaded"
        result = subprocess.run(
            [
                sys.executable,
                str(self.release / "prepare_source.py"),
                "--output-dir",
                str(output),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(pm.git(output, "write-tree").decode().strip(), self.tree)

    def test_materialized_repository_does_not_depend_on_source_alternates(self):
        self.generate()
        borrowed = self.root / "borrowed"
        pm.git(self.root, "clone", "--shared", str(self.repo), str(borrowed))
        output = self.root / "independent"
        pm.materialize(self.release, borrowed, "full-default", output)
        self.repo.rename(self.root / "unavailable-source")
        pm.git(output, "fsck", "--connectivity-only", "--no-dangling")
        self.assertEqual(pm.object_id(output, "HEAD"), self.base)
        self.assertFalse((output / ".git/objects/info/alternates").exists())

    def test_profiles_apply_in_order_and_full_profile_matches_tree(self):
        manifest = pm.load_manifest(self.folder)
        manifest["groups"] = [
            {
                "name": "text",
                "output": "feature/0001-text.patch",
                "patterns": ["*.txt"],
            },
            {
                "name": "other",
                "output": "feature/0002-other.patch",
                "patterns": ["**"],
                "exclude": ["*.txt"],
            },
        ]
        manifest["profiles"] = {
            "full-default": ["text", "other"],
            "text-only": ["text"],
        }
        (self.folder / "manifest.json").write_text(json.dumps(manifest))
        self.generate()
        pm.verify(self.release, self.repo)
        target = self.root / "text-only"
        pm.materialize(self.release, self.repo, "text-only", target)
        self.assertEqual((target / "a.txt").read_text(), "changed\n")
        self.assertEqual((target / "binary.bin").read_bytes(), b"\x00old\xff")

    def test_audit_rejects_unowned_and_multiply_owned_paths(self):
        manifest = pm.load_manifest(self.folder)
        for mode in ("unowned", "overlap"):
            with self.subTest(mode=mode):
                value = copy.deepcopy(manifest)
                if mode == "unowned":
                    value["groups"][0]["patterns"] = ["not-a-changed-path"]
                else:
                    value["groups"].append({"name": "overlap", "patterns": ["**"]})
                with self.assertRaisesRegex(pm.PatchError, "exactly one owner"):
                    pm.audit(value, self.repo, "HEAD")

    def test_export_ignores_local_text_conversion_drivers(self):
        marker = self.root / "textconv-ran"
        converter = self.root / "textconv.py"
        converter.write_text(
            "from pathlib import Path\n"
            + f"Path({str(marker)!r}).write_text('ran')\n"
            + "print('converted view')\n"
        )
        (self.repo / ".gitattributes").write_text("a.txt diff=smoke\n")
        pm.git(self.repo, "add", ".gitattributes")
        pm.git(self.repo, "commit", "--quiet", "-m", "attribute fixture")
        pm.git(self.repo, "config", "diff.smoke.textconv", f"{sys.executable} {converter}")
        self.generate()
        self.assertFalse(marker.exists())
        pm.verify(self.release, self.repo)

    def test_dirty_source_is_rejected_and_preserved(self):
        (self.repo / "a.txt").write_text("uncommitted\n")
        with self.assertRaisesRegex(pm.PatchError, "working-tree"):
            self.generate()
        self.assertFalse(self.release.exists())
        self.assertEqual((self.repo / "a.txt").read_text(), "uncommitted\n")

    def test_existing_outputs_are_preserved(self):
        self.generate()
        before = (self.release / "SHA256SUMS").read_bytes()
        with self.assertRaisesRegex(pm.PatchError, "already exists"):
            self.generate()
        with self.assertRaisesRegex(pm.PatchError, "already exists"):
            pm.materialize(self.release, self.repo, "full-default", self.repo)
        self.assertEqual((self.release / "SHA256SUMS").read_bytes(), before)
        self.assertEqual(pm.object_id(self.repo, "HEAD"), self.source)

    def test_generate_inside_source_is_rejected(self):
        with self.assertRaisesRegex(pm.PatchError, "outside"):
            pm.generate(self.folder, self.repo, "HEAD", self.repo / "release")
        self.assertEqual(pm.git(self.repo, "status", "--porcelain"), b"")

    def test_checksum_corruption_and_unlisted_patch_are_rejected(self):
        self.generate()
        patch = self.release / "feature/0001-openviking-context-integration.patch"
        patch.write_bytes(patch.read_bytes() + b"tamper\n")
        with self.assertRaisesRegex(pm.PatchError, "Checksum mismatch"):
            pm.verify(self.release)
        (self.release / "extra.patch").write_text("unlisted")
        with self.assertRaisesRegex(pm.PatchError, "inventory mismatch"):
            pm.verify(self.release)

    def test_failed_application_never_publishes_output(self):
        self.generate()
        patch = self.release / "feature/0001-openviking-context-integration.patch"
        patch.write_text("invalid patch\n")
        pm.checksums(self.release)
        output = self.root / "failed"
        with self.assertRaises(pm.PatchError):
            pm.materialize(self.release, self.repo, "full-default", output)
        self.assertFalse(output.exists())
        self.assertEqual(pm.object_id(self.repo, "HEAD"), self.source)
        self.assertEqual(pm.git(self.repo, "status", "--porcelain"), b"")

    def test_tree_mismatch_and_moved_tag_are_rejected(self):
        self.generate()
        path = self.release / "manifest.json"
        manifest = pm.load_manifest(self.release)
        manifest["source_tree"] = "0" * 40
        path.write_text(json.dumps(manifest))
        pm.checksums(self.release)
        with self.assertRaisesRegex(pm.PatchError, "tree differs"):
            pm.verify(self.release, self.repo)
        manifest["base_ref"] = self.source
        path.write_text(json.dumps(manifest))
        pm.checksums(self.release)
        with self.assertRaisesRegex(pm.PatchError, "tag moved"):
            pm.materialize(self.release, None, "full-default", self.root / "tag-moved")
        self.assertFalse((self.root / "tag-moved").exists())

    def test_manifest_paths_and_metadata_symlinks_are_rejected(self):
        original = pm.load_manifest(self.folder)
        for output in (
            "../outside.patch",
            "/tmp/outside.patch",
            "feature/../../outside.patch",
        ):
            with self.subTest(output=output):
                manifest = copy.deepcopy(original)
                manifest["groups"][0]["output"] = output
                (self.folder / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaises(pm.PatchError):
                    pm.load_manifest(self.folder)
        (self.folder / "manifest.json").write_text(json.dumps(original))
        outside = self.root / "outside"
        outside.write_text("private")
        (self.folder / "README.md").unlink()
        (self.folder / "README.md").symlink_to(outside)
        with self.assertRaises(pm.PatchError):
            pm.checksums(self.folder)
        self.assertEqual(outside.read_text(), "private")

    def test_invalid_profile_and_missing_baseline_fail_cleanly(self):
        self.generate()
        with self.assertRaisesRegex(pm.PatchError, "Unknown feature"):
            pm.materialize(self.release, self.repo, "missing", self.root / "missing-profile")
        with self.assertRaises(pm.PatchError):
            pm.object_id(self.repo, "0" * 40)
        self.assertFalse((self.root / "missing-profile").exists())

    def test_export_excludes_internal_paths_and_records_public_tree(self):
        manifest = pm.load_manifest(self.folder)
        manifest["excluded_paths"] = ["examples/*", "benchmark/*", "tests/*"]
        (self.folder / "manifest.json").write_text(json.dumps(manifest))
        for name in ("examples/chat_server.py", "benchmark/office/task.md", "tests/private.py"):
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("internal-only-marker\n")
        pm.git(self.repo, "add", ".")
        pm.git(self.repo, "commit", "--quiet", "-m", "internal tools")
        manifest = self.generate()
        self.assertNotEqual(manifest["source_tree"], manifest["development_tree"])
        patch = (self.release / "feature/0001-openviking-context-integration.patch").read_text()
        self.assertNotIn("internal-only-marker", patch)
        pm.verify(self.release, self.repo)
        checkout = self.root / "public-source"
        pm.materialize(self.release, self.repo, "full-default", checkout)
        self.assertFalse((checkout / "examples/chat_server.py").exists())
        self.assertEqual((checkout / "a.txt").read_text(), "changed\n")
        self.assertEqual(pm.git(checkout, "write-tree").decode().strip(), manifest["source_tree"])

    def test_exclusion_patterns_require_a_list_of_strings(self):
        manifest = pm.load_manifest(self.folder)
        for value in ("examples/*", [None]):
            manifest["excluded_paths"] = value
            (self.folder / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(pm.PatchError, "excluded_paths"):
                pm.load_manifest(self.folder)

    def test_public_template_owns_only_library_and_build_paths(self):
        manifest = pm.load_manifest(TOOLS)
        paths = (
            "openviking/integrations/langchain/middleware.py",
            "openviking_cli/client/http.py",
            "src/store/bytes_row.cpp",
            "setup.py",
            "pyproject.toml",
            "uv.lock",
        )
        import fnmatch

        for path in paths:
            self.assertTrue(
                any(
                    fnmatch.fnmatchcase(path, pattern)
                    for group in manifest["groups"]
                    for pattern in group["patterns"]
                )
            )
        for path in (
            "examples/deepagents_ov_context/chat_server.py",
            "benchmark/office/run.py",
            "tests/st/ov_context/run.py",
            "docs/design/private.md",
        ):
            self.assertTrue(
                any(fnmatch.fnmatchcase(path, pattern) for pattern in manifest["excluded_paths"])
            )

    def build_gate(self):
        root = TOOLS.parents[1]
        gate = self.root / "pipeline-checkout"
        gate.mkdir()
        paths = pm.git(root, "ls-files", "-z").decode().strip("\0").split("\0")
        for name in paths:
            target = gate / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, target)
        self.generate()
        bundle = gate / "patches/OpenViking_V0.4.1"
        shutil.rmtree(bundle)
        shutil.copytree(self.release, bundle)
        pm.git(gate, "init", "--quiet")
        pm.git(gate, "add", ".")
        return gate, bundle

    def run_gate(self, gate, *, cached=True, trace=False):
        env = {**os.environ, "TMPDIR": str(self.root)}
        env.pop("OV_PATCH_SOURCE_TREE", None)
        if cached:
            env["OV_PATCH_SOURCE_TREE"] = str(self.repo)
        if trace:
            env["GIT_TRACE"] = str(self.root / "ci-git-trace.log")
        return subprocess.run(
            ["bash", str(gate / "build_ci.sh")],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_build_gate_applies_patch_once_and_ignores_document_links(self):
        gate, _ = self.build_gate()
        readme = gate / "README.md"
        readme.write_text(readme.read_text() + "\n[broken](missing-delivery-file.md)\n")
        result = self.run_gate(gate, cached=False, trace=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        commands = (self.root / "ci-git-trace.log").read_text().splitlines()
        applies = [line for line in commands if "built-in: git apply --index " in line]
        self.assertEqual(len(applies), 2, applies)
        self.assertEqual(sum("--check" in line for line in applies), 1)

    def test_build_gate_rejects_unappliable_patch_with_valid_checksums(self):
        gate, bundle = self.build_gate()
        patch = bundle / "feature/0001-openviking-context-integration.patch"
        patch.write_text("invalid patch\n")
        pm.checksums(bundle)
        result = self.run_gate(gate)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(pm.git(self.repo, "status", "--porcelain"), b"")
        self.assertEqual(pm.object_id(self.repo, "HEAD"), self.source)

    def test_build_gate_rejects_reconstructed_tree_mismatch(self):
        gate, bundle = self.build_gate()
        path = bundle / "manifest.json"
        manifest = pm.load_manifest(bundle)
        manifest["source_tree"] = "0" * 40
        path.write_text(json.dumps(manifest))
        pm.checksums(bundle)
        result = self.run_gate(gate)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("tree differs", result.stderr)

    def test_build_gate_downloads_pinned_baseline_and_cleans_temporary_source(self):
        gate, _ = self.build_gate()
        result = self.run_gate(gate, cached=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("all source profiles", result.stdout)
        self.assertEqual(list(self.root.glob("ov-patch-ci.*")), [])

    def test_build_gate_rejects_moved_upstream_tag(self):
        gate, _ = self.build_gate()
        pm.git(self.repo, "tag", "--force", "v0.4.1", self.source)
        result = self.run_gate(gate, cached=False)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("tag moved", result.stderr)
        self.assertEqual(list(self.root.glob("ov-patch-ci.*")), [])

    def test_verify_cli_reports_release_integrity(self):
        self.generate()
        result = subprocess.run(
            [sys.executable, str(self.release / "patch_manager.py"), "verify"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Verified release integrity", result.stdout)


if __name__ == "__main__":
    unittest.main()
