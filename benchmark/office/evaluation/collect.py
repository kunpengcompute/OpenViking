# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Collect explicit office Runs into a local, indexed AI review directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path, PurePosixPath

from benchmark.paths import PROJECT_ROOT

CASES = {"schedule", "timesheet", "mail", "ppt_update", "reqdoc_generation"}
RUN_FILES = (
    "condition.json",
    "result.json",
    "trace.json",
    "evidence.json",
    "session.json",
    "state.json",
    "events.jsonl",
    "stream-events.json",
    "stream-events.jsonl",
    "terminal.json",
    "answer.md",
    "tool-calls.json",
    "tool-events.json",
    "model-inputs.json",
    "first-human-input.txt",
    "first-input-gate.json",
    "input-receipt.json",
    "input-receipts.json",
    "artifact-downloads.json",
    "artifact-list-response.json",
    "memory-before.json",
    "memory-after.json",
    "memory-bodies.json",
    "memory-audit.json",
    "memory-acceptance.json",
    "frozen-memory.json",
    "preparation-review.json",
    "automatic-commit-evidence.json",
    "commit-receipt.json",
    "extraction-task.json",
    "failure.json",
    "runner-error.json",
    "chat-response-error.json",
)
PRIOR_FILES = ("grade.json", "automatic-grade.json", "automatic-reviews.json", "review.json")
SUFFIXES = {
    ".md",
    ".txt",
    ".json",
    ".jsonl",
    ".png",
    ".jpg",
    ".jpeg",
    ".pdf",
    ".pptx",
    ".docx",
    ".xlsx",
    ".py",
    ".csv",
    ".html",
}
HASH = re.compile(r"[0-9a-f]{64}")
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def component(value):
    return isinstance(value, str) and ID.fullmatch(value) is not None


class Bundle:
    def __init__(self, output):
        self.output, self.files, self.issues = output, {}, []

    def issue(self, code, source, detail):
        self.issues.append({"code": code, "source": str(source), "detail": detail})

    def copy(self, source, relative, expected=None):
        source = Path(source)
        if source.is_symlink() or any(p.is_symlink() for p in source.parents):
            self.issue("symlink", source, "Symbolic links are excluded")
            return
        if not source.is_file():
            self.issue("missing_file", source, "Required file is unavailable")
            return
        target = self.output / relative
        if relative in self.files:
            previous = self.files[relative]["sha256"]
            if previous != digest(source):
                self.issue("conflicting_file", source, f"Different sources map to {relative}")
            if expected is not None and previous != expected:
                self.issue("hash_mismatch", source, f"Expected {expected}, copied {previous}")
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        sha = digest(target)
        self.files[relative] = {
            "path": relative,
            "source": str(source),
            "sha256": sha,
            "bytes": target.stat().st_size,
            "kind": "original",
        }
        if sha != digest(source) or (expected is not None and sha != expected):
            self.issue(
                "hash_mismatch", source, f"Expected {expected or 'stable source'}, copied {sha}"
            )
        return target

    def generated(self, relative, value, source):
        path = self.output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        text = (
            value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        )
        path.write_text(text, encoding="utf-8")
        self.files[relative] = {
            "path": relative,
            "source": str(source),
            "sha256": digest(path),
            "bytes": path.stat().st_size,
            "kind": "derived",
        }

    def json(self, path, required=False):
        if not path.is_file():
            if required:
                self.issue("missing_file", path, "Required JSON is unavailable")
            return {}
        if path.is_symlink() or any(p.is_symlink() for p in path.parents):
            self.issue("symlink", path, "Symbolic links are excluded")
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            self.issue("invalid_json", path, str(exc))
            return {}

    def object(self, path, required=False):
        value = self.json(path, required=required)
        if isinstance(value, dict):
            return value
        self.issue("invalid_json_shape", path, "Expected a JSON object")
        return {}

    def tree(self, directory, destination):
        if directory.is_symlink():
            self.issue("symlink", directory, "Symbolic links are excluded")
            return
        if not directory.is_dir():
            return
        for path in sorted(directory.iterdir()):
            if path.is_symlink():
                self.issue("symlink", path, "Symbolic links are excluded")
            elif path.is_dir():
                self.tree(path, destination + "/" + path.name)
            elif path.suffix.lower() in SUFFIXES and not path.name.startswith("."):
                self.copy(path, destination + "/" + path.name)


def snapshots(bundle, audit, exported, destination):
    manifest = bundle.object(audit / "manifest.json") if audit else {}
    manifest = manifest or exported.get("manifest", {})
    rows = []
    if audit and (audit / "snapshots.jsonl").is_file():
        bundle.copy(audit / "manifest.json", destination + "/manifest.json")
        source = bundle.copy(audit / "snapshots.jsonl", destination + "/snapshots.jsonl")
        if source:
            for line, text in enumerate(source.read_text().splitlines(), 1):
                try:
                    rows.append(json.loads(text))
                except ValueError as exc:
                    bundle.issue("invalid_snapshot", f"{audit}/snapshots.jsonl:{line}", str(exc))
    else:
        rows = exported.get("snapshots", [])
        if manifest:
            bundle.generated(destination + "/manifest.json", manifest, "evidence.json#manifest")
        if rows:
            bundle.generated(
                destination + "/snapshots.jsonl",
                "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows),
                "evidence.json#snapshots",
            )
    if not rows:
        bundle.issue("missing_snapshots", audit or destination, "No execution snapshots available")
    elif rows[-1].get("event") != "run.end":
        bundle.issue("incomplete_run", audit or destination, "Terminal snapshot is missing")
    if not manifest:
        bundle.issue("missing_manifest", audit or destination, "Frozen case reference is missing")
    references = {}
    source = manifest.get("reference", {}).get("source")
    if isinstance(source, str):
        references.setdefault(source, set())
    for row in rows:
        if "capture_error" in row:
            bundle.issue(
                "capture_error", destination, json.dumps(row["capture_error"], ensure_ascii=False)
            )
        for field in ("files", "published"):
            for logical, sha in row.get(field, {}).items():
                if sha is None:
                    bundle.issue(
                        "missing_snapshot_file", logical, "Snapshot has an unreadable artifact"
                    )
                    continue
                if not isinstance(sha, str) or not HASH.fullmatch(sha):
                    bundle.issue("invalid_blob_hash", logical, str(sha))
                    continue
                references.setdefault(sha, set()).add(logical)
    for sha, names in references.items():
        if not HASH.fullmatch(sha):
            bundle.issue("invalid_blob_hash", destination, sha)
            continue
        blob = audit / "blobs" / sha if audit else Path("/nonexistent-audit") / sha
        copied = bundle.copy(blob, destination + "/blobs/" + sha, expected=sha)
        if copied:
            for name in names:
                logical = PurePosixPath(name)
                if logical.is_absolute() or ".." in logical.parts or "\\" in name:
                    bundle.issue(
                        "unsafe_snapshot_path", name, "Only relative artifact paths are accepted"
                    )
                    continue
                if logical.suffix.lower() in SUFFIXES:
                    bundle.copy(
                        blob, destination + "/versions/" + sha + "/" + str(logical), expected=sha
                    )
    return manifest


def inputs(bundle, folder, chat_data, sid, manifest, condition, source_root, destination):
    spec = manifest.get("reference", {}).get("spec", {})
    records = bundle.json(folder / "input-receipts.json")
    if not records:
        records = condition.get("inputs", [])
    if not records:
        receipt = bundle.object(folder / "input-receipt.json")
        records = [receipt] if receipt else spec.get("inputs", [])
    if not spec and condition.get("version"):
        candidate = bundle.object(
            source_root / "benchmark/office/cases/reqdoc_generation/spec.json"
        )
        if candidate.get("version") == condition["version"]:
            spec = candidate
    sources = {}
    if spec.get("source"):
        records = records or [
            {"filename": Path(spec["source"]).name, "sha256": spec["input_sha256"]}
        ]
        sources[Path(spec["source"]).name] = spec["source"]
    for item in spec.get("support_files", []):
        sources[item["filename"]] = item["source"]
    if not isinstance(records, list):
        bundle.issue("invalid_json_shape", folder, "Input receipts must be a list")
        return
    for item in records:
        if not isinstance(item, dict):
            bundle.issue("invalid_json_shape", folder, "Each input receipt must be an object")
            continue
        name, expected = item.get("filename"), item.get("sha256")
        if not isinstance(name, str) or Path(name).name != name or "\\" in name:
            bundle.issue("invalid_input_name", folder, str(name))
            continue
        candidates = [folder / "inputs" / name]
        if chat_data and component(sid):
            candidates.append(chat_data / "inputs" / sid / name)
        relative = sources.get(name) or (
            str(PurePosixPath(spec["input_dir"]) / name) if spec.get("input_dir") else None
        )
        if relative:
            p = PurePosixPath(relative)
            if p.is_absolute() or ".." in p.parts or "\\" in relative:
                bundle.issue(
                    "unsafe_input_path", relative, "Input source must be relative to source root"
                )
            else:
                candidates.append(source_root / relative)
        existing = next((p for p in candidates if p.is_file()), candidates[0])
        bundle.copy(existing, destination + "/" + name, expected=expected)


def collect(runs, output, *, chat_data=None, source_root=PROJECT_ROOT):
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output = output.resolve()
    source_root = Path(source_root).resolve()
    chat_data = Path(chat_data).resolve() if chat_data is not None else None
    if output.resolve().is_relative_to((source_root / "benchmark").resolve()):
        raise ValueError("Output must be outside benchmark source assets")
    selected = []
    seen = set()
    for item in runs:
        folder = Path(item["directory"]).resolve()
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        if output.is_relative_to(folder) or (chat_data and output.is_relative_to(chat_data)):
            raise ValueError("Output must be outside runner and Chat data directories")
        if folder.resolve() in seen:
            raise ValueError("Duplicate Run directory")
        seen.add(folder.resolve())
        selected.append((folder, item.get("audit_directory")))
    if not selected:
        raise ValueError("Select at least one Run")
    bundle = Bundle(output)
    loaded = []
    ids = set()
    for folder, override in selected:
        result = bundle.object(folder / "result.json", required=True)
        rid = result.get("run_id")
        if rid is not None and rid in ids:
            raise ValueError(f"Duplicate Run ID: {rid}")
        if rid is not None:
            ids.add(rid)
        loaded.append((folder, override, result))
    output.mkdir(parents=True)
    bundle.copy(
        source_root / "benchmark/office/evaluation/AI_REVIEW_PROMPT.md", "AI_REVIEW_PROMPT.md"
    )
    bundle.copy(source_root / "benchmark/office/evaluation/policy.json", "policy.json")
    index = []
    for ordinal, (folder, override, result) in enumerate(loaded, 1):
        condition = bundle.object(folder / "condition.json", required=True)
        exported = bundle.object(folder / "evidence.json")
        sid, rid = result.get("session_id"), result.get("run_id")
        audit = Path(override).resolve() if override else None
        if audit is None and chat_data and component(sid) and component(rid):
            audit = chat_data / "benchmark" / sid / rid
        if audit is None and exported.get("evidence_directory"):
            candidate = Path(exported["evidence_directory"])
            if candidate.is_absolute() and candidate.is_dir():
                if candidate.name == rid and candidate.parent.name == sid:
                    audit = candidate.resolve()
                else:
                    bundle.issue(
                        "identity_mismatch",
                        candidate,
                        "Exported audit directory differs from Session/Run",
                    )
        local_manifest = bundle.object(audit / "manifest.json") if audit else {}
        case = (
            result.get("case")
            or condition.get("case")
            or exported.get("case")
            or local_manifest.get("case")
        )
        phase = (
            result.get("phase")
            or condition.get("phase")
            or exported.get("phase")
            or local_manifest.get("phase")
        )
        mode = (
            result.get("mode")
            or result.get("group")
            or condition.get("mode")
            or condition.get("group")
            or condition.get("execution", {}).get("mode")
        )
        if case not in CASES:
            bundle.issue("unknown_case", folder, str(case))
        if phase not in {"H", "Q"}:
            bundle.issue("unknown_phase", folder, str(phase))
        if not component(mode):
            bundle.issue("unknown_group", folder, str(mode))
        if not component(rid) or not component(sid):
            bundle.issue("unknown_identity", folder, "Result must identify its Session and Run")
        prefix = f"runs/{ordinal:03}-{case if case in CASES else 'unknown'}-{mode if component(mode) else 'unknown'}"
        for name in RUN_FILES:
            if (folder / name).exists():
                bundle.copy(folder / name, prefix + "/execution/" + name)
        if not (folder / "trace.json").is_file():
            bundle.issue("missing_trace", folder, "Model and tool trace is unavailable")
        trace = bundle.object(folder / "trace.json")
        if trace.get("run_id") and trace["run_id"] != rid:
            bundle.issue("identity_mismatch", folder, "Trace and result Run IDs differ")
        for document in (condition, exported, local_manifest):
            for field, expected in (
                ("case", case),
                ("phase", phase),
                ("session_id", sid),
                ("run_id", rid),
            ):
                if document.get(field) is not None and document[field] != expected:
                    bundle.issue("identity_mismatch", folder, f"Conflicting {field}")
        manifest = snapshots(bundle, audit, exported, prefix + "/audit")
        query = condition.get("query")
        if isinstance(query, str):
            bundle.generated(prefix + "/task.md", query, str(folder / "condition.json") + "#query")
            expected = condition.get("query_sha256") or condition.get("prompt_sha256")
            if expected and hashlib.sha256(query.encode()).hexdigest() != expected:
                bundle.issue(
                    "hash_mismatch", folder / "condition.json", "Task differs from recorded hash"
                )
        else:
            bundle.issue("missing_task", folder, "Actual task text is unavailable")
        inputs(bundle, folder, chat_data, sid, manifest, condition, source_root, prefix + "/inputs")
        bundle.tree(folder / "artifacts", prefix + "/artifacts")
        if chat_data and component(sid) and component(rid):
            workspace = chat_data / "artifacts" / sid / rid
            for receipt_path in sorted(workspace.glob("*.artifact.json")):
                receipt = bundle.object(receipt_path)
                name = receipt.get("filename")
                if not isinstance(name, str) or Path(name).name != name or "\\" in name:
                    bundle.issue("invalid_artifact_name", receipt_path, str(name))
                    continue
                if receipt.get("run_id") != rid:
                    bundle.issue(
                        "identity_mismatch", receipt_path, "Published artifact Run differs"
                    )
                    continue
                bundle.copy(
                    receipt_path, prefix + "/execution/published-receipts/" + receipt_path.name
                )
                bundle.copy(workspace / name, prefix + "/artifacts/" + name)
        downloads = bundle.json(folder / "artifact-downloads.json")
        for receipt in downloads if isinstance(downloads, list) else []:
            name = receipt.get("filename")
            if isinstance(name, str) and Path(name).name == name and receipt.get("status") == 200:
                bundle.copy(
                    folder / "artifacts" / name,
                    prefix + "/artifacts/" + name,
                    expected=receipt.get("sha256"),
                )
        for name in PRIOR_FILES:
            if (folder / name).is_file():
                bundle.copy(folder / name, prefix + "/prior-reviews/" + name)
        if case in CASES:
            for kind in ("tasks", "datasets"):
                bundle.tree(
                    source_root / f"benchmark/office/{kind}/{case}",
                    f"reference-source/{kind}/{case}",
                )
            spec = source_root / f"benchmark/office/cases/{case}/spec.json"
            if spec.exists():
                bundle.copy(spec, f"reference-source/cases/{case}/spec.json")
        index.append(
            {
                "case": case,
                "phase": phase,
                "mode": mode,
                "session_id": sid,
                "run_id": rid,
                "directory": prefix,
                "source_directory": str(folder),
                "review_status": "pending",
                "statistics": result.get("statistics"),
            }
        )
    bundle.generated("runs.json", index, "Explicit command-line selection")
    bundle.generated(
        "README.md",
        """# AI成功率复核材料

1. 先读 `AI_REVIEW_PROMPT.md`，以 `runs.json` 为本次明确选择的样本清单；H、Q分别核验。
2. 先看 `issues.json`。缺失、损坏或身份冲突的材料需补齐，无法补齐的项目按提示词列待核定。
3. 每个Run目录中，`task.md`是实际请求正文；`execution/`保存原条件、轨迹和回执；`audit/`保存冻结参考与快照。
4. `audit/blobs/`保留原字节；`audit/versions/<SHA256>/`按原逻辑文件名保存同一版本，便于直接读取图片、正文和PPT。快照中的文件引用与这里的SHA256对应。
5. `inputs/`保存能按上传回执或冻结spec核对的输入，`artifacts/`保存runner下载产物。邮件已发布正文也可从audit的版本目录读取。
6. `reference-source/`为采集时提供的源码参考；原始任务以task.md为准，案例和输入以Run冻结参考及哈希为准。发现版本冲突时保留待核定。
7. 先完成独立判定，再对照可选的 `prior-reviews/` 原评分；最终结果另存 `reviews/`。

`files.json`提供包内相对路径、原来源、大小和SHA256。`derived`表示从已保存字段重新排版，原JSON同时保留。完整性标志只表示材料检查情况，任务质量由AI结合证据判定。
""",
        "Collector instructions",
    )
    bundle.generated("issues.json", bundle.issues, "Collection integrity checks")
    result = {
        "schema_version": 1,
        "complete": not bundle.issues,
        "runs": index,
        "issues": bundle.issues,
        "review_status": "pending",
    }
    bundle.generated("manifest.json", result, "Collection summary")
    (output / "files.json").write_text(
        json.dumps(list(bundle.files.values()), ensure_ascii=False, indent=2) + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--runs", nargs="+", type=Path, help="Explicit runner output directories"
    )
    selection.add_argument(
        "--run-list", type=Path, help="JSON list with directory and optional audit_directory"
    )
    parser.add_argument(
        "--chat-data", type=Path, help="Chat data root, containing benchmark/ and inputs/"
    )
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.run_list:
        rows = json.loads(args.run_list.read_text())
        for row in rows:
            for key in ("directory", "audit_directory"):
                if key in row and not Path(row[key]).is_absolute():
                    row[key] = args.run_list.parent / row[key]
    else:
        rows = [{"directory": p} for p in args.runs]
    result = collect(rows, args.output, chat_data=args.chat_data, source_root=args.source_root)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "runs": len(result["runs"]),
                "complete": result["complete"],
                "issues": len(result["issues"]),
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
