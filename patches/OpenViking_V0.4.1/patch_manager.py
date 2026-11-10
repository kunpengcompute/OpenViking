#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Audit, generate, verify and materialize pinned OpenViking patch bundles."""

from __future__ import annotations

import argparse
import contextlib
import fnmatch
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
METADATA = (
    "manifest.json",
    "patch_manager.py",
    "prepare_source.py",
    "README.md",
    "LICENSE",
    "NOTICE",
)


class PatchError(RuntimeError):
    """An input, integrity or source-reconstruction check failed."""


def git(repo: Path, *args: str) -> bytes:
    command = [
        "git",
        "--literal-pathspecs",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "commit.gpgsign=false",
        "-C",
        str(repo),
        *args,
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode:
        raise PatchError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def object_id(repo: Path, ref: str) -> str:
    # --end-of-options prevents a supplied ref from becoming a Git option.
    return (
        git(repo, "rev-parse", "--verify", "--end-of-options", ref + "^{commit}").decode().strip()
    )


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_file(folder: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if not relative or path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise PatchError(f"Unsafe bundle path: {relative!r}")
    result = folder / relative
    if not result.resolve().is_relative_to(folder.resolve()):
        raise PatchError(f"Bundle path escapes its directory: {relative}")
    for parent in [result, *result.parents]:
        if parent == folder:
            break
        if parent.is_symlink():
            raise PatchError(f"Bundle paths must be regular files: {relative}")
    return result


def load_manifest(folder: Path) -> dict:
    value = json.loads(safe_file(folder, "manifest.json").read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or value.get("status") not in {
        "template",
        "release",
    }:
        raise PatchError("Unsupported manifest schema or status")
    if not re.fullmatch(r"[0-9a-f]{40}", value.get("base_ref", "")):
        raise PatchError("base_ref must be a full immutable commit ID")
    if not value.get("base_repository") or not value.get("base_tag"):
        raise PatchError("The upstream repository and tag are required")
    excluded = value.get("excluded_paths", [])
    if not isinstance(excluded, list) or any(not isinstance(p, str) for p in excluded):
        raise PatchError("excluded_paths must be a list of path patterns")
    groups = value.get("groups", [])
    names, outputs = [], []
    for group in groups:
        name = group["name"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise PatchError(f"Invalid patch group: {name}")
        safe_file(folder, group["output"])
        if not group["output"].startswith("feature/") or not group["output"].endswith(".patch"):
            raise PatchError("Group output must be feature/*.patch")
        if (
            not isinstance(group.get("patterns"), list)
            or not group["patterns"]
            or any(not isinstance(p, str) for p in group["patterns"])
        ):
            raise PatchError(f"Group {name} requires path patterns")
        if not isinstance(group.get("exclude", []), list) or any(
            not isinstance(p, str) for p in group.get("exclude", [])
        ):
            raise PatchError(f"Group {name} has invalid exclusion patterns")
        names.append(name)
        outputs.append(group["output"])
    if not names or len(set(names)) != len(names) or len(set(outputs)) != len(outputs):
        raise PatchError("Patch groups and outputs must be nonempty and unique")
    profiles = value.get("profiles", {})
    for name, selected in profiles.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or not isinstance(selected, list):
            raise PatchError("Invalid profile definition")
        if not selected or len(set(selected)) != len(selected) or set(selected) - set(names):
            raise PatchError(f"Profile {name} has unknown or duplicate patch groups")
        if selected != [group for group in names if group in selected]:
            raise PatchError(f"Profile {name} must follow manifest group order")
    if not any(set(selected) == set(names) for selected in profiles.values()):
        raise PatchError("At least one profile must contain every patch group")
    if value["status"] == "release":
        for key in ("source_commit", "source_tree"):
            if not re.fullmatch(r"[0-9a-f]{40}", value.get(key, "")):
                raise PatchError(f"Release requires {key}")
    return value


def selected_groups(manifest: dict, profile: str) -> list[dict]:
    if profile not in manifest["profiles"]:
        raise PatchError(f"Unknown feature set: {profile}")
    names = manifest["profiles"][profile]
    return [g for g in manifest["groups"] if g["name"] in names]


def bundle_files(folder: Path, manifest: dict) -> list[str]:
    files = list(METADATA)
    if manifest["status"] == "release":
        files += ["SOURCE_COMMIT"] + [g["output"] for g in manifest["groups"]]
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*.patch")}
    expected = {name for name in files if name.endswith(".patch")}
    if actual != expected:
        raise PatchError(f"Patch inventory mismatch: {sorted(actual ^ expected)}")
    return sorted(files)


def checksums(folder: Path) -> None:
    manifest = load_manifest(folder)
    lines = [f"{sha(safe_file(folder, name))}  {name}\n" for name in bundle_files(folder, manifest)]
    safe_file(folder, "SHA256SUMS").write_text("".join(lines), encoding="utf-8")


def verify_integrity(folder: Path) -> dict:
    manifest = load_manifest(folder)
    expected = set(bundle_files(folder, manifest))
    recorded = {}
    for line in safe_file(folder, "SHA256SUMS").read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match or match[2] in recorded:
            raise PatchError("Invalid or duplicate checksum entry")
        recorded[match[2]] = match[1]
    if set(recorded) != expected:
        raise PatchError("SHA256SUMS must cover exactly the bundle metadata and patch files")
    for name, digest in recorded.items():
        if sha(safe_file(folder, name)) != digest:
            raise PatchError(f"Checksum mismatch: {name}")
    if manifest["status"] == "release" and (
        safe_file(folder, "SOURCE_COMMIT").read_text().strip() != manifest["source_commit"]
    ):
        raise PatchError("SOURCE_COMMIT differs from the manifest")
    return manifest


def audit(manifest: dict, repo: Path, source_ref: str) -> tuple[str, dict[str, list[str]]]:
    if git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise PatchError("Commit or remove source working-tree changes before exporting")
    base = object_id(repo, manifest["base_ref"])
    source = object_id(repo, source_ref)
    raw = git(repo, "diff", "--name-only", "--no-renames", "-z", base, source, "--")
    ownership = {g["name"]: [] for g in manifest["groups"]}
    for path in raw.decode("utf-8").split("\0"):
        if not path or any(
            fnmatch.fnmatchcase(path, p) for p in manifest.get("excluded_paths", [])
        ):
            continue
        owners = [
            g["name"]
            for g in manifest["groups"]
            if any(fnmatch.fnmatchcase(path, p) for p in g["patterns"])
            and not any(fnmatch.fnmatchcase(path, p) for p in g.get("exclude", []))
        ]
        if len(owners) != 1:
            raise PatchError(f"Changed path must have exactly one owner: {path} -> {owners}")
        ownership[owners[0]].append(path)
    return source, ownership


@contextlib.contextmanager
def temporary_checkout(repo: Path, base: str):
    with tempfile.TemporaryDirectory(prefix="ov-patch-verify-") as temp:
        checkout = Path(temp) / "source"
        git(repo, "worktree", "add", "--detach", str(checkout), base)
        try:
            yield checkout
        finally:
            git(repo, "worktree", "remove", "--force", str(checkout))


def apply_profile(folder: Path, manifest: dict, profile: str, checkout: Path) -> str:
    groups = selected_groups(manifest, profile)
    for group in groups:
        patch = safe_file(folder, group["output"])
        if patch.stat().st_size:
            print(f"[{profile}] git apply --index --check {group['output']}", flush=True)
            git(checkout, "apply", "--index", "--check", str(patch))
            print(f"[{profile}] git apply --index {group['output']}", flush=True)
            git(checkout, "apply", "--index", str(patch))
    tree = git(checkout, "write-tree").decode().strip()
    if (
        len(groups) == len(manifest["groups"])
        and manifest.get("source_tree")
        and tree != manifest["source_tree"]
    ):
        raise PatchError("Reconstructed full source tree differs from the recorded source tree")
    print(f"[{profile}] source tree: {tree}", flush=True)
    return tree


def verify(folder: Path, repo: Path | None = None) -> dict:
    manifest = verify_integrity(folder)
    if repo is not None:
        if manifest["status"] != "release":
            raise PatchError(
                "Template contains no released feature patches; generate a bundle first"
            )
        object_id(repo, manifest["base_ref"])
        for profile in manifest["profiles"]:
            with temporary_checkout(repo, manifest["base_ref"]) as checkout:
                apply_profile(folder, manifest, profile, checkout)
    return manifest


def publish_directory(temp: Path, output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise PatchError(f"Output already exists: {output}")
    temp.rename(output)


def generate(folder: Path, repo: Path, source_ref: str, output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise PatchError(f"Output already exists: {output}")
    manifest = load_manifest(folder)
    if output.resolve().is_relative_to(repo.resolve()):
        raise PatchError("Keep generated bundles outside the source working tree")
    source, ownership = audit(manifest, repo, source_ref)
    if not any(ownership.values()):
        raise PatchError("Source has no changes against the pinned baseline")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ov-bundle-", dir=output.parent) as temp:
        bundle = Path(temp) / "bundle"
        bundle.mkdir()
        for name in METADATA:
            shutil.copyfile(safe_file(folder, name), bundle / name)
        manifest.update(
            status="release",
            source_commit=source,
            development_tree=git(repo, "rev-parse", source + "^{tree}").decode().strip(),
        )
        manifest.pop("source_tree", None)
        (bundle / "SOURCE_COMMIT").write_text(source + "\n")
        for group in manifest["groups"]:
            patch = safe_file(bundle, group["output"])
            patch.parent.mkdir(parents=True, exist_ok=True)
            paths = ownership[group["name"]]
            data = (
                git(
                    repo,
                    "diff",
                    "--binary",
                    "--full-index",
                    "--no-ext-diff",
                    "--no-textconv",
                    "--no-renames",
                    manifest["base_ref"],
                    source,
                    "--",
                    *paths,
                )
                if paths
                else b""
            )
            patch.write_bytes(data)
        full_profile = next(
            name
            for name, groups in manifest["profiles"].items()
            if len(groups) == len(manifest["groups"])
        )
        with temporary_checkout(repo, manifest["base_ref"]) as checkout:
            manifest["source_tree"] = apply_profile(bundle, manifest, full_profile, checkout)
        (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        checksums(bundle)
        verify(bundle, repo)
        publish_directory(bundle, output)
    print(f"Generated and verified patch bundle: {output}")


def materialize(folder: Path, repo: Path | None, profile: str, output: Path) -> None:
    manifest = verify_integrity(folder)
    if manifest["status"] != "release":
        raise PatchError("Template contains no released feature patches; generate a bundle first")
    selected_groups(manifest, profile)
    if output.exists() or output.is_symlink():
        raise PatchError(f"Output already exists: {output}")
    if repo is not None:
        if output.resolve().is_relative_to(repo.resolve()):
            raise PatchError("Keep materialized sources outside the development working tree")
        object_id(repo, manifest["base_ref"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ov-source-", dir=output.parent) as temp:
        checkout = Path(temp) / "source"
        if repo is None:
            git(
                output.parent,
                "clone",
                "--no-checkout",
                "--depth",
                "1",
                "--branch",
                manifest["base_tag"],
                "--",
                manifest["base_repository"],
                str(checkout),
            )
            if object_id(checkout, "HEAD") != manifest["base_ref"]:
                raise PatchError("Upstream tag moved away from the pinned baseline")
        else:
            git(
                output.parent,
                "clone",
                "--no-checkout",
                "--no-hardlinks",
                "--dissociate",
                "--",
                str(repo),
                str(checkout),
            )
        git(checkout, "checkout", "--detach", manifest["base_ref"])
        tree = apply_profile(folder, manifest, profile, checkout)
        receipt = {
            "base_ref": manifest["base_ref"],
            "source_commit": manifest["source_commit"],
            "source_tree": tree,
            "feature_set": profile,
            "manifest_sha256": sha(folder / "manifest.json"),
        }
        (checkout / ".git" / "openviking-patches.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        publish_directory(checkout, output)
    print(f"Created OpenViking source: {output} ({profile}); patch changes are staged")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-dir", type=Path, default=HERE)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "generate"):
        command = commands.add_parser(name)
        command.add_argument("--source-tree", type=Path, required=True)
        command.add_argument("--source-ref", default="HEAD")
        if name == "generate":
            command.add_argument("--output-dir", type=Path, required=True)
    command = commands.add_parser("verify")
    command.add_argument("--source-tree", type=Path)
    command = commands.add_parser("materialize")
    command.add_argument("--source-tree", type=Path)
    command.add_argument("--feature-set", default="full-default")
    command.add_argument("--output-dir", type=Path, required=True)
    commands.add_parser("checksums")
    args = parser.parse_args(argv)
    try:
        folder = args.patch_dir.expanduser().resolve()
        repo = (
            args.source_tree.expanduser().resolve() if getattr(args, "source_tree", None) else None
        )
        if args.command == "audit":
            _, owned = audit(load_manifest(folder), repo, args.source_ref)
            print(json.dumps({name: len(paths) for name, paths in owned.items()}, indent=2))
        elif args.command == "generate":
            generate(folder, repo, args.source_ref, args.output_dir.expanduser().absolute())
        elif args.command == "verify":
            manifest = verify(folder, repo)
            print(
                f"Verified {manifest['status']} integrity"
                + (" and all source profiles" if repo else "")
            )
        elif args.command == "checksums":
            checksums(folder)
            print("Updated SHA256SUMS")
        else:
            materialize(folder, repo, args.feature_set, args.output_dir.expanduser().absolute())
    except (OSError, ValueError, KeyError, TypeError, PatchError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
