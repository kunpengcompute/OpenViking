# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""Middleware-owned recall retrieval, ranking, content selection and budgeting."""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

from openviking.core.namespace import canonical_user_root, canonicalize_uri, classify_uri
from openviking.core.retrieval_targets import default_target_directories, resolve_retrieval_targets
from openviking.utils.token_estimation import text_token_weight
from openviking_cli._exception_utils import auth_error
from openviking_cli.client.identity import OVIdentity, _identity_arguments
from openviking_cli.exceptions import InvalidArgumentError
from openviking_cli.retrieve.context import (
    MEMORY_CATEGORIES,
    QUOTA_CATEGORIES,
    ContextSearchOptions,
    ContextSearchResult,
    RecallPolicy,
)
from openviking_cli.retrieve.types import ContextType, MatchedContext
from openviking_cli.session.user_id import UserIdentifier

from .recall_content import has_information, memory_features, memory_view
from .recall_content import overview as _overview
from .recall_dedup import RecallDeduplicator
from .recall_ranking import rank_candidates
from .recall_selection import method_capabilities, query_purpose

READ_CONCURRENCY = 8
SEARCH_CONCURRENCY = 8


@dataclass(frozen=True)
class _RecallScope:
    """URI-resolution data derived from OVIdentity; OV authenticates every SDK request."""

    user: UserIdentifier
    actor_peer_id: str | None
    legacy_agent_id: str | None = None
    role: str = "user"


def _path_scopes(roots):
    return {
        "op": "or",
        "conds": [
            {"op": "must", "field": "uri", "conds": [root], "para": "-d=-1"} for root in roots
        ],
    }


class _CandidateTransport:
    def __init__(self, client, identity):
        self.client, self.identity = client, identity
        self.requests = 0

    def arguments(self, value):
        return {key: item for key, item in value.items() if item is not None}

    async def find(self, request):
        self.requests += 1
        return await self.client.find(
            **self.arguments(request), **_identity_arguments(self.identity)
        )

    async def read(self, uri):
        return await self.client.read(uri, **_identity_arguments(self.identity))


@dataclass
class Candidate:
    item: MatchedContext
    category: str
    experience: bool


def _base(uri: str) -> str:
    return re.sub(r"/\.(?:abstract|overview)\.md$", "", uri.rstrip("/"))


def _within(uri: str, root: str) -> bool:
    return uri == root or uri.startswith(root.rstrip("/") + "/")


def _event_replica_scope(candidate: Candidate, ctx: Any) -> tuple[str, str] | None:
    """Identify a possible user/current-peer mirror by its dated event path.

    Paths only select files to compare. The complete stored bodies must also
    match before either copy can be suppressed.
    """
    if candidate.category != "events" or candidate.item.level != 2 or not ctx.actor_peer_id:
        return None
    root = canonical_user_root(ctx)
    uri = _base(canonicalize_uri(candidate.item.uri, ctx))
    for scope, prefix in (
        ("user", f"{root}/memories/events/"),
        ("peer", f"{root}/peers/{ctx.actor_peer_id}/memories/events/"),
    ):
        if uri.startswith(prefix):
            relative = uri[len(prefix) :]
            if re.fullmatch(r"\d{4}/\d{2}/\d{2}/[^/]+\.md", relative):
                return scope, relative
    return None


def _category(item: MatchedContext) -> str:
    if item.context_type == ContextType.MEMORY:
        path = classify_uri(item.uri)
        index = path.content_index
        name = path.parts[index + 1] if index is not None and len(path.parts) > index + 1 else ""
        return name if name in MEMORY_CATEGORIES else "memories"
    return "skills" if item.context_type == ContextType.SKILL else "resources"


def _preset(item: MatchedContext) -> bool:
    if item.level == 2:
        return False
    uri = _base(item.uri)
    return bool(
        re.fullmatch(
            r"viking://(?:resources|agent(?:/[^/]+)?|user/[^/]+"
            r"(?:/peers/[^/]+)?(?:/(?:memories|resources|skills))?)",
            uri,
        )
    )


def _category_roots(category: str, ctx: Any) -> list[str]:
    """Own each quota bucket with paths supported by this branch's namespace."""
    if category not in QUOTA_CATEGORIES:
        raise InvalidArgumentError(f"Unknown context quota category: {category}")
    if category in MEMORY_CATEGORIES:
        roots = [f"{canonical_user_root(ctx)}/memories"]
        if ctx.actor_peer_id:
            roots.append(f"{canonical_user_root(ctx)}/peers/{ctx.actor_peer_id}/memories")
        if ctx.legacy_agent_id:
            roots.append(f"viking://agent/{ctx.legacy_agent_id}/memories")
        return [f"{root}/{category}" for root in roots]
    context_type = ContextType.SKILL if category == "skills" else ContextType.RESOURCE
    return default_target_directories(ctx, context_type=context_type) or (
        [f"{canonical_user_root(ctx)}/skills"]
        if category == "skills"
        else ["viking://resources", f"{canonical_user_root(ctx)}/resources"]
    )


def _intersect_roots(roots: list[str], scopes: list[str] | None) -> list[str]:
    """Intersect subtrees before retrieval, retaining explicit empty scopes."""
    if scopes is None:
        return list(dict.fromkeys(roots))
    intersected = []
    for root in roots:
        for scope in scopes:
            if _within(root, scope):
                intersected.append(root)
            elif _within(scope, root):
                intersected.append(scope)
    return list(dict.fromkeys(intersected))


def _render_entry(entry: dict[str, Any]) -> str:
    def attr(value: str) -> str:
        return value.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")

    # Escape the container's tags so untrusted memory text cannot forge entry
    # boundaries. Preserve other markup for Markdown/code fidelity. This is
    # not a prompt-injection sanitizer: consuming prompts must treat recalled
    # text as untrusted data, including any instructions it contains.
    text = re.sub(r"<(/?)memory(?=[\s/>])", r"<\\\1memory", entry["text"], flags=re.I)
    return (
        f'<memory uri="{attr(entry["uri"])}" type="{attr(entry["category"])}"'
        f' score="{entry["score"]:.2f}" detail="{entry["detail"]}">\n{text}\n</memory>'
    )


def _render(entries: list[dict[str, Any]]) -> str:
    return "\n".join(_render_entry(entry) for entry in entries)


@dataclass
class _ContextBudget:
    """Accumulate actual rendered fragments without rescanning accepted text."""

    max_tokens: int
    max_chars: int | None = None
    experience_max_chars: int | None = None
    used_chars: int = 0
    _parts: list[str] = field(default_factory=list)
    _weight: float = 0.0
    _experience_chars: int = 0
    _separator_weight: float = field(default_factory=lambda: text_token_weight("\n"))

    @property
    def used_tokens(self) -> int:
        # Weights are multiples of 0.25, exactly representable as floats. Round
        # only the combined weight, not individual entries or separators.
        return math.ceil(self._weight)

    def try_add(self, entry: dict[str, Any]) -> bool:
        fragment = _render_entry(entry)
        chars = self.used_chars + len(fragment) + bool(self._parts)
        experience = entry["category"] == "experiences"
        experience_chars = self._experience_chars
        if experience:
            experience_chars += len(fragment) + bool(self._experience_chars)
        if (self.max_chars is not None and chars > self.max_chars) or (
            experience
            and self.experience_max_chars is not None
            and experience_chars > self.experience_max_chars
        ):
            return False
        weight = self._weight + text_token_weight(fragment)
        if self._parts:
            weight += self._separator_weight
        if math.ceil(weight) > self.max_tokens:
            return False
        self._parts.append(fragment)
        self.used_chars, self._weight = chars, weight
        self._experience_chars = experience_chars
        return True

    def render(self) -> str:
        return "\n".join(self._parts)


async def _assemble_context(
    *,
    transport: _CandidateTransport,
    query: str,
    ctx: Any,
    options: ContextSearchOptions,
    target_uri: str | list[str] = "",
    limit: int = 10,
    score_threshold: float | None = None,
    filter: Any = None,
    level: list[int] | None = None,
    deduplicator: RecallDeduplicator,
) -> ContextSearchResult:
    """Use the original query for all tasks; no session read or intent analysis."""
    started = time.monotonic()
    preserve_event_identity = query_purpose(query).kind != "execute"
    if not query.strip():
        raise InvalidArgumentError("Search query must not be empty.")
    if options.quotas is None and not 1 <= limit <= 200:
        raise InvalidArgumentError("Context limit must be between 1 and 200")
    if score_threshold is not None and (
        not math.isfinite(score_threshold) or not 0 <= score_threshold <= 1
    ):
        raise InvalidArgumentError("Context score_threshold must be finite and between 0 and 1")
    policy = options.recall_policy
    ranking_policy = policy or RecallPolicy(
        ranking="score",
        prefer_leaf=False,
        score_threshold=score_threshold or 0,
        experience_score_threshold=score_threshold or 0,
    )
    excluded = {_base(canonicalize_uri(uri, ctx)) for uri in options.exclude_uris}
    global_roots = (
        resolve_retrieval_targets(target_uri, ctx).target_directories if target_uri else []
    )
    stats: dict[str, Any] = {
        "query_expansion": "off",
        "planned_queries": [query],
        "retrieval_errors": [],
        "read_errors": [],
        "searches": 0,
        "body_reads": 0,
    }
    quotas = options.quotas
    tasks: list[tuple[str | list[str], bool | None, str | None, int, float | None]] = []
    if quotas is not None:
        # Quotas own category activation and output counts. Reader group switches
        # and legacy count caps apply only to quota-free recall.
        width = sum(quotas.values())
        reader_roots = None
        if policy and policy.target_uris is not None:
            reader_roots = (
                resolve_retrieval_targets(list(policy.target_uris), ctx).target_directories
                if policy.target_uris
                else []
            )
        for category, count in quotas.items():
            if count <= 0:
                continue
            roots = _intersect_roots(_category_roots(category, ctx), reader_roots)
            roots = _intersect_roots(roots, global_roots if target_uri else None)
            experience = category == "experiences"
            candidate_limit = max(
                width,
                (policy.experience_candidate_limit if experience else policy.candidate_limit)
                if policy
                else width,
            )
            threshold = (
                (policy.experience_score_threshold if experience else policy.score_threshold)
                if policy
                else score_threshold
            )
            if roots:
                target = roots[0] if len(roots) == 1 else roots
                tasks.append((target, None, category, candidate_limit, threshold))
    groups = (
        [
            (
                False,
                list(
                    dict.fromkeys(
                        policy.target_uris
                        if policy.target_uris is not None
                        else ("viking://user/memories",)
                    )
                ),
                policy.candidate_limit,
                0.0,
            )
        ]
        if quotas is None
        and policy
        and policy.enabled
        and policy.recall_limit
        and policy.candidate_limit
        else []
    )
    if (
        quotas is None
        and policy
        and policy.experience_enabled
        and policy.experience_limit
        and policy.experience_candidate_limit
        and policy.experience_max_chars
    ):
        groups.append(
            (
                True,
                list(dict.fromkeys(policy.experience_target_uris)),
                policy.experience_candidate_limit,
                policy.experience_score_threshold,
            )
        )
    if quotas is None and not policy:
        roots = target_uri if isinstance(target_uri, list) else [target_uri]
        groups = [(None, roots, limit, score_threshold)]
    for experience, roots, candidate_limit, threshold in groups:
        if roots:
            target = roots[0] if len(roots) == 1 else roots
            tasks.append((target, experience, None, candidate_limit, threshold))

    search_slots = asyncio.Semaphore(SEARCH_CONCURRENCY)

    def search_arguments(task):
        root, experience, category, candidate_limit, threshold = task
        constraints = [filter] if filter else []
        if category:
            context_type = {"resources": "resource", "skills": "skill"}.get(category, "memory")
            constraints.append({"op": "must", "field": "context_type", "conds": [context_type]})
            # Category scopes must reach the vector query, rather than relying
            # on post-top-k classification (including nested same-name paths).
            constraints.append(_path_scopes(_category_roots(category, ctx)))
        if policy and global_roots:
            constraints.append(_path_scopes(global_roots))
        # Keep level selection in the legacy retriever: applying a level-only
        # scope filter here would also hide directories needed for traversal.
        scope = {"op": "and", "conds": constraints} if constraints else None
        return {
            "query": query,
            "target_uri": root,
            "limit": candidate_limit + len(excluded),
            "score_threshold": threshold,
            "filter": scope,
            "level": level,
        }

    async def search_one(task):
        root, experience, category, candidate_limit, threshold = task
        try:
            async with search_slots:
                stats["searches"] += 1
                result = await transport.find(search_arguments(task))
        except Exception as exc:
            if auth_error(exc) or isinstance(exc, (InvalidArgumentError, ValueError)):
                raise
            stats["retrieval_errors"].append(type(exc).__name__)
            return []
        found = []
        for name in ("memories", "resources", "skills"):
            for item in getattr(result, name, []) or []:
                if not item.uri or not math.isfinite(item.score) or _preset(item):
                    continue
                base = _base(canonicalize_uri(item.uri, ctx))
                if base.endswith("/profile.md"):
                    continue
                if any(_within(base, x) for x in excluded):
                    continue
                if global_roots and not any(_within(base, r) for r in global_roots):
                    continue
                cat = _category(item)
                if experience is not None and (cat == "experiences") != experience:
                    continue
                if category is not None and cat != category:
                    continue
                found.append(Candidate(item, cat, cat == "experiences"))
        return found

    retrieval_started = time.monotonic()
    pending = [asyncio.create_task(search_one(task)) for task in tasks]
    try:
        outcomes = await asyncio.gather(*pending)
    finally:
        for task in pending:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    stats["retrieval_ms"] = (time.monotonic() - retrieval_started) * 1000
    candidates = [c for outcome in outcomes for c in outcome]
    # Rank quota buckets independently: a leaf or duplicate summary in another
    # category must not erase a bucket before its quota is allocated.
    ranked: list[Candidate] = []
    selection: list[dict[str, Any]] = []
    for key in quotas if quotas is not None else (True, False):
        experience = key == "experiences" if quotas is not None else key
        group = [
            c
            for c in candidates
            if (c.category == key if quotas is not None else c.experience == experience)
        ]
        by_uri = {c.item.uri: c for c in sorted(group, key=lambda c: c.item.score)}
        ranked.extend(
            by_uri[i.uri]
            for i in rank_candidates(
                [c.item for c in group],
                query,
                ranking_policy,
                experience=experience,
                diagnostics=selection,
            )
        )
    if quotas is not None:
        first, spare = [], []
        for category, count in quotas.items():
            bucket = [c for c in ranked if c.category == category]
            first.extend(bucket[:count])
            spare.extend(bucket[count:] if count else [])
        # Within each reader group, quota ceilings may leave reusable slots.
        ranked = first + sorted(spare, key=lambda c: -c.item.score)
    if policy and quotas is None:
        ranked.sort(key=lambda c: not c.experience)
    replica_scopes = {c.item.uri: _event_replica_scope(c, ctx) for c in ranked}
    path_scopes: dict[str, set[str]] = {}
    for replica in replica_scopes.values():
        if replica is not None:
            scope, path = replica
            path_scopes.setdefault(path, set()).add(scope)
    mirrored_paths = {path for path, scopes in path_scopes.items() if len(scopes) == 2}

    def needs_replica_body(candidate: Candidate) -> bool:
        replica = replica_scopes[candidate.item.uri]
        return replica is not None and replica[1] in mirrored_paths

    def pin(candidate: Candidate) -> str:
        detail = options.detail
        if isinstance(detail, dict):
            selected = detail.get(candidate.category)
        else:
            selected = detail
        chosen = selected or (
            "full"
            if candidate.experience or (policy and not policy.prefer_abstract)
            else "abstract"
        )
        return chosen

    read_slots = asyncio.Semaphore(READ_CONCURRENCY)
    reads: dict[str, asyncio.Task] = {}

    async def read_body(uri):
        try:
            async with read_slots:
                stats["body_reads"] += 1
                raw = await transport.read(uri)
            return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw or "")
        except Exception as exc:
            if auth_error(exc):
                raise
            stats["read_errors"].append(type(exc).__name__)
            return ""

    def content_uri(candidate):
        item = candidate.item
        return _base(item.uri) + "/.overview.md" if item.level in (0, 1) else item.uri

    total_limit = sum(quotas.values()) if quotas is not None else limit

    def group_limit(candidate: Candidate) -> int:
        if policy and quotas is None:
            return policy.experience_limit if candidate.experience else policy.recall_limit
        return total_limit

    # Prefetch only an initial output-sized window. If budgeting rejects an
    # entry, later candidates can be read on demand, without reading the whole
    # overfetched pool up front.
    prefetched = {True: 0, False: 0}
    window = 0
    for c in ranked:
        if prefetched[c.experience] >= group_limit(c):
            continue
        if (not policy or quotas is not None) and window >= total_limit:
            break
        prefetched[c.experience] += 1
        window += 1
        if (
            needs_replica_body(c)
            or pin(c) != "abstract"
            or not (c.item.abstract or c.item.overview or "").strip()
        ):
            uri = content_uri(c)
            if uri not in reads:
                reads[uri] = asyncio.create_task(read_body(uri))
    try:
        if reads:
            await asyncio.gather(*reads.values())
    finally:
        for task in reads.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*reads.values(), return_exceptions=True)

    entries: list[dict[str, Any]] = []
    budget = _ContextBudget(
        options.max_tokens,
        policy.max_injected_chars if policy else None,
        policy.experience_max_chars if policy else None,
    )
    counts = {True: 0, False: 0}
    seen: set[str] = set()
    accepted_replicas: dict[tuple[str, bytes], set[str]] = {}
    stats["event_replicas_suppressed"] = 0
    selection_by_uri = {row["uri"]: row for row in selection}
    for decision in selection:
        if decision.get("status") == "rejected":
            decision["output_status"] = "not_applicable"
    for c in ranked:
        decision = selection_by_uri.get(c.item.uri, {})
        if counts[c.experience] >= group_limit(c) or (not policy and len(entries) >= total_limit):
            decision["output_status"] = "count_limit"
            continue
        if quotas is not None and len(entries) >= sum(quotas.values()):
            decision["output_status"] = "count_limit"
            break
        key = _base(c.item.uri)
        if key in seen:
            decision["output_status"] = "duplicate_uri"
            continue
        chosen = pin(c)
        summary = (c.item.abstract or c.item.overview or "").strip()
        summary_detail = "abstract" if c.item.abstract else "overview"
        uri = content_uri(c)
        if (needs_replica_body(c) or chosen != "abstract" or not summary) and uri not in reads:
            reads[uri] = asyncio.create_task(read_body(uri))
            await reads[uri]
        body = reads[uri].result().strip() if uri in reads else ""
        replica = replica_scopes[c.item.uri]
        replica_key = None
        if body and replica is not None and replica[1] in mirrored_paths:
            scope, path = replica
            replica_key = (path, hashlib.sha256(body.encode("utf-8")).digest())
            if accepted_replicas.get(replica_key, set()) - {scope}:
                stats["event_replicas_suppressed"] += 1
                decision["output_status"] = "mirrored_event"
                continue
        fields = {}
        if c.item.context_type == ContextType.MEMORY:
            fields = memory_view(body)
            body = fields["content"]
            summary = memory_view(summary)["content"]
        body_detail = "overview" if c.item.level in (0, 1) else "full"
        if chosen == "full":
            texts = [(body_detail if body else summary_detail, body or summary)]
        elif chosen == "overview":
            overview = _overview(body, fields)
            texts = [("overview" if overview else summary_detail, overview or summary)]
        else:
            texts = [(summary_detail if summary else body_detail, summary or body)]
        if chosen == "full":
            texts.append(
                ("overview" if body else summary_detail, _overview(body or summary, fields))
            )
        if chosen != "abstract":
            texts.append((summary_detail, summary))
        dedup_scope = c.item.uri if c.category == "events" and preserve_event_identity else ""
        budget_rejected = False
        duplicate_only = False
        for detail, text in texts:
            if not text or (detail != "full" and not has_information(text)):
                continue
            filtered = deduplicator.filter(text, scope=dedup_scope)
            decision.update(
                query_duplicate_blocks=filtered.query_blocks,
                memory_duplicate_blocks=filtered.memory_blocks,
                content_chars_before=len(text),
                content_chars_after=len(filtered.text),
            )
            if not filtered.text:
                duplicate_only = duplicate_only or bool(
                    filtered.query_blocks or filtered.memory_blocks
                )
                if detail == "full" and body and duplicate_only:
                    # A fully covered body has no new information. Its generated
                    # excerpt must not take a slot merely by using different words.
                    decision["output_status"] = "duplicate_content"
                    break
                continue
            entry = {
                "uri": c.item.uri,
                "category": c.category,
                "score": c.item.score,
                "detail": detail,
                "text": filtered.text,
            }
            if not budget.try_add(entry):
                budget_rejected = True
                continue
            entries.append(entry)
            deduplicator.remember(filtered.text, scope=dedup_scope)
            decision.update(output_status="selected", actual_detail=detail)
            counts[c.experience] += 1
            seen.add(key)
            if replica_key is not None and replica is not None:
                accepted_replicas.setdefault(replica_key, set()).add(replica[0])
            break
        else:
            decision["output_status"] = (
                "budget"
                if budget_rejected
                else "duplicate_content"
                if duplicate_only
                else "no_information"
            )
    rendered = budget.render()
    for decision in selection:
        decision.setdefault("output_status", "not_visited_after_limit")
    stats.update(
        candidates=len(candidates),
        returned=len(entries),
        used_tokens=budget.used_tokens,
        used_chars=budget.used_chars,
        quotas=quotas,
        total_ms=(time.monotonic() - started) * 1000,
        selection=selection,
    )
    if quotas is not None:
        stats["ignored"] = ["limit"]
    return {"entries": entries, "rendered": rendered if options.render else "", "stats": stats}


async def assemble_recall(
    client: Any, query: str, identity: OVIdentity, config, options: ContextSearchOptions
):
    ctx = _RecallScope(
        UserIdentifier(identity.account_id, identity.user_id), identity.actor_peer_id
    )
    transport = _CandidateTransport(client, identity)
    result = await _assemble_context(
        transport=transport,
        query=query,
        ctx=ctx,
        options=options,
        deduplicator=RecallDeduplicator(
            query, enabled=query_purpose(query).kind not in {"history", "mixed"}
        ),
    )
    result["stats"]["candidate_sdk_calls"] = transport.requests
    result["stats"]["structured_method_capabilities"] = sorted(
        set().union(
            *(method_capabilities(memory_features(entry["text"])) for entry in result["entries"])
        )
    )
    return result
