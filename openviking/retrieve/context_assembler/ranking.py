# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""OpenClaw-style candidate ranking and task experience gating."""

from __future__ import annotations

import math
import re
from typing import Protocol
from urllib.parse import unquote

from openviking_cli.retrieve.types import MatchedContext


class RankingConfig(Protocol):
    """Read-only ranking fields shared by reader config and wire recall policy."""

    @property
    def score_threshold(self) -> float: ...

    @property
    def experience_score_threshold(self) -> float: ...

    @property
    def ranking(self) -> str: ...

    @property
    def prefer_leaf(self) -> bool: ...


_PREFERENCE = re.compile(r"prefer|favorite|favourite|like|偏好|喜欢|爱好|更倾向", re.I)
_TEMPORAL = re.compile(
    r"when|what time|date|day|month|year|yesterday|today|tomorrow|last|next|"
    r"什么时候|何时|哪天|几月|几年|昨天|今天|明天|上周|下周|上个月|下个月|去年|明年",
    re.I,
)
_STOPWORDS = set(
    "what when where which who whom whose why how did does is are was were "
    "the and for with from that this your you".split()
)
_TASK_PATTERNS = (
    (
        r"\b(write|edit|modify|delete|remove|migrate|deploy|release|publish|configure|patch)\b|"
        r"写|改|删除|迁移|部署|发布|配置|打补丁",
        3,
    ),
    (
        r"\b(fix|debug|test|build|run|implement|refactor|integrate|repair|troubleshoot)\b|"
        r"修复|调试|测试|构建|运行|实现|重构|对接|排查",
        2,
    ),
    (
        r"\b(error|exception|traceback|failed|failure|retry|exit code)\b|报错|异常|失败|重试|挂了|不通过",
        2,
    ),
    (
        r"\b(repo|workspace|plugin|service|component|hook|api|tool|package|module)\b|"
        r"仓库|工作区|插件|服务|组件|接口|工具|模块|文件|`[^`]+`|"
        r"[\w./-]+\.(?:tsx?|jsx?|py|go|rs|java|md|json|ya?ml|toml|sh|sql)\b",
        2,
    ),
    (r"经验|踩坑|最佳实践|不要再|按之前|avoid|best practice|lesson|pitfall", 1),
)


def is_experience(item: MatchedContext) -> bool:
    """Recognize experiences by either memory category or URI."""
    return item.category.lower() in {"experience", "experiences"} or "/experiences/" in item.uri


def should_recall_experience(query: str, minimum_chars: int) -> bool:
    """Port the task_start gate; cron and retry triggers belong to future hooks."""
    if len(query.strip()) < minimum_chars or "<openviking-context" in query.lower():
        return False
    score = sum(weight for pattern, weight in _TASK_PATTERNS if re.search(pattern, query, re.I))
    if re.search(r"闲聊|翻译|总结当前对话|天气|笑话|hello|hi\b|你好", query, re.I):
        score -= 3
    if re.search(
        r"^(什么是|是什么|区别|解释|讲讲|怎么看|为什么|如何理解|what is|explain)|[?？]$",
        query,
        re.I,
    ):
        if not any(
            re.search(pattern, query, re.I)
            for pattern, _ in _TASK_PATTERNS[1:2] + _TASK_PATTERNS[3:4]
        ):
            score -= 2
    return score >= 3


def rank_candidates(
    items: list[MatchedContext], query: str, config: RankingConfig, *, experience: bool
) -> list[MatchedContext]:
    """Filter by semantic score, then rank and deduplicate without changing candidates."""
    threshold = config.experience_score_threshold if experience else config.score_threshold
    tokens = [t for t in re.findall(r"[a-z0-9]{2,}", query.lower()) if t not in _STOPWORDS]

    def rank(item: MatchedContext) -> float:
        score = min(1.0, max(0.0, item.score))
        if config.ranking == "score" or experience:
            return score
        category = item.category.lower()
        text = f"{item.uri} {item.abstract or item.overview or ''}".lower()
        overlap = sum(t in text for t in tokens[:8])
        return (
            score
            + (0.12 if item.level == 2 else 0)
            + (
                0.1
                if _TEMPORAL.search(query) and (category == "events" or "/events/" in item.uri)
                else 0
            )
            + (
                0.08
                if _PREFERENCE.search(query)
                and (category == "preferences" or "/preferences/" in item.uri)
                else 0
            )
            + (min(0.2, overlap / min(len(tokens), 4) * 0.2) if tokens else 0)
        )

    eligible = [
        item
        for item in items
        if item.uri and math.isfinite(item.score) and min(1.0, max(0.0, item.score)) >= threshold
    ]
    # URI duplicates retain their highest semantic match before query-aware ranking.
    by_uri: dict[str, MatchedContext] = {}
    for item in sorted(eligible, key=lambda item: item.score, reverse=True):
        by_uri.setdefault(item.uri, item)
    ranked = sorted(by_uri.values(), key=rank, reverse=True)
    if config.prefer_leaf and not experience:
        leaves = [item for item in ranked if item.level == 2]
        ranked = leaves or ranked
    result: list[MatchedContext] = []
    seen: set[tuple[str, str]] = set()
    for item in ranked:
        category = item.category.lower()
        abstract = " ".join((item.abstract or item.overview or "").lower().split())
        # Separate events/cases can describe the same text and must keep their identity.
        distinct_event = category in {"events", "cases"} or any(
            segment in item.uri.lower() for segment in ("/events/", "/cases/")
        )
        key = (category, abstract) if abstract and not distinct_event else ("uri", item.uri)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def render_experience(item: MatchedContext, content: str) -> str:
    """Render existing Situation/Approach/Reflect sections without inventing instructions."""
    title = unquote(item.uri.rstrip("/").rsplit("/", 1)[-1]).removesuffix(".md")
    sections = []
    for heading, label in (
        ("Situation", "适用情境"),
        ("Approach", "任务步骤"),
        ("Reflect", "注意事项"),
    ):
        match = re.search(rf"(?:^|\n)##\s+{heading}\s*\n(.*?)(?=\n##\s|\Z)", content, re.I | re.S)
        if match and match.group(1).strip():
            sections.append(f"{label}：\n{match.group(1).strip()}")
    body = "\n\n".join(sections) if sections else content
    return f"### {title}\n来源：{item.uri}\n{body}"
