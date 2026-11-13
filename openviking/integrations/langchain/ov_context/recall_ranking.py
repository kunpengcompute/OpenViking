# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""OpenClaw-style candidate ranking and task experience gating."""

from __future__ import annotations

import math
import re
from typing import Any, Protocol
from urllib.parse import unquote

from openviking_cli.retrieve.types import MatchedContext

from .recall_selection import select_by_value


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


_PREFERENCE = re.compile(r"\b(?:prefer|favorite|favourite|like)\b|偏好|喜欢|爱好|更倾向", re.I)
_TEMPORAL = re.compile(
    r"\b(?:when|what time|date|day|month|year|yesterday|today|tomorrow|last|next)\b|"
    r"什么时候|何时|哪天|几月|几年|昨天|今天|明天|上周|下周|上个月|下个月|去年|明年",
    re.I,
)
_STOPWORDS = set(
    "what when where which who whom whose why how did does is are was were "
    "the and for with from that this your you please help can could would "
    "summary user assistant chatlog guidelines good bad cases static description".split()
)
_CJK_STOPWORDS = {
    "用户",
    "要求",
    "本次",
    "任务",
    "进行",
    "需要",
    "相关",
    "这些",
    "一个",
    "帮我",
    "请帮",
}
_URL = re.compile(r"(?:https?|viking)://[^\s<>\"')]+", re.I)
_FILE = re.compile(r"(?<!\w)[\w./-]+\.[a-zA-Z][a-zA-Z0-9]{0,9}(?!\w)")
_FILE_LOOKUP = re.compile(
    r"查找|搜索|检索|寻找|读取|打开|找到|\b(?:read|open|find|search|locate)\b", re.I
)
_FILE_OUTPUT = re.compile(
    r"保存|生成|输出|导出|命名|写入|\b(?:save|write|create|generate|export|output|name)\b", re.I
)


def _clean_text(text: str) -> str:
    text = re.sub(r"<!--.*?(?:-->|\Z)", " ", text, flags=re.S)
    return _URL.sub(" ", text).lower()


def _terms(text: str, *, numbers: bool = False) -> set[str]:
    """Use content words and Chinese bigrams, never URI scope identifiers."""
    text = _clean_text(text)
    terms = {
        word
        for word in re.findall(r"[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*", text)
        if len(word) > 1
        and word not in _STOPWORDS
        and not (len(word) >= 8 and re.fullmatch(r"[a-f0-9]+", word))
    }
    for run in re.findall(r"[\u3400-\u9fff]+", text):
        terms.update(run[i : i + 2] for i in range(len(run) - 1))
    if numbers:
        terms.update(re.findall(r"\b\d+(?:[.:/-]\d+)*\b", text))
    return terms - _CJK_STOPWORDS


def _ranking_query(query: str) -> str:
    """Output filenames describe deliverables, rather than a memory lookup."""
    query = _clean_text(query)

    def filename(match: re.Match[str]) -> str:
        prefix = query[max(0, match.start() - 48) : match.start()]
        lookup = list(_FILE_LOOKUP.finditer(prefix))
        output = list(_FILE_OUTPUT.finditer(prefix))
        if lookup and (not output or lookup[-1].start() > output[-1].start()):
            return match.group().rsplit("/", 1)[-1]
        suffix = query[match.end() : match.end() + 16]
        return (
            match.group().rsplit("/", 1)[-1]
            if re.match(r"[`\"'\s]*(?:在哪|在哪里|where)", suffix)
            else " "
        )

    return _FILE.sub(filename, query)


def _candidate_terms(item: MatchedContext) -> set[str]:
    # The basename can be a useful title. Account/session/parent paths cannot.
    title = unquote(item.uri.rstrip("/").rsplit("/", 1)[-1]).removesuffix(".md")
    return _terms(f"{title} {item.abstract or item.overview or ''}")


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
    items: list[MatchedContext],
    query: str,
    config: RankingConfig,
    *,
    experience: bool,
    diagnostics: list[dict[str, Any]] | None = None,
) -> list[MatchedContext]:
    """Rank ordinary candidates by explicit applicability and missing method evidence.

    Unknown/mixed requests keep relevance ordering. Explicit score ranking and
    the experience channel retain their semantic-only ordering.
    """
    threshold = config.experience_score_threshold if experience else config.score_threshold
    cleaned_query = _ranking_query(query)
    tokens = _terms(cleaned_query)
    lexical: dict[int, float] = {}

    def rank(item: MatchedContext) -> float:
        score = min(1.0, max(0.0, item.score))
        if config.ranking == "score" or experience:
            return score
        category = item.category.lower()
        lexical[id(item)] = (
            0.2 * len(tokens & _candidate_terms(item)) / len(tokens) if tokens else 0.0
        )
        weighted_score = (
            score
            + (0.12 if item.level == 2 else 0)
            + (
                0.1
                if _TEMPORAL.search(cleaned_query)
                and (category == "events" or "/events/" in item.uri)
                else 0
            )
            + (
                0.08
                if _PREFERENCE.search(cleaned_query)
                and (category == "preferences" or "/preferences/" in item.uri)
                else 0
            )
            + lexical[id(item)]
        )
        # Keep relevance ordering within each category, without promoting a
        # barely relevant method over a strongly relevant historical fact.
        return weighted_score

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
    # Equal excerpts can hide different implementations or facts. Deduplicate
    # actual content after reading, rather than dropping a distinct URI here.
    result = ranked
    if config.ranking == "score" or experience:
        if diagnostics is not None:
            diagnostics.extend(
                {"uri": item.uri, "reason": "semantic_only", "status": "ranked"} for item in result
            )
        return result

    base_scores = {id(item): rank(item) for item in result}
    start = len(diagnostics) if diagnostics is not None else 0
    ordered = select_by_value(
        result,
        query,
        base_scores=base_scores,
        terms=_terms,
        query_terms=lambda text: _terms(_ranking_query(text)),
        diagnostics=diagnostics,
    )
    if diagnostics is not None:
        by_uri = {item.uri: item for item in result}
        for row in diagnostics[start:]:
            item = by_uri[row["uri"]]
            row.update(
                semantic_score=item.score,
                lexical_bonus=lexical[id(item)],
                base_score=base_scores[id(item)],
            )
    return ordered


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
