# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""NLP anchors, source records and rendering rules for deterministic extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from openviking_cli.utils import get_logger

logger = get_logger(__name__)
# ----------------------------------------------------------------------------
# N3: bilingual anchor regex registry (dates / quantities)
# Chinese forms are the original registry; English forms follow semantica's
# time_patterns / extract_entities_regex style (methods.py:688 / event_detector.py:131).
# ----------------------------------------------------------------------------
_ABS_DATE_RE = re.compile(r"(?<!\d)(\d{4})\s*[-/年]\s*(\d{1,2})\s*[-/月]\s*(\d{1,2})\s*日?")
_MD_DATE_RE = re.compile(r"(?<!\d)(\d{1,2})\s*月\s*(\d{1,2})\s*[号日]")
_REL_DAY_RE = re.compile(
    r"(大前天|前天|昨天|今天|明天|后天)"
    r"|((?:the\s+)?(?:day\s+before\s+yesterday|day\s+after\s+tomorrow))"
    r"|(yesterday|today|tomorrow)",
    re.IGNORECASE,
)
_EN_MONTHS = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?"
    r"|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
# English absolute dates: "May 7 2023" / "7 May 2023" / "May 2023" / "in June 2023".
_EN_ABS_DATE_RE = re.compile(
    r"(?<!\w)(?:on\s+|in\s+)?(" + _EN_MONTHS + r")\s+(\d{1,2}),?\s+(\d{4})"
    r"|(?<!\w)(\d{1,2})\s+(" + _EN_MONTHS + r")\s*,?\s+(\d{4})"
    r"|(?<!\w)(?:in\s+|of\s+)?(" + _EN_MONTHS + r")\s+(\d{4})"
)
_WEEK_RE = re.compile(
    r"(?:(上周|这周|本周|下周)([一二三四五六日天]))"
    r"|(?:(周|星期)([一二三四五六日天]))"
    r"|(?:(last|this|next)\s+week\b)"
    r"|(?:(last|this|next)?\s*(monday|tuesday|wednesday|thursday|friday|saturday|sunday))",
    re.IGNORECASE,
)
_N_DAY_RE = re.compile(
    r"(?<!\d)(\d+)\s*天(前|后)"
    r"|(?<!\d)(\d+)\s+(days?|weeks?|months?|years?)\s+(ago|later)"
    r"|(?<!\d)(?:in\s+)(\d+)\s+(days?|weeks?|months?)(?:\s+time)?\b",
    re.IGNORECASE,
)
_N_DAY_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}
_QUANTITY_RE = re.compile(
    r"(?<!\w)(\d+(?:\.\d+)?)\s*(?:岁|块|元|万|千|个|位|人|次|年|月|天|周|公里|千米|小时|分钟)"
    r"|(?<!\w)(\d+(?:\.\d+)?)\s*(?:years?|months?|weeks?|days?|people|million|thousand)\b"
)
_PERCENT_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*%")
_MONEY_RE = re.compile(r"\b(\$[\d,]+(?:\.\d{2})?)\b")
_REL_YEAR_RE = re.compile(r"(last|this|next)\s+year", re.IGNORECASE)
# Code-domain fallback (mirrors OpenViking _WM_PATH_LIKE_RE / _CODE_PATTERNS).
_CODE_PATH_RE = re.compile(
    r"(?:[\w./\\-]+\.(?:py|ts|tsx|js|jsx|md|yaml|yml|json|sh|ps1|cmd|bat|toml|ini|cfg|rs|go))"
    r"|(?:[a-zA-Z_][\w\-]*(?:/[a-zA-Z_][\w\-]*){1,})"
)
_CODE_SYMBOL_RE = re.compile(
    r"\b(?:from\s+\S+\s+import\s+\S+|import\s+\S+)"
    r"|(?:[a-zA-Z_]\w*\s*(?:\(|::|\.\w+\s*=\s*))"
)
_CODE_VERSION_RE = re.compile(r"(?<![.\d])\d+\.\d+(?:\.\d+)?[a-zA-Z0-9._-]*")
_CODE_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_CODE_URL_RE = re.compile(r"https?://[^\s\"'<>]+")
# ----------------------------------------------------------------------------
# N5: title terms / N7: relation lexicon
# ----------------------------------------------------------------------------
_TITLE_TERMS = ("总", "经理", "老师", "医生", "老板", "先生", "女士", "教授", "同学", "工")
_TITLE_RE = re.compile(
    r"([\u4e00-\u9fa5]{1,3})(?:%s)|((?:Mr|Ms|Mrs|Dr|Prof)\.?\s+[A-Z][a-z]+)"
    % "|".join(_TITLE_TERMS)
)
# ----------------------------------------------------------------------------
# N8: Chinese + English event verb dictionary
# (Chinese hand-written; English forms follow semantica event_patterns style,
# event_detector.py:116-122: \b-word-boundary + tense variants)
# ----------------------------------------------------------------------------
_EVENT_VERB_PATTERNS: Dict[str, List[re.Pattern]] = {
    "travel": [
        re.compile(
            r"(?:去|到|飞)(?:了|过|往)?\s*(?P<place>[\u4e00-\u9fa5]{2,6})\s*(?:旅游|出差|旅行)"
        ),
        re.compile(r"(?:计划|打算|约好|准备)\s*去\s*(?P<place>[\u4e00-\u9fa5]{2,6})"),
        re.compile(r"(?:出差|旅游|旅行)(?:去|到|前往)?\s*(?P<place>[\u4e00-\u9fa5]{2,6})"),
        re.compile(
            r"\b(?:travel(?:ed|led|ing)?|went|go|fly|flew|flying|trip|vacation|camping|camped|hiking)\b"
        ),
        re.compile(
            r"\b(?:go|went|heading|planning|thinking)\s+(?:on\s+|for\s+)?(?:a\s+|the\s+)?(?P<place>[A-Za-z]+)\s+(?:trip|vacation|holiday)\b"
        ),
    ],
    "birthday_celebration": [
        re.compile(r"(过|庆祝|庆祝一下|办)?\s*生日"),
        re.compile(r"\b(?:birthday)\b"),
    ],
    "meeting": [
        re.compile(r"(开\w*会|见面|聚餐|聚会|约了|约好|碰头|会面)"),
        re.compile(
            r"\b(?:meet|meeting|met\s+up|meet\s+up|get\s+together|got\s+together|gathering|hang\s+out)\b"
        ),
    ],
    "move": [
        re.compile(r"(搬家|搬去|搬到|入住|退租)"),
        # Strong verb forms (moved/moving/relocate) or "move to/into..." with a
        # complement; bare "move" (big move / next move) is too noisy.
        re.compile(r"\b(?:moved|moving|relocat\w*)\b"),
        re.compile(r"\bmove\s+(?:to|into|out|back|houses?|apartments?|homes?)\b"),
    ],
    "job_change": [
        re.compile(r"(换工作|入职|辞职|离职|升职|晋升|面试|跳槽)"),
        re.compile(
            r"\b(?:job\s+change|joined?|quit|quitting|resign\w*|promot\w*|interview\w*|hired?|started?\s+(?:a\s+)?new\s+job|switch\w*\s+jobs?)\b"
        ),
    ],
    "commitment": [
        re.compile(r"(答应|承诺|保证|决定了|决定|确认了|确认|定了|约定)"),
        re.compile(
            r"\b(?:promise\w*|commit\w*|decid\w*|confirm\w*|agreed?\s+(?:to\s+)?|plan(?:ned|s|ning)?\s+to)\b"
        ),
    ],
    "health": [
        re.compile(r"(生病|住院|感冒|发烧|手术|体检|不舒服)"),
        re.compile(
            r"\b(?:sick|hospital|flu|cold|surgery|surger\w*|checkup|not\s+feeling\s+well)\b"
        ),
    ],
}
_EVENT_GOALS = {
    "travel": "出行安排",
    "birthday_celebration": "生日庆祝",
    "meeting": "会议安排",
    "move": "搬家安排",
    "job_change": "工作变动",
    "commitment": "承诺与决定",
    "health": "健康事件",
}
_EVENT_NAME_SUFFIX = {
    "travel": "出行",
    "birthday_celebration": "生日",
    "meeting": "会面",
    "move": "搬家",
    "job_change": "工作变动",
    "commitment": "确认",
    "health": "事件",
}
# NER label -> entity category (unmapped labels fall back to lowercase label).
_LABEL_CATEGORY = {
    "PERSON": "person",
    "ORG": "organization",
    "GPE": "place",
    "LOC": "place",
    "FAC": "place",
    "NORP": "group",
    "PRODUCT": "product",
    "WORK_OF_ART": "work",
    "LAW": "law",
    "LANGUAGE": "language",
    "EVENT": "event",
}
_ENTITY_LABELS = frozenset(_LABEL_CATEGORY)
_DATE_TOKEN_RES = (
    _ABS_DATE_RE,
    _MD_DATE_RE,
    _EN_ABS_DATE_RE,
    _REL_DAY_RE,
    _WEEK_RE,
    _N_DAY_RE,
    _REL_YEAR_RE,
)


# ----------------------------------------------------------------------------
# Internal records
# ----------------------------------------------------------------------------
@dataclass
class _ExtractedMsg:
    index: int
    role: str
    ts: Optional[datetime]
    text: str
    source_id: str = ""
    speaker: str = ""
    indices: List[int] = field(default_factory=list)


@dataclass
class _EntityHit:
    text: str
    category: str
    msg_index: int
    signals: Set[str] = field(default_factory=set)


@dataclass
class _EntityRecord:
    canonical: str
    category: str
    aliases: Set[str] = field(default_factory=set)
    signal_types: Set[str] = field(default_factory=set)
    msg_indices: Set[int] = field(default_factory=set)
    facts: List[str] = field(default_factory=list)
    fact_records: List[Dict[str, Any]] = field(default_factory=list)
    relations: List[Tuple[str, str]] = field(default_factory=list)
    dates: Set[str] = field(default_factory=set)
    orgs: Set[str] = field(default_factory=set)


@dataclass
class _EventRecord:
    event_type: str
    fact: str
    participants: List[str]
    dates: List[str]
    msg_indices: List[int]
    fact_records: List[Dict[str, Any]] = field(default_factory=list)


# ----------------------------------------------------------------------------
# N4: relative-date resolution
# ----------------------------------------------------------------------------
_WEEKDAY_MAP = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "日": 7,
    "天": 7,
    "monday": 1,
    "tuesday": 2,
    "wednesday": 3,
    "thursday": 4,
    "friday": 5,
    "saturday": 6,
    "sunday": 7,
}
_MONTH_EN = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def _resolve_relative_date(token: str, ts: datetime) -> Optional[str]:
    """Resolve one relative-date token to ISO ``YYYY-MM-DD`` against ``ts``."""
    rel_year = _REL_YEAR_RE.fullmatch(token)
    if rel_year:
        delta = {"last": -1, "this": 0, "next": 1}[rel_year.group(1).lower()]
        return (ts.replace(year=ts.year + delta, month=1, day=1)).strftime("%Y-%m-%d")
    rel = _REL_DAY_RE.fullmatch(token)
    if rel:
        if rel.group(1):  # Chinese day (大前天/昨天/今天/...)
            day_delta = {
                "大前天": -3,
                "前天": -2,
                "昨天": -1,
                "今天": 0,
                "明天": 1,
                "后天": 2,
            }[rel.group(1)]
        elif rel.group(2):  # the day before/after yesterday/tomorrow
            day_delta = -2 if "day before yesterday" in rel.group(2).lower() else 2
        else:  # English day
            day_delta = {"yesterday": -1, "today": 0, "tomorrow": 1}[rel.group(3).lower()]
        return (ts + timedelta(days=day_delta)).strftime("%Y-%m-%d")
    week = _WEEK_RE.fullmatch(token)
    if week:
        current_weekday = ts.isoweekday()  # Monday=1 ... Sunday=7
        if week.group(1):  # Chinese anchored week (上周X/这周X/下周X)
            anchor, target = week.group(1), week.group(2)
            delta = _WEEKDAY_MAP[target] - current_weekday
            if anchor == "上周":
                delta -= 7
            elif anchor == "下周":
                delta += 7
            return (ts + timedelta(days=delta)).strftime("%Y-%m-%d")
        if week.group(3):  # Chinese bare 周X: nearest occurrence (keep original delta semantics).
            target = week.group(4)
            delta = _WEEKDAY_MAP[target] - current_weekday
            return (ts + timedelta(days=delta)).strftime("%Y-%m-%d")
        if week.group(5):  # last/this/next week (week granularity)
            delta = {"last": -7, "this": 0, "next": 7}[week.group(5).lower()]
            return (ts + timedelta(days=delta)).strftime("%Y-%m-%d")
        if week.group(7):  # English weekday (last/this/next Monday | bare Monday)
            anchor_en, target_en = week.group(6), week.group(7)
            delta = _WEEKDAY_MAP[target_en.lower()] - current_weekday
            if anchor_en is not None:
                anchor_en = anchor_en.lower()
            if anchor_en == "last" and delta >= 0:
                delta -= 7  # still ahead this week -> previous week's occurrence
            elif anchor_en == "next" and delta <= 0:
                delta += 7  # already passed this week -> next week's occurrence
            elif anchor_en is None and delta <= 0:
                delta += 7
            return (ts + timedelta(days=delta)).strftime("%Y-%m-%d")
    n_day = _N_DAY_RE.fullmatch(token)
    if n_day:
        if n_day.group(1):  # Chinese 3天前/3天后
            days = int(n_day.group(1))
            direction = -1 if n_day.group(2) == "前" else 1
        elif n_day.group(3):  # 3 days ago / 2 weeks later
            unit = n_day.group(4).lower().rstrip("s")
            days = int(n_day.group(3)) * _N_DAY_UNIT_DAYS[unit]
            direction = -1 if n_day.group(5).lower() == "ago" else 1
        elif n_day.group(6):  # in 3 days / in 2 weeks' time -> future
            unit = n_day.group(7).lower().rstrip("s")
            days = int(n_day.group(6)) * _N_DAY_UNIT_DAYS[unit]
            direction = 1
        else:
            return None
        return (ts + timedelta(days=days * direction)).strftime("%Y-%m-%d")
    return None


def _abs_date_from_token(token: str, ts: Optional[datetime] = None) -> Optional[str]:
    """Resolve numeric absolute dates (2026-09-23 / 9月30号 / May 7 2023)."""
    abs_match = _ABS_DATE_RE.search(token)
    if abs_match:
        year, month, day = (int(g) for g in abs_match.groups())
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            return None
    md_match = _MD_DATE_RE.search(token)
    if md_match:
        month, day = (int(g) for g in md_match.groups())
        if ts is None:
            return None
        year = ts.year
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            return None
    en_match = _EN_ABS_DATE_RE.search(token)
    if en_match:
        if en_match.group(1):  # May 7 2023 / on May 7, 2023
            month = _MONTH_EN[en_match.group(1).lower()[:3]]
            day, year = int(en_match.group(2)), int(en_match.group(3))
        elif en_match.group(5):  # 7 May 2023
            month = _MONTH_EN[en_match.group(5).lower()[:3]]
            day, year = int(en_match.group(4)), int(en_match.group(6))
        else:  # May 2023 / in June 2023 -> first day of month
            month = _MONTH_EN[en_match.group(7).lower()[:3]]
            day, year = 1, int(en_match.group(8))
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            return None
    return None


def _resolve_date_token(token: str, ts: Optional[datetime]) -> Optional[str]:
    if token is None:
        return None
    resolved = _abs_date_from_token(token, ts)
    if resolved:
        return resolved
    if ts is not None:
        return _resolve_relative_date(token, ts)
    return None


def _extract_date_tokens(text: str) -> List[str]:
    """Collect raw date-like tokens with their matched spans."""
    hits: List[Tuple[int, int, str]] = []
    for pattern in _DATE_TOKEN_RES:
        for match in pattern.finditer(text):
            hits.append((match.start(), match.end(), match.group(0)))
    # Deduplicate overlapping spans (keep the longer/first match).
    hits.sort()
    merged: List[Tuple[int, int, str]] = []
    for start, end, token in hits:
        if merged and start < merged[-1][1]:
            prev_start, prev_end, _ = merged[-1]
            if end > prev_end:
                merged[-1] = (prev_start, end, token)
            continue
        merged.append((start, end, token))
    return [token for _, _, token in merged]


def _resolve_dates_in_text(text: str, ts: Optional[datetime]) -> Tuple[str, List[str]]:
    """Replace relative date tokens in ``text`` with absolute ISO dates."""
    matches: List[Tuple[int, int, str]] = []
    for pattern in _DATE_TOKEN_RES:
        for match in pattern.finditer(text):
            resolved = _resolve_date_token(match.group(0), ts)
            if resolved:
                matches.append((match.start(), match.end(), resolved))
    if not matches:
        return text, []
    matches.sort()
    # Merge overlapping spans, keeping the longest resolution.
    merged: List[Tuple[int, int, str]] = []
    for start, end, resolved in matches:
        if merged and start < merged[-1][1]:
            prev_start, prev_end, _ = merged[-1]
            if end > prev_end:
                merged[-1] = (prev_start, end, resolved)
            continue
        merged.append((start, end, resolved))
    pieces: List[str] = []
    resolved_dates: List[str] = []
    cursor = 0
    for start, end, resolved in merged:
        if cursor < start:
            pieces.append(text[cursor:start])
        pieces.append(resolved)
        resolved_dates.append(resolved)
        cursor = end
    if cursor < len(text):
        pieces.append(text[cursor:])
    return "".join(pieces), sorted(set(resolved_dates))


# ----------------------------------------------------------------------------
# N2: NER loader with regex-only degradation
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# N5 / N7: title merge, alias aggregation, relation pairing
# ----------------------------------------------------------------------------
def _merge_title_entities(ner_entities: List[str], text: str) -> List[str]:
    """Replace bare-surname PERSON spans with their title forms (张 -> 张总)."""
    title_forms = {match.group(0) for match in _TITLE_RE.finditer(text)}
    result: List[str] = []
    for entity in ner_entities:
        if len(entity) == 1 and any(form.startswith(entity) for form in title_forms):
            continue  # bare surname shadowed by a title form
        result.append(entity)
    for form in sorted(title_forms, key=lambda item: (-len(item), item)):
        if form not in result:
            result.append(form)
    return result


def _canonicalize(entity: str) -> str:
    """Normalize an entity key: latin lowercase/underscore, Chinese untouched.
    Speaker-prefix brackets (``[Melanie]: ...``) leak into NER spans, so strip
    bracket/quote characters from latin keys instead of carrying them into
    filenames.
    """
    normalized = entity.strip()
    if not normalized:
        return normalized
    if re.search(r"[\u4e00-\u9fa5]", normalized):
        return normalized
    return re.sub(r"[\s\-\[\]\(\)\{\}<>\'\"`]+", "_", normalized.lower()).strip("_")


# ----------------------------------------------------------------------------
# N9: confidence scoring (never boosts)
# ----------------------------------------------------------------------------
def _confidence(entity: str) -> float:
    score = 1.0
    if len(entity) < 2:
        score *= 0.7
    elif len(entity) > 50:
        score *= 0.9
    if _CODE_PATH_RE.fullmatch(entity) or _QUANTITY_RE.fullmatch(entity):
        score = min(score + 0.05, 1.0)
    return round(score, 2)


# ----------------------------------------------------------------------------
# Entity content builder
# ----------------------------------------------------------------------------
_ENTITY_SECTION_FACTS = "## 原始事实"
_ENTITY_SECTION_RELATIONS = "## 关系"
_ENTITY_SECTION_DATES = "## 提及时间"


def _build_entity_content(record: _EntityRecord, text_facts: List[str]) -> str:
    lines: List[str] = ["# " + record.canonical, ""]
    lead = text_facts[0] if text_facts else ""
    if lead:
        sentence = lead.split("。", 1)[0]
        lines.append(sentence[:60] + ("…" if len(sentence) > 60 else ""))
    else:
        lines.append("对话中提及该实体，相关事实如下。")
    lines.append("")
    sections: List[Tuple[str, List[str]]] = []
    facts = text_facts
    if facts:
        sections.append((_ENTITY_SECTION_FACTS, ["- " + fact for fact in facts]))
    if record.relations:
        sections.append(
            (
                _ENTITY_SECTION_RELATIONS,
                ["- 与{0}：{1}".format(other, term) for other, term in record.relations],
            )
        )
    if record.dates:
        sections.append((_ENTITY_SECTION_DATES, ["- " + date for date in sorted(record.dates)[:8]]))
    for heading, bullets in sections[:4]:
        lines.append(heading)
        lines.extend(bullets)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _merge_entity_content(old_content: str, new_content: str) -> str:
    """Append new fact bullets into the old card, preserving every old fact.
    ``content`` uses merge_op=patch, and a plain-string patch is a *full
    replacement* (merge_op/patch.py Case 3).  We therefore must carry old facts
    forward ourselves so later windows do not erase earlier ones.
    """
    old = (old_content or "").strip()
    if not old:
        return new_content
    new_bullets = [
        line.strip() for line in new_content.splitlines() if line.strip().startswith("- ")
    ]
    missing = [
        bullet
        for bullet in new_bullets
        if bullet not in {line.strip() for line in old.splitlines()}
    ]
    if not missing:
        return old + "\n"
    if _ENTITY_SECTION_FACTS in old:
        lines = old.rstrip().splitlines()
        insert_at = len(lines)
        for idx, line in enumerate(lines):
            if line.strip() == _ENTITY_SECTION_FACTS:
                insert_at = idx + 1
                while insert_at < len(lines) and lines[insert_at].strip().startswith("- "):
                    insert_at += 1
                break
        merged = lines[:insert_at] + ["- " + bullet[2:] for bullet in missing] + lines[insert_at:]
        return "\n".join(merged).rstrip() + "\n"
    return (
        old.rstrip()
        + "\n\n"
        + _ENTITY_SECTION_FACTS
        + "\n"
        + "\n".join("- " + bullet[2:] for bullet in missing)
        + "\n"
    )


# ----------------------------------------------------------------------------
# Event naming / summary builders
# ----------------------------------------------------------------------------
_PUNCT_RE = re.compile(r"[\s,，。.!！?？:：;；、（）()\[\]【】“”\"']+")
_SPEAKER_PREFIX_RE = re.compile(r"^\s*\[([^\]]+)\]:\s*")


def _split_speaker(text: str) -> Tuple[Optional[str], str]:
    """Split a ``[Name]: body`` chat line into (speaker, body)."""
    match = _SPEAKER_PREFIX_RE.match(text)
    if not match:
        return None, text
    return match.group(1).strip(), text[match.end() :]


def _sanitize_event_name(text: str) -> str:
    """Keep Chinese/letters only; strip digits, dates and punctuation."""
    for pattern in _DATE_TOKEN_RES:
        text = pattern.sub("", text)
    cleaned = _PUNCT_RE.sub("", text)
    cleaned = re.sub(r"[0-9._:/\\-]+", "", cleaned)
    return cleaned[:12]


def _event_name_for(event_type: str, fact: str, participants: List[str]) -> str:
    suffix = _EVENT_NAME_SUFFIX[event_type]
    for pattern in _EVENT_VERB_PATTERNS[event_type]:
        match = pattern.search(fact)
        if not match:
            continue
        # Captured destination/place is the most specific topic (成都出行).
        place = match.groupdict().get("place") if match.groupdict() else None
        if place:
            return f"{place}{suffix}"[:12]
        tail = fact[match.end() : match.end() + 10]
        topic = _sanitize_event_name(tail)
        if topic:
            return f"{topic}{suffix}"[:12]
    # Fall back to the participant, then the bare suffix.
    if participants:
        name = f"{participants[0]}{suffix}"
        if not re.search(r"[0-9]", name):
            return name[:12]
    return suffix


def _build_event_summary(event: _EventRecord, fact: str) -> str:
    parts: List[str] = []
    if event.dates:
        parts.append("时间：" + "、".join(sorted(set(event.dates))))
    if event.participants:
        parts.append("参与者：" + "、".join(event.participants))
    prefix = "，".join(parts)
    if prefix:
        return f"{prefix}。{fact}"
    return fact


def _build_ranges(indices: List[int]) -> str:
    """Compact ``0-3,7`` style ranges from a sorted index list."""
    if not indices:
        return ""
    indices = sorted(set(indices))
    ranges: List[str] = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = idx
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


# ----------------------------------------------------------------------------
# Main extractor
# ----------------------------------------------------------------------------
