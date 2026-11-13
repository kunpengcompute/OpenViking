# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Conservative, source-grounded task matching and complementary recall selection."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from openviking.integrations.langchain.ov_context.recall_content import (
    MemoryFeatures,
    memory_features,
)
from openviking_cli.retrieve.types import MatchedContext

from .recall_methods import file_method_capabilities

_HISTORY = re.compile(
    r"上次|之前|以前|此前|历史|过去|昨天|上周|\b(?:last|previous|history|earlier)\b", re.I
)
_LOOKUP = re.compile(
    r"查询|查看|查找|找回|找|回顾|查一下|多少|什么|哪些|哪里|结果|原因|情况|\b(?:where|when|what|show|find|retrieve)\b",
    re.I,
)
_NEGATED = re.compile(
    r"不要|不得|禁止|不能|无需|不必|不(?=修改|删除|新增|创建|新建|编辑|取消)|\b(?:do not|don't|must not|never)\b",
    re.I,
)
_ACTIONS = {
    "create": re.compile(
        r"新增|新建|创建|登记|预约|预订|安排|\b(?:create|register|schedule)\b", re.I
    ),
    "update": re.compile(r"修改|编辑|更新|调整|\b(?:edit|modify|update)\b", re.I),
    "close": re.compile(
        r"删除|移除|撤销|取消|关闭|停用|\b(?:delete|remove|cancel|deactivate)\b", re.I
    ),
}
_FAMILIES = {
    "create_new_object": "create",
    "update_existing_object": "update",
    "corrective_adjustment": "update",
    "close_or_deactivate_object": "close",
    "read_verify_only": "verify",
}
_FILE = re.compile(r"[\w/-]+\.(?:png|csv|txt|xlsx|pdf|docx|pptx|json|md)\b", re.I)
_FILE_LOOKUP = re.compile(r"读取|打开|查找|找回|搜索|\b(?:read|open|find|search|locate)\b", re.I)
_TOPIC_STOP = {
    "新增",
    "新建",
    "创建",
    "登记",
    "修改",
    "编辑",
    "更新",
    "调整",
    "删除",
    "移除",
    "取消",
    "完成",
    "执行",
    "操作",
    "流程",
    "方法",
    "表单",
    "字段",
    "数据",
    "信息",
    "工作",
    "平台",
    "系统",
    "记录",
    "目标",
    "核对",
    "create",
    "edit",
    "update",
    "delete",
    "please",
}


@dataclass(frozen=True)
class QueryPurpose:
    kind: str = "unknown"
    family: str = ""
    logged_in: bool | None = None
    artifacts: bool = False
    task: str = ""


def query_purpose(query: str) -> QueryPurpose:
    """Recognize explicit requests only; quoted/negative/past actions are not new writes."""
    task = query.split("【任务】", 1)[-1].split("【方法提示】", 1)[0]
    task = re.sub(r"```[\s\S]*?(?:```|\Z)", " ", task)
    task = re.sub(r"(?m)^>[^\n]*$", " ", task)
    families: set[str] = set()
    history = False
    for clause in re.split(r"[，,。；;\n]+|然后|接着|随后|同时", task):
        positive = _NEGATED.split(clause, 1)[0]
        positive = _FILE.sub(" ", positive)
        past_lookup = bool(_HISTORY.search(positive) and _LOOKUP.search(positive))
        history |= past_lookup
        if re.match(
            r"\s*(?:请|帮我)?(?:核对|核验|检查|验证|确认|查看|查询)", positive
        ) and re.search(
            r"已有|现有|已(?:新增|新建|创建|保存)|\b(?:existing|already)\b", positive, re.I
        ):
            continue
        if not past_lookup or re.search(
            r"(?:再|另外|现在|继续|重新).*(?:新增|新建|创建|修改|删除)", positive
        ):
            families.update(name for name, pattern in _ACTIONS.items() if pattern.search(positive))
    explicit_file = bool(
        _FILE.search(task)
        and _FILE_LOOKUP.search(task)
        and not families
        and not re.search(
            r"分析|统计|计算|绘图|生成|导出|处理|\b(?:analyze|compute|calculate|plot|generate)\b",
            task,
            re.I,
        )
    )
    history |= explicit_file
    # The deliverable is the action. Business rules quoted inside a document
    # request are data, even when they mention create/update/cancel operations.
    primary = re.split(r"\n\s*\n|\n[【#]", task.strip(), maxsplit=1)[0]
    document_task = re.search(
        r"(?:编写|撰写|起草|生成|制作|输出|修改|更新|编辑).{0,30}(?:文档|报告|说明|PPT|幻灯片|表格|文件)|"
        r"\b(?:write|draft|generate|create|edit|update)\b.{0,60}\b(?:document|report|presentation|spreadsheet)\b",
        primary,
        re.I,
    )
    history_request = bool(_HISTORY.search(primary) and _LOOKUP.search(primary))
    if document_task and not history_request:
        families, history = (
            {
                "update"
                if re.match(r"修改|更新|编辑|edit\b|update\b", document_task.group(0), re.I)
                else "create"
            },
            False,
        )
    family = next(iter(families)) if len(families) == 1 else ""
    if history and families or len(families) > 1:
        kind = "mixed"
    elif history:
        kind = "history"
    elif family:
        kind = "execute"
    elif re.search(r"核验|核对|检查|验证|\b(?:verify|validate|check)\b", task, re.I):
        kind, family = "execute", "verify"
    else:
        kind = "unknown"
    state_prefix = (
        r"(?:^|[，,。\n;；])\s*(?:我(?:现在)?|当前(?:浏览器|会话)?|浏览器|会话|现在|目前|i am)?\s*"
    )
    state_query = re.sub(r"```[\s\S]*?(?:```|\Z)", " ", query)
    state_query = re.sub(r"(?m)^>[^\n]*$", " ", state_query)
    logged = bool(
        re.search(state_prefix + r"(?:已经登录|已登录|already logged in)", state_query, re.I)
    )
    not_logged = bool(
        re.search(state_prefix + r"(?:尚未登录|还没登录|未登录|not logged in)", state_query, re.I)
    )
    return QueryPurpose(
        kind,
        family,
        logged if logged != not_logged else None,
        bool(
            re.search(r"截图|附件|下载|文件|\b(?:artifact|attachment|file|download)\b", task, re.I)
            or explicit_file
        ),
        task,
    )


def _kind(item: MatchedContext) -> str:
    parts = item.uri.rstrip("/").split("/")
    if "memories" in parts:
        rest = parts[parts.index("memories") + 1 :]
        return rest[0] if rest else ""
    return item.category.lower()


def is_event_memory(item: MatchedContext) -> bool:
    return _kind(item) == "events"


def _login_method(features: MemoryFeatures) -> bool:
    return bool(
        features.steps
        and re.search(r"登录|login|sign.in", features.domain + " " + features.target, re.I)
    )


def _explicitly_excludes(features: MemoryFeatures, family: str) -> bool:
    pattern = _ACTIONS.get(family)
    if pattern is None:
        return False
    for line in features.exclusions:
        match = re.match(r"(?:不适用于|不适用|not for|not applicable to)\s*(.+)", line, re.I)
        if not match or re.search(r"以外|之外|except|other than", match[1], re.I):
            continue
        target = re.split(r"但|不过|\bbut\b", match[1], maxsplit=1, flags=re.I)[0]
        if re.match(
            r"新增|新建|创建|登记|修改|编辑|删除|移除|取消|create\b|edit\b|delete\b", target, re.I
        ):
            if pattern.search(target):
                return True
    return False


def method_capabilities(features: MemoryFeatures) -> set[str]:
    """Only literal tool/parameter evidence is normalized; paraphrases stay uncertain."""
    text = "\n".join(_NEGATED.split(unit, 1)[0] for unit in features.steps + features.parameters)
    result = set()
    if re.search(r"snapshot|快照", text, re.I) and re.search(r"refs?|引用|元素", text, re.I):
        result.add("refresh_refs")
    if re.search(r"\bbatch\b", text, re.I) and re.search(r"表单|字段|填写|form|fill", text, re.I):
        result.add("batch_form")
        if "--stdin" in text:
            result.add("batch_stdin")
    if re.search(r"\bhref\b", text, re.I) and re.search(r"\beval\b|\bopen\b|导航", text, re.I):
        result.add("href_navigation")
    if re.search(r"screenshot", text, re.I) and re.search(r"filename|full\s*[:=]", text, re.I):
        result.add("capture_parameters")
    return result | file_method_capabilities(text)


def _file_needs(purpose: QueryPurpose) -> set[str]:
    positive = "\n".join(
        _NEGATED.split(clause, 1)[0] for clause in re.split(r"[，,。；;\n]+", purpose.task)
    )
    return file_method_capabilities(positive)


def _needed(features: MemoryFeatures, purpose: QueryPurpose) -> set[str]:
    text = purpose.task + "\n" + "\n".join(features.steps)
    needs = set()
    if re.search(r"填写|表单|字段|导航|进入|打开|点击|form|fill|navigate|\bopen\b", text, re.I):
        needs.update(("refresh_refs", "href_navigation"))
    if re.search(r"填写|表单|字段|form|fill", text, re.I) and re.search(
        r"多|批量|分别|各|\||fields", text, re.I
    ):
        needs.update(("batch_form", "batch_stdin"))
    if re.search(r"截图|screenshot", text, re.I):
        needs.add("capture_parameters")
    return needs | _file_needs(purpose)


def select_by_value(
    items: list[MatchedContext],
    query: str,
    *,
    base_scores: dict[int, float],
    terms: Callable[[str], set[str]],
    query_terms: Callable[[str], set[str]] | None = None,
    diagnostics: list[dict] | None = None,
) -> list[MatchedContext]:
    """Prioritize applicable contracts and explicit missing capabilities, without quotas.

    Only explicit same-domain operation-family conflicts are rejected. Ordinary
    file/data procedures may provide literal methods without a declared family.
    Unknown state and mixed requests otherwise retain relevance ordering.
    No original memory, model prompt, count cap or token budget is modified.
    """
    purpose = query_purpose(query)
    features = {id(item): memory_features(item.abstract or item.overview or "") for item in items}
    wanted_terms = (query_terms or terms)(purpose.task) - _TOPIC_STOP
    capabilities = {key: method_capabilities(f) for key, f in features.items()}
    evidence_matches = {
        key: wanted_terms & (terms(" ".join((f.title, *f.steps))) - _TOPIC_STOP)
        for key, f in features.items()
    }

    def topic(item: MatchedContext) -> set[str]:
        f = features[id(item)]
        return terms(" ".join((f.domain, f.trigger, f.target))) - _TOPIC_STOP

    def matches(item: MatchedContext) -> set[str]:
        return wanted_terms & topic(item)

    reasons = {id(item): "relevance_fallback" for item in items}
    additions: dict[int, set[str]] = {}
    rejected: set[int] = set()
    preferred: list[MatchedContext] = []
    if purpose.kind == "execute":
        for item in items:
            f = features[id(item)]
            family = _FAMILIES.get(f.family)
            if (
                family
                and family != purpose.family
                and matches(item)
                and _kind(item) not in {"events", "cases"}
                and not _login_method(f)
            ):
                rejected.add(id(item))
                reasons[id(item)] = "explicit_operation_conflict"
            elif (
                matches(item)
                and _kind(item) not in {"events", "cases"}
                and _explicitly_excludes(f, purpose.family)
            ):
                rejected.add(id(item))
                reasons[id(item)] = "explicit_applicability_exclusion"
        mains = [
            item
            for item in items
            if id(item) not in rejected
            and _kind(item) not in {"events", "cases"}
            and _FAMILIES.get(features[id(item)].family) == purpose.family
            and len(features[id(item)].steps) >= 2
            and matches(item)
            and not _login_method(features[id(item)])
        ]
        if mains:
            main = max(mains, key=lambda item: (len(matches(item)), base_scores[id(item)]))
            preferred.append(main)
            reasons[id(main)] = "applicable_primary_procedure"
            main_features = features[id(main)]
            covered = method_capabilities(main_features)
            needed = _needed(main_features, purpose)
            while True:
                extra = [
                    (item, (method_capabilities(features[id(item)]) & needed) - covered)
                    for item in items
                    if item not in preferred
                    and id(item) not in rejected
                    and _kind(item) not in {"events", "cases"}
                    and features[id(item)].methods
                    and not _login_method(features[id(item)])
                    and (not features[id(item)].domain or matches(item))
                ]
                extra = [(item, caps) for item, caps in extra if caps]
                if not extra:
                    break
                item, caps = max(extra, key=lambda pair: (len(pair[1]), base_scores[id(pair[0])]))
                preferred.append(item)
                covered.update(caps)
                additions[id(item)] = caps
                reasons[id(item)] = "explicit_missing_capability"
            for item in items:
                if (
                    item not in preferred
                    and id(item) not in rejected
                    and features[id(item)].methods
                ):
                    reasons[id(item)] = "no_verified_capability_gap"
            if re.search(r"登录|login", "\n".join(main_features.conditions), re.I):
                login = [
                    item
                    for item in items
                    if _login_method(features[id(item)]) and id(item) not in rejected
                ]
                if login and purpose.logged_in is not True:
                    item = max(login, key=lambda item: base_scores[id(item)])
                    preferred.insert(0 if purpose.logged_in is False else len(preferred), item)
                    reasons[id(item)] = (
                        "required_login_prerequisite"
                        if purpose.logged_in is False
                        else "conditional_login_prerequisite"
                    )
                elif purpose.logged_in is True:
                    for item in login:
                        rejected.add(id(item))
                        reasons[id(item)] = "login_already_satisfied"
        else:
            # Ordinary memories need not declare a create/update contract. Use
            # explicit file/data evidence only when the current task asks for it.
            needed = _file_needs(purpose)
            covered: set[str] = set()
            while needed - covered:
                extra = []
                for item in items:
                    f = features[id(item)]
                    if item in preferred or id(item) in rejected or f.family or not f.steps:
                        continue
                    if not evidence_matches[id(item)]:
                        continue
                    caps = (capabilities[id(item)] & needed) - covered
                    if caps:
                        extra.append((item, caps))
                if not extra:
                    break
                item, caps = max(extra, key=lambda pair: (len(pair[1]), base_scores[id(pair[0])]))
                preferred.append(item)
                covered.update(caps)
                additions[id(item)] = caps
                reasons[id(item)] = (
                    "applicable_evidence_method"
                    if len(preferred) == 1
                    else "explicit_missing_capability"
                )
    elif purpose.kind == "history":
        history = [
            item
            for item in items
            if _kind(item) in {"events", "cases"}
            and wanted_terms & (terms(item.abstract or item.overview or "") - _TOPIC_STOP)
        ]
        if purpose.artifacts:
            preferred = [item for item in history if features[id(item)].has_artifacts]
            for item in preferred:
                reasons[id(item)] = "historical_artifact_lookup"
        else:
            request_lookup = bool(
                re.search(r"任务要求|原始请求|原始任务|任务描述|要求是什么", purpose.task)
            )
            role = "request" if request_lookup else "result"
            preferred = [item for item in history if features[id(item)].historical_role == role]
            for item in preferred:
                reasons[id(item)] = (
                    "historical_original_request"
                    if request_lookup
                    else "historical_observed_result"
                )
    result = preferred + [
        item for item in items if item not in preferred and id(item) not in rejected
    ]
    if diagnostics is not None:
        for item in [*result, *(item for item in items if id(item) in rejected)]:
            f = features[id(item)]
            diagnostics.append(
                {
                    "uri": item.uri,
                    "purpose": purpose.kind,
                    "requested_family": purpose.family,
                    "memory_family": f.family,
                    "domain_matches": sorted(matches(item)),
                    "evidence_matches": sorted(evidence_matches[id(item)]),
                    "provided_capabilities": sorted(capabilities[id(item)]),
                    "reason": reasons[id(item)],
                    "status": "rejected" if id(item) in rejected else "ranked",
                    "added_capabilities": sorted(additions.get(id(item), set())),
                    "pending_preconditions": list(f.conditions) if item in preferred else [],
                    "known_login_state": purpose.logged_in,
                }
            )
    return result
