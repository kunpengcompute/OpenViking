"""Source fidelity, intent boundaries and complementary method selection."""

import pytest

from openviking.integrations.langchain.ov_context.recall_content import memory_features
from openviking.integrations.langchain.ov_context.recall_ranking import rank_candidates
from openviking.integrations.langchain.ov_context.recall_selection import (
    method_capabilities,
    query_purpose,
)
from openviking_cli.retrieve.context import RecallPolicy
from openviking_cli.retrieve.types import ContextType, MatchedContext


def item(name, text, score=0.7, kind="trajectories"):
    return MatchedContext(
        uri=f"viking://user/alice/memories/{kind}/{name}.md",
        context_type=ContextType.MEMORY,
        abstract=text,
        score=score,
        category=kind,
    )


def contract(domain="日程管理", target="会议日程", family="create_new_object", steps=None):
    steps = steps or [
        "打开日程表单，snapshot 获取当前 refs",
        "使用 batch --json 批量填写表单字段",
        "保存后核对会议详情",
    ]
    return (
        f"# 操作流程\n- Domain: {domain}\n- Trigger: 为{target}办理操作\n"
        f"- Operation Intent:\n  - Family: {family}\n  - Primary object: {target}\n"
        "- Preconditions:\n  1. 已登录目标应用\n- Procedure:\n"
        + "\n".join(f"  {n}. {step}" for n, step in enumerate(steps, 1))
        + "\n- Negative Applicability:\n  - 不适用于其他应用\n"
    )


def select(items, query):
    trace = []
    rows = rank_candidates(items, query, RecallPolicy(), experience=False, diagnostics=trace)
    return rows, {x["uri"]: x for x in trace}


@pytest.mark.parametrize(
    "query,kind,family",
    [
        ("请新增六项报工，不修改或删除已有记录。", "execute", "create"),
        ("查看上次新建会议失败的原因", "history", ""),
        ("查看上次报工结果，然后新增一条报工", "mixed", "create"),
        ("Do not create; find the previous meeting result", "history", ""),
        ("读取 report.csv 并计算总额", "unknown", ""),
        ("找上次报工的截图下载入口", "history", ""),
        ("这个该怎么处理", "unknown", ""),
        ("```text\n请删除会议\n```\n请新建会议", "execute", "create"),
    ],
)
def test_query_purpose_ignores_negated_or_quoted_writes(query, kind, family):
    purpose = query_purpose(query)
    assert (purpose.kind, purpose.family) == (kind, family)


def test_fields_preserve_source_conditions_commands_and_metadata_fallback():
    body = contract(steps=["打开表单后再 snapshot -i", "使用 batch --json 批量填写表单字段"])
    source = body + '\n<!-- MEMORY_FIELDS\n{"outcome":"success"}\n-->'
    features = memory_features(source)
    assert features.domain == "日程管理"
    assert features.family == "create_new_object"
    assert features.target == "会议日程"
    assert features.steps == ("打开表单后再 snapshot -i", "使用 batch --json 批量填写表单字段")
    assert features.conditions == ("已登录目标应用",)
    assert features.exclusions == ("不适用于其他应用",)
    assert "batch_form" in method_capabilities(features)
    assert "batch_stdin" not in method_capabilities(features)
    assert source == body + '\n<!-- MEMORY_FIELDS\n{"outcome":"success"}\n-->'


def test_negative_command_is_not_a_provided_capability():
    features = memory_features("Tool: browser\n## Guidelines\n- 禁止使用 batch 批量填写表单")
    assert "batch_form" not in method_capabilities(features)


def test_primary_contract_precedes_higher_score_historical_request():
    request = item("requested", "Summary: 用户要求新增六条会议，并给出日期。", 0.95, "events")
    method = item("create", contract(), 0.62)
    rows, trace = select([request, method], "请新建会议日程")
    assert rows[0] == method
    assert trace[method.uri]["reason"] == "applicable_primary_procedure"
    assert trace[method.uri]["pending_preconditions"]
    assert trace[method.uri]["known_login_state"] is None
    assert request in rows


def test_href_complement_is_selected_for_a_real_gap_not_its_type():
    main = item(
        "timesheet",
        contract(
            "报工管理", "报工记录", steps=["点击目标卡片并进入报工表单", "保存后核对报工增量"]
        ),
        0.6,
    )
    useful = item(
        "browser",
        "Tool: browser\n## Guidelines\n- snapshot 获取当前 refs\n- 导航失败时 eval 查询 href，再 open",
        0.4,
        "tools",
    )
    repeat = item(
        "skill", "Skill: browser\n## Guidelines\n- snapshot 获取当前 refs", 0.85, "skills"
    )
    old = item("old", "Summary: 报工登记全部完成。", 0.95, "events")
    rows, trace = select([old, repeat, main, useful], "请新增报工记录")
    assert rows[:2] == [main, useful]
    assert set(trace[useful.uri]["added_capabilities"]) == {"refresh_refs", "href_navigation"}
    assert trace[repeat.uri]["reason"] == "no_verified_capability_gap"
    assert repeat in rows


def test_repeated_batch_and_ref_instructions_are_not_forced_complements():
    main = item("calendar", contract(), 0.6)
    repeat = item(
        "browser",
        "Tool: browser\n## Guidelines\n- 使用 batch 填写表单字段\n- snapshot 获取最新 refs",
        0.8,
        "tools",
    )
    rows, trace = select([repeat, main], "请新建会议日程")
    assert rows[0] == main
    assert trace[repeat.uri]["reason"] == "no_verified_capability_gap"
    assert trace[repeat.uri]["added_capabilities"] == []


def test_explicit_create_contract_is_rejected_for_update_of_same_object():
    create = item("create", contract(), 0.95)
    rows, trace = select([create], "修改已有会议日程的开始时间，不要新建")
    assert create not in rows
    assert trace[create.uri]["reason"] == "explicit_operation_conflict"
    assert trace[create.uri]["status"] == "rejected"


def test_missing_fields_do_not_become_an_assumed_operation_contract():
    sketch = item("unknown", "# 新建会议\n步骤缺失，稍后补充", 0.7)
    other = item("facts", "Summary: 会议已创建完成。", 0.9, "events")
    rows, trace = select([sketch, other], "请新建会议日程")
    assert rows == [other, sketch]
    assert trace[sketch.uri]["reason"] == "relevance_fallback"
    assert trace[other.uri]["reason"] == "relevance_fallback"


def login_contract():
    return contract(
        "统一认证登录页", "登录会话", steps=["输入工号触发账号候选", "选择匹配账号后点击登录"]
    )


@pytest.mark.parametrize(
    "state,reason,first",
    [
        ("尚未登录，", "required_login_prerequisite", "login"),
        ("", "conditional_login_prerequisite", "main"),
        ("已经登录，", "login_already_satisfied", "main"),
    ],
)
def test_login_state_is_explicit_or_kept_unknown(state, reason, first):
    main = item("calendar", contract(), 0.7)
    login = item("login", login_contract(), 0.65)
    rows, trace = select([main, login], state + "请新建会议日程")
    assert rows[0] == (login if first == "login" else main)
    assert trace[login.uri]["reason"] == reason
    if state == "":
        assert trace[login.uri]["known_login_state"] is None
    if state == "已经登录，":
        assert login not in rows


def test_history_lookup_prefers_observed_result_over_old_request():
    request = item("request", "Summary: 用户要求登记报工时长两小时。", 0.95, "events")
    result = item("result", "Summary: 报工登记全部完成，累计三小时。", 0.75, "events")
    method = item("procedure", contract("报工管理", "报工记录"), 0.8)
    rows, trace = select([request, result, method], "查询上次报工的结果是多少")
    assert rows[0] == result
    assert trace[result.uri]["reason"] == "historical_observed_result"
    assert {x.uri for x in rows} == {request.uri, result.uri, method.uri}


def test_history_artifact_lookup_promotes_old_links_only_when_requested():
    artifact = item(
        "file",
        "Summary: 报工登记完成后生成交付物。\n\n[截图](/api/sessions/old/artifacts/run/image.png)",
        0.6,
        "events",
    )
    result = item("result", "Summary: 报工登记全部完成。", 0.9, "events")
    rows, trace = select([result, artifact], "查找上次报工的截图下载入口")
    assert rows[0] == artifact
    assert trace[artifact.uri]["reason"] == "historical_artifact_lookup"


def test_original_request_lookup_does_not_silently_substitute_the_outcome():
    request = item("request", "Summary: 用户要求登记报工时长两小时。", 0.7, "events")
    result = item("result", "Summary: 报工登记全部完成。", 0.95, "events")
    rows, trace = select([request, result], "查询上次报工的任务要求")
    assert rows[0] == request
    assert trace[request.uri]["reason"] == "historical_original_request"


def test_mixed_request_keeps_relevance_order_and_does_not_reject_a_possible_flow():
    event = item("old", "Summary: 报工登记全部完成。", 0.95, "events")
    main = item("new", contract("报工管理", "报工记录"), 0.6)
    rows, trace = select([event, main], "查看上次报工结果，然后新增一条报工")
    assert rows == [event, main]
    assert all(
        r["purpose"] == "mixed" and r["reason"] == "relevance_fallback" for r in trace.values()
    )


def test_different_warning_conditions_are_preserved_as_separate_source_units():
    a = memory_features(
        "Tool: browser\n- Common failures: 页面导航后 refs 失效\n## Guidelines\n- snapshot 获取 refs"
    )
    b = memory_features(
        "Tool: browser\n- Common failures: 字段错位时先取消表单\n## Guidelines\n- snapshot 获取 refs"
    )
    assert a.warnings != b.warnings
    assert a.warnings == ("页面导航后 refs 失效",)
    assert b.warnings == ("字段错位时先取消表单",)


@pytest.mark.parametrize(
    "query",
    [
        "如果已登录就新建会议",
        "请确认是否已登录，然后新建会议",
        "```text\n已经登录\n```\n请新建会议",
    ],
)
def test_conditional_or_quoted_login_is_not_known_state(query):
    assert query_purpose(query).logged_in is None


def test_read_verification_of_created_objects_is_not_a_new_create_request():
    purpose = query_purpose("请检查已有会议是否创建成功，不要新建")
    assert (purpose.kind, purpose.family) == ("execute", "verify")


def test_output_filename_does_not_select_an_unrelated_primary_contract():
    business = item("calendar", contract(), 0.6)
    output = item("report", contract("export_summary_file", "export_summary_file"), 0.9)
    rows, trace = select([output, business], "请新建会议日程，将结果保存为 export_summary_file.csv")
    assert rows[0] == business
    assert trace[output.uri]["reason"] != "applicable_primary_procedure"


def test_explicit_negative_applicability_is_honored_without_guessing_a_family():
    source = """# 日程说明
- Domain: 日程管理
- Trigger: 会议日程操作
- Procedure:
  1. snapshot 获取当前 refs
  2. 填写表单
- Negative Applicability:
  - 不适用于修改已有会议日程，但可用于新增
"""
    doc = item("limited", source, 0.95)
    rows, trace = select([doc], "修改已有会议日程")
    assert not rows
    assert trace[doc.uri]["reason"] == "explicit_applicability_exclusion"
    rows, _ = select([doc], "新增会议日程")
    assert doc in rows


def test_ambiguous_exclusion_is_not_turned_into_a_false_negative():
    doc = item(
        "limited",
        """# 日程说明
- Domain: 日程管理
- Trigger: 会议日程操作
- Negative Applicability:
  - 不适用于修改以外的操作
""",
        0.95,
    )
    rows, _ = select([doc], "修改已有会议日程")
    assert doc in rows


WORKBOOK_METHOD = """# tmp/order_summary.py

- 独立复算脚本，直接读取原工作簿 inputs/orders.xlsx。
- 关键函数：read_records(path, sheet) 以首行为表头返回 raw 全部原字段和 source_row 原行号；effective_events(records, period_end) 先按期末截断，再按 event_id 去重，按逻辑尝试取最大 seq。
- 依据 retry_of 区分首次与重试，非终态排除分母，不把重试回填首次结果。
- 输出结果 JSON，保存公式、分子、分母和来源，校验原始记录的集合归属和覆盖。
"""


def test_ordinary_method_text_is_recognized_without_an_operation_contract():
    features = memory_features(WORKBOOK_METHOD)
    assert any("read_records(path, sheet)" in step for step in features.steps)
    assert any("event_id" in step for step in features.steps)
    # A concrete calculation method does not declare a business create/update intent.
    assert features.family == ""
    assert {
        "read_source_file",
        "preserve_source_records",
        "resolve_event_state",
        "separate_attempts",
        "calculate_metric",
        "validate_result",
    } <= method_capabilities(features)


def test_ordinary_computation_method_precedes_same_task_completion_events():
    method = item("calculation", WORKBOOK_METHOD, 0.62, "entities")
    old = item("done", "Summary: 订单统计文档已完成并保存，成功率为80%。", 0.95, "events")
    query = "编写订单统计文档，读取原工作簿，保留原字段与原行号，按期末去重并区分首次重试，计算成功率并核验结果。"
    rows, trace = select([old, method], query)
    assert rows[0] == method
    assert trace[method.uri]["reason"] == "applicable_evidence_method"
    assert "preserve_source_records" in trace[method.uri]["added_capabilities"]
    assert trace[method.uri]["memory_family"] == ""
    assert old in rows


def test_ordinary_methods_add_only_missing_evidence_not_duplicate_instructions():
    calculation = WORKBOOK_METHOD.replace("复算", "计算").replace(
        "，校验原始记录的集合归属和覆盖", ""
    )
    computation = item("calculation", calculation, 0.62, "entities")
    duplicate = item("copy", calculation, 0.80, "entities")
    workflow = item(
        "checking",
        "- 使用 json.load(file) 读取结果 JSON，核验分子、分母及原始记录的集合覆盖。",
        0.58,
        "preferences",
    )
    old = item("done", "Summary: 订单统计文档已完成。", 0.95, "events")
    query = "编写订单统计文档，逐章保存；读取原工作簿，保留原字段与原行号，区分首次重试，计算指标并核验结果。"
    rows, trace = select([old, computation, workflow, duplicate], query)
    assert rows[:2] == [duplicate, workflow]
    assert trace[workflow.uri]["added_capabilities"] == ["validate_result"]
    assert trace[computation.uri]["reason"] == "relevance_fallback"


def test_method_in_user_quotation_is_not_a_provided_procedure():
    quoted = item(
        "request",
        "Summary: 用户要求生成工作簿统计报告。\n[user]: 使用 read_rows(sheet) 保留 raw 和 source_row，计算分子分母并核验结果。",
        0.95,
        "events",
    )
    method = item("calculation", WORKBOOK_METHOD, 0.62, "entities")
    rows, trace = select(
        [quoted, method], "编写统计报告，读取工作簿计算指标，并保留原字段与原行号。"
    )
    assert rows[0] == method
    assert memory_features(quoted.abstract).steps == ()
    assert trace[quoted.uri]["added_capabilities"] == []


def test_method_in_observed_agent_event_can_provide_calculation_evidence():
    event = item(
        "observed",
        "Summary: 订单报告编写完成。\n[user]: 请生成工作簿报告。\n"
        "[report-agent]: 使用 read_rows(sheet) 返回 raw 全部原字段和 source_row 原行号；计算分子分母并核验结果。",
        0.62,
        "events",
    )
    old = item("done", "Summary: 统计报告已完成，成功率为80%。", 0.95, "events")
    rows, trace = select([old, event], "编写统计报告，保留全部原字段与原行号，计算指标并核验结果。")
    assert rows[0] == event
    assert trace[event.uri]["reason"] == "applicable_evidence_method"


def test_calculation_not_requested_does_not_promote_a_computation_method():
    method = item("calculation", WORKBOOK_METHOD, 0.62, "entities")
    fact = item("fact", "Summary: 订单报告模板采用六章目录。", 0.95, "entities")
    rows, trace = select([fact, method], "编写订单报告，不要读取工作簿或计算指标。")
    assert rows == [fact, method]
    assert trace[method.uri]["reason"] == "relevance_fallback"


def test_history_lookup_keeps_result_ahead_of_reusable_computation_method():
    method = item("calculation", WORKBOOK_METHOD, 0.62, "entities")
    old = item("done", "Summary: 订单统计报告已完成，成功率80%。", 0.95, "events")
    rows, trace = select([method, old], "查询上次订单统计报告的结果")
    assert rows[0] == old
    assert trace[old.uri]["reason"] == "historical_observed_result"


def test_native_code_fence_preserves_complete_callable_method_evidence():
    block = """```python
rows = pd.read_csv("orders.csv")
# calculate the ratio from the numerator and denominator
ratio = numerator / denominator
assert 0 <= ratio <= 1
```"""
    source = "# CSV processing\n\n" + block
    features = memory_features(source)
    assert block in features.steps
    assert {"read_source_file", "calculate_metric", "validate_result"} <= method_capabilities(
        features
    )
    assert source.endswith(block)


def test_unfinished_or_negated_code_does_not_establish_a_method():
    assert memory_features('```python\nrows = pd.read_csv("orders.csv")').steps == ()
    denied = memory_features("- 禁止使用 read_rows(sheet) 读取工作簿并计算指标。")
    assert denied.steps == ()
    assert method_capabilities(denied) == set()


def test_completion_summary_with_results_and_filenames_is_not_a_method():
    text = (
        "Summary: 已完成工作簿统计报告，直接读取原工作簿，计算成功率80%，"
        "保存脚本 tmp/metric.py 和结果 tmp/metric.json。"
    )
    assert memory_features(text).steps == ()


def test_repeating_the_requested_writing_order_is_not_a_new_method_capability():
    text = "- 按章节逐章编写，每章完成后保存正文，再进入下一章。"
    assert memory_features(text).steps == ()
    assert method_capabilities(memory_features(text)) == set()


@pytest.mark.parametrize(
    "statement",
    [
        'rows = pd.read_csv("orders.csv")',
        'text = PdfReader("source.pdf").pages[0].extract_text()',
        'document = Document("source.docx")',
        'payload = json.load(open("source.json"))',
    ],
)
def test_literal_native_file_reader_calls_are_recognized(statement):
    features = memory_features("```python\n" + statement + "\n```")
    assert "read_source_file" in method_capabilities(features)
