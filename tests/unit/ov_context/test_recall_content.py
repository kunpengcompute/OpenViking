"""Memory views preserve source facts and methods while keeping metadata internal."""

import json

import pytest

from openviking.integrations.langchain.ov_context.recall_content import memory_view, overview


def stored(body, **fields):
    return body + "\n\n<!-- MEMORY_FIELDS\n" + json.dumps(fields) + "\n-->"


def test_tool_preview_keeps_usage_parameters_and_failure_lesson():
    body = (
        "Tool: browser\n\n- When to use: 填写网页表单。\n"
        "- Optimal params: batch --json 批量填写；先 snapshot 获取 refs。\n"
        "- Common failures: 旧 refs 会导致字段错位。\n\n"
        "## Guidelines\n- 保存前核对字段。\n### Good Cases\n- 多字段一次填写。"
    )
    view = memory_view(stored(body, user_id="private-user", ranges="0-10"))
    result = overview(view["content"], view)
    assert "batch --json" in result and "snapshot" in result
    assert "填写网页表单" in result and "旧 refs 会导致字段错位" in result
    assert "private-user" not in result and "MEMORY_FIELDS" not in result
    assert len(result) <= 400


def test_skill_preview_includes_guideline_actions_beyond_introductory_paragraph():
    body = (
        "Skill: browser\n\n" + "介绍文字" * 150 + "\n\n"
        "- Best for: 网页交互。\n- Recommended flow: 读取页面，再填写并核对。\n\n"
        "## Guidelines\n- 使用 batch 批量填写。\n- 保存前检查结果。\n"
        "### Bad Cases\n- 页面跳转后沿用旧 refs。"
    )
    result = overview(body)
    assert "网页交互" in result and "batch" in result and "旧 refs" in result
    assert "介绍文字" not in result


def test_trajectory_preview_keeps_conditions_steps_and_applicability():
    body = (
        "# Batch form\n- Preconditions:\n  1. User is logged in.\n"
        "- Procedure:\n  1. Read fresh refs.\n  2. Run batch with args: exact values.\n"
        "- Negative Applicability:\n  - Editing existing records.\n"
        "- Anti-patterns:\n  - Reusing stale refs."
    )
    result = overview(body)
    for fact in ["User is logged in", "fresh refs", "args: exact values", "Editing", "stale refs"]:
        assert fact in result


def test_structured_fields_supply_methods_when_template_omits_them():
    view = memory_view(
        stored(
            "Tool: browser",
            when_to_use="Web forms.",
            optimal_params="Use batch --json with fresh refs.",
            common_failures="Stale refs fill incorrect fields.",
            source_session_id="private-source",
            call_count=100,
        )
    )
    result = overview(view["content"], view)
    assert "batch --json" in result and "Stale refs" in result
    assert "private-source" not in result and "call_count" not in result


@pytest.mark.parametrize("heading", ["## Summary", "# Summary", "## 摘要"])
def test_explicit_summary_is_selected_once(heading):
    result = overview(f"{heading}\nThe job completed.\n\n## Details\nOther facts.")
    assert result.count("The job completed.") == 1
    assert result.count(heading) == 1


@pytest.mark.parametrize(
    "text",
    ["", "## Guidelines\n### Good Cases\n### Bad Cases", "Tool: browser", "Recommended flow:"],
)
def test_heading_or_name_only_preview_is_empty(text):
    assert overview(text) == ""


def test_plain_factual_paragraph_survives():
    assert overview("# Event\n\nAlice confirmed the meeting for Friday.") == (
        "Alice confirmed the meeting for Friday."
    )


def test_large_command_is_omitted_as_a_whole():
    command = "  - execute --json '" + "a" * 500 + "'"
    result = overview("## Guidelines\n" + command + "\n- Check the result.\n")
    assert "execute --json" not in result and "a" * 100 not in result
    assert "Check the result." in result and len(result) <= 400


@pytest.mark.parametrize("fence", ["```", "~~~"])
def test_large_fenced_json_is_omitted_as_a_whole(fence):
    command = fence + 'json\n[\n  {"value":"' + "x" * 500 + '"}\n]\n' + fence
    result = overview("## Guidelines\n" + command + "\n- Check the result.")
    assert fence not in result and '"value"' not in result
    assert "Check the result." in result


def test_short_fenced_command_remains_complete():
    command = '```json\n[["fill", "@e1", "value"]]\n```'
    result = overview("## Guidelines\n" + command)
    assert command in result


@pytest.mark.parametrize("fence", ["```json\n```", '```json\n[["fill", "@e1"'])
def test_empty_or_unfinished_fence_is_not_a_useful_preview(fence):
    assert overview("## Guidelines\n" + fence) == ""


@pytest.mark.parametrize(
    "comment",
    [
        '<!-- MEMORY_FIELDS\n{"source_id":"private"}\n-->',
        '<!-- MEMORY_FIELDS\n{"source_id":"private"',
        '<!-- MEMORY_FIELDS\n{"broken JSON}\n-->',
    ],
)
def test_internal_comments_are_cleaned_including_truncated_index_excerpts(comment):
    assert memory_view("A confirmed fact.\n\n" + comment)["content"] == "A confirmed fact."
