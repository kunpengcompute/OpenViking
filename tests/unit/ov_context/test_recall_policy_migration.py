"""Recall policy executes in Reader over raw find/read, including long document tasks."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.integrations.langchain.ov_context.reader import MemoryReaderConfig, OVContextReader
from openviking_cli.client.identity import OVIdentity
from openviking_cli.retrieve.types import ContextType, FindResult, MatchedContext

PROMPT = "根据资料，为创建、更新、取消流程编写一份需求文档。\n\n## 1 新建规则\n## 2 更新规则\n## 3 取消与幂等\n最后核对并交付文件。"
IDENTITY = OVIdentity("tenant", "alice", "doc")


def fixture_client():
    event = MatchedContext(
        uri="viking://user/alice/memories/events/2026/11/10/task.md",
        context_type=ContextType.MEMORY,
        category="events",
        abstract="Summary: 用户下达编写需求文档任务。",
        score=0.99,
    )
    method = MatchedContext(
        uri="viking://user/alice/peers/doc/memories/skills/write.md",
        context_type=ContextType.MEMORY,
        category="skills",
        abstract="Skill: 文档编写\nBest for: 编写需求文档\nGuidelines:\n- 逐章保存，按原始记录独立计算。",
        score=0.7,
    )
    body = event.abstract + "\n2026-11-10 ChatLog:\n[user]: " + PROMPT
    client = SimpleNamespace(
        find=AsyncMock(return_value=FindResult(memories=[event, method], resources=[], skills=[])),
        read=AsyncMock(
            side_effect=lambda uri, **kwargs: body if uri == event.uri else method.abstract
        ),
        search=AsyncMock(side_effect=AssertionError("Reader must not use server policy assembly")),
    )
    return client, event, method


async def test_original_document_task_uses_local_policy_and_budget():
    client, event, method = fixture_client()
    reader = OVContextReader(
        client,
        MemoryReaderConfig(recall_detail="full", max_injected_chars=4000, recall_max_tokens=4000),
    )
    result = await reader.recall(PROMPT, identity=IDENTITY)
    assert result.status == "success" and result.count == 2
    assert method.uri in result.content and event.uri in result.content
    assert PROMPT not in result.content and "ChatLog" not in result.content
    assert result.stats["used_tokens"] <= 4000 and result.stats["used_chars"] <= 4000
    assert result.stats["selection"][0]["purpose"] == "execute"
    client.search.assert_not_awaited()
    assert all(call.kwargs["identity"] == IDENTITY for call in client.find.await_args_list)


@pytest.mark.parametrize(
    "ranking,query", [("openclaw", "回顾之前需求文档任务的执行结果"), ("score", PROMPT)]
)
async def test_history_and_explicit_score_keep_original_event_view(ranking, query):
    client, _, _ = fixture_client()
    result = await OVContextReader(
        client,
        MemoryReaderConfig(
            ranking=ranking, recall_detail="full", max_injected_chars=10000, recall_max_tokens=8000
        ),
    ).recall(query, identity=IDENTITY)
    assert result.status == "success"
    if ranking == "score":
        assert PROMPT not in result.content and "ChatLog" not in result.content
    else:
        assert "ChatLog" in result.content
    client.search.assert_not_awaited()


async def test_budget_does_not_overflow_after_local_rendering():
    client, _, _ = fixture_client()
    result = await OVContextReader(
        client,
        MemoryReaderConfig(recall_detail="full", recall_max_tokens=30, max_injected_chars=100),
    ).recall(PROMPT, identity=IDENTITY)
    assert result.count == 0
    client.search.assert_not_awaited()


async def test_context_recall_rejects_missing_invocation_identity_before_io():
    client, _, _ = fixture_client()
    result = await OVContextReader(client).recall(PROMPT)
    assert result.status == "error"
    client.find.assert_not_awaited()
    client.search.assert_not_awaited()


async def test_recall_has_no_server_assembly_dependency(monkeypatch):
    from openviking.retrieve.context_assembler import pipeline

    monkeypatch.setattr(
        pipeline,
        "assemble_context",
        AsyncMock(side_effect=AssertionError("Server assembly must be independent")),
    )
    client, _, _ = fixture_client()
    result = await OVContextReader(client).recall(PROMPT, identity=IDENTITY)
    assert result.status == "success" and result.count == 2
    pipeline.assemble_context.assert_not_awaited()


def test_middleware_has_no_direct_server_or_context_assembler_imports():
    import ast
    from pathlib import Path

    import openviking.integrations.langchain.ov_context as package

    folder = Path(package.__file__).parent
    for name in (
        "recall_assembly.py",
        "recall_content.py",
        "recall_ranking.py",
        "recall_selection.py",
    ):
        for node in ast.walk(ast.parse((folder / name).read_text())):
            modules = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [item.name for item in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            assert not any(
                value.startswith(
                    (
                        "openviking.retrieve.context_assembler",
                        "openviking.server",
                        "openviking.storage.expr",
                        "openviking.models.embedder",
                    )
                )
                for value in modules
            ), (name, modules)
