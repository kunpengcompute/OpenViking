"""I01–I09 over deployed OV, real Ark, Chat HTTP/SSE and durable SQLite."""

from __future__ import annotations

import asyncio
import json
import re
from copy import deepcopy

import httpx
import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from examples.deepagents_ov_context.chat.config import ModelConfig
from examples.deepagents_ov_context.chat.model import ArkChatOpenAI
from openviking.integrations.langchain.ov_context.assembler import validate_tool_pairs
from openviking.integrations.langchain.ov_context.errors import (
    ContextReadError,
    IncompleteWriteError,
)
from openviking.integrations.langchain.ov_context.reader import MemoryReaderConfig, OVContextReader
from openviking.integrations.langchain.ov_context.state import OVContext
from openviking.integrations.langchain.ov_context.writer import OVMessageWriter, WriterConfig
from openviking_cli.client.http import AsyncHTTPClient
from openviking_cli.exceptions import NotFoundError, PermissionDeniedError, UnavailableError

from .support import ChatProcess, NetworkGate, events


def history(rounds=4, parallel=2):
    messages = [HumanMessage(id="current", content="核对实验记录，记住本次任务目标是检查回执。")]
    for i in range(rounds):
        calls = [
            {"name": "receipt", "id": f"c-{i}-{j}", "args": {"batch": i, "part": j}}
            for j in range(parallel)
        ]
        messages.append(AIMessage(id=f"a-{i}", content=f"核对第{i}组回执", tool_calls=calls))
        messages.extend(
            ToolMessage(
                id=f"t-{i}-{j}",
                tool_call_id=c["id"],
                name="receipt",
                content=f"第{i}组{j}号回执：交付成功，金额为{i * 10 + j}元。",
            )
            for j, c in enumerate(reversed(calls))
        )
    messages.append(AIMessage(id="final", content="已核对，任务可以继续。"))
    return messages


async def test_i01_chat_rebuild_keeps_checkpoint_and_model_history(live):
    """Every /api/chat creates DeepAgents anew, reusing persisted OVContext."""
    marker = live.prefix + "-secret"
    async with ChatProcess(live) as chat:
        session = await chat.new()
        sid = session["session_id"]
        await chat.run(sid, f"记住校验码{marker}，只回复收到。不要调用工具。")
        first = await chat.state(sid)
        await chat.run(sid, "刚才的校验码是什么？只回复校验码，不要调用工具。")
        second = await chat.state(sid)
        before, after = first["ov_context"], second["ov_context"]
        assert before["write_index"] == len(first["messages"]) == 2
        assert after["write_index"] == len(second["messages"]) == 4
        assert after["owner"] == before["owner"]
        assert after["completed_runs"] == before["completed_runs"] + 1
        assert after["run_id"] is None and after["original_user"] is None
        assert marker in str(second["messages"][-1])
        remote = await live.sdk.get_session_context(sid, identity=live.sessions[-1][1])
        assert len(remote["messages"]) == 4
        # Observability and provider usage survive page reconnection.
        runs = (await chat.http.get(f"/api/sessions/{sid}/runs")).json()
        assert len(runs) == 2
        for run in runs:
            trace = (await chat.http.get(f"/api/sessions/{sid}/runs/{run['run_id']}")).json()
            assert any(e["event"] == "model_input" for e in trace["events"])
            assert run["statistics"]["total_tokens"] > 0
        rows = [
            json.loads(line)
            for line in (chat.data / "logs" / "runs.jsonl").read_text().splitlines()
        ]
        inputs = {r["model_call_id"]: r for r in rows if r["event"] == "model_input"}
        usage = {r["model_call_id"]: r for r in rows if r["event"] == "model_usage"}
        assert len(inputs) == len(usage) == sum(r["statistics"]["model_calls"] for r in runs)
        for call_id, row in inputs.items():
            assert row["data"]["estimated_input_tokens"] <= row["data"]["input_limit"]
            assert usage[call_id]["data"]["total_tokens"] > 0
        assert sum(r["data"]["total_tokens"] for r in usage.values()) == sum(
            r["statistics"]["total_tokens"] for r in runs
        )


async def test_i02_real_process_restart_retains_config_and_native_messages(live):
    async with ChatProcess(live) as chat:
        session = await chat.new()
        sid = session["session_id"]
        await chat.run(
            sid, '调用debug_echo工具一次，text参数为"报告编号R20261029"。拿到结果后只回复收到。'
        )
        old = await chat.state(sid)
        tools = [m for m in old["messages"] if m["type"] == "tool"]
        assert tools and all(m["tool_call_id"] for m in tools)
        detail = (await chat.http.get(f"/api/sessions/{sid}")).json()
        snapshot = detail["session"]["config_snapshot"]
        old_pid = chat.process.pid
        await chat.stop(kill=True)
        await chat.start(ratio="0.4")
        assert chat.process.pid != old_pid
        assert (await chat.state(sid))["ov_context"] == old["ov_context"]
        current = (await chat.http.get(f"/api/sessions/{sid}")).json()
        assert current["session"]["config_snapshot"] == snapshot
        fresh = await chat.new(user="bob")
        assert fresh["config_snapshot"]["context_config"]["base_context_ratio"] == 0.4
        await chat.run(sid, "报告编号是什么？只回复编号，不使用工具。")
        state = await chat.state(sid)
        assert "R20261029" in str(state["messages"][-1])
        assert state["ov_context"]["write_index"] == len(state["messages"])


@pytest.mark.parametrize("window", ["accepted", "terminal", "uncheckpointed"])
async def test_i03_recreated_engine_reconciles_real_commit(live, window):
    context, identity = await live.new()
    engine = live.engine(keep=0)
    messages = history(rounds=2, parallel=1)
    await engine.write(context, messages, identity=identity)
    before = deepcopy(context.to_state())
    assert await engine.commit(context, identity=identity, trigger="capacity", pending_tokens=1000)
    pending = deepcopy(context.pending_commit)
    assert pending and pending["task_id"]
    if window == "terminal":
        await live.terminal(
            context.session_id,
            identity,
            {
                "task_id": pending["task_id"],
                "archive_uri": f"viking://session/{context.session_id}/history/{pending['target_archive_id']}",
            },
        )
    restored = OVContext.from_state(before if window == "uncheckpointed" else context.to_state())
    restored.end_run()
    new_engine = live.engine(keep=0)
    new_engine.recover_commit(restored, persisted=True)
    if restored.pending_commit is not None:
        assert restored.pending_commit["task_id"] == pending["task_id"]
    await new_engine.wait_summary(restored, identity=identity)
    assert restored.archive_id == pending["target_archive_id"]
    assert restored.pending_commit is None
    tasks = await live.sdk.list_tasks(
        task_type="session_commit", resource_id=context.session_id, identity=identity, limit=100
    )
    assert len([t for t in tasks if t["task_id"] == pending["task_id"]]) == 1


async def test_i03_uncheckpointed_write_allows_duplicate_as_designed(live):
    context, identity = await live.new()
    raw = [HumanMessage(id="u", content="已发送但游标未保存的消息")]
    writer = OVMessageWriter(live.sdk)
    first = await writer.write(context.session_id, raw, 0, "old", identity=identity)
    second = await writer.write(context.session_id, raw, 0, "new", identity=identity)
    assert first.write_index == second.write_index == 1
    view = await live.sdk.get_session_context(context.session_id, identity=identity)
    assert len(view["messages"]) == 2


async def test_i04_long_parallel_tools_compress_and_read_back_exact_originals(live):
    context, identity = await live.new()
    calls = [{"name": "report", "id": f"parallel-{i}", "args": {"i": i}} for i in range(3)]
    raw = [
        HumanMessage(id="u", content="核对三个部门的日报"),
        AIMessage(id="a", content="", tool_calls=calls),
    ]
    raw.extend(
        ToolMessage(
            id=f"t-{i}",
            tool_call_id=call["id"],
            name="report",
            content=(f"部门{i} 独立记录 交付数据12345\n" * 1500) + f"END_MARKER_{i}",
            artifact={"local": i},
        )
        for i, call in enumerate(reversed(calls))
    )
    before = deepcopy(raw)
    writer = OVMessageWriter(live.sdk)
    result = await writer.write(context.session_id, raw, 0, identity=identity)
    assert raw == before and result.write_index == len(raw)
    assert result.model_messages[:2] == raw[:2]
    for original, short in zip(raw[2:], result.model_messages[2:], strict=True):
        assert short.tool_call_id == original.tool_call_id
        assert short.artifact == original.artifact
        assert len(short.content) < len(original.content)
        ref = re.search(r"viking://[^\s\]\)<>\"']+/tool-results/[^\s\]\)<>\"']+", short.content)
        assert ref, short.content[:200]
        result_id = ref[0].rstrip(".,，。").rsplit("/", 1)[-1]
        restored = ""
        while len(restored) < len(original.content):
            part = await live.sdk.read_tool_result(
                context.session_id, result_id, offset=len(restored), limit=20000, identity=identity
            )
            text = part["content"]
            assert text
            restored += text
        assert restored == original.content
    validate_tool_pairs(result.model_messages)
    # A repeated Hook delta performs no second HTTP write.
    previous = len(live.wire)
    repeated = await writer.write(context.session_id, raw, result.write_index, identity=identity)
    assert repeated.model_messages == [] and len(live.wire) == previous


async def test_i04_model_calls_real_readback_tool_after_restore(live):
    async with ChatProcess(live) as chat:
        session = await chat.new()
        sid = session["session_id"]
        identity = live.sessions[-1][1]
        await live.sdk.create_session(
            sid,
            identity=identity,
            memory_policy={"self": {"enabled": False}, "peer": {"enabled": False}},
        )
        marker = "READBACK_" + live.prefix
        text = "部门流水记录，金额12345元。\n" * 1800 + marker
        raw = [
            HumanMessage(id="seed-u", content="查阅部门流水"),
            AIMessage(
                id="seed-a",
                content="",
                tool_calls=[{"name": "report", "args": {}, "id": "seed-call"}],
            ),
            ToolMessage(id="seed-t", tool_call_id="seed-call", name="report", content=text),
        ]
        result = await OVMessageWriter(live.sdk).write(sid, raw, 0, identity=identity)
        ref = re.search(
            r"viking://[^\s\]\)<>\"']+/tool-results/[^\s\]\)<>\"']+",
            result.model_messages[-1].content,
        )
        assert ref
        reference = ref[0].rstrip(".,，。")
        response = await chat.http.post(f"/api/sessions/{sid}/restore")
        response.raise_for_status()
        await chat.run(
            sid,
            f'请调用openviking_tool_result_read一次，ref="{reference}"，offset={len(text) - len(marker)}，limit=100。只根据工具返回给出末尾校验标记，不使用其他工具。',
        )
        state = await chat.state(sid)
        reads = [
            m
            for m in state["messages"]
            if m["type"] == "tool" and m.get("name") == "openviking_tool_result_read"
        ]
        assert reads and marker in str(reads[-1]["content"])
        assert marker in str(state["messages"][-1]["content"])
        assert state["ov_context"]["write_index"] == len(state["messages"])


async def test_i04_protected_readback_and_over_100_messages(live):
    context, identity = await live.new()
    messages = []
    for i in range(51):
        messages.extend(
            [
                HumanMessage(id=f"u{i}", content=f"问题{i}"),
                AIMessage(id=f"a{i}", content=f"答复{i}"),
            ]
        )
    call = {"id": "readback", "name": "openviking_tool_result_read", "args": {}}
    messages += [
        AIMessage(id="read-a", content="", tool_calls=[call]),
        ToolMessage(
            id="read-t",
            tool_call_id="readback",
            name=call["name"],
            content="精确原文123456\n" * 1000,
        ),
    ]
    result = await OVMessageWriter(live.sdk, WriterConfig(batch_size=100)).write(
        context.session_id, messages, 0, identity=identity
    )
    assert result.write_index == len(messages)
    assert result.model_messages[-1] is messages[-1]
    view = await live.sdk.get_session_context(
        context.session_id, token_budget=256000, identity=identity
    )
    restored = await OVContextReader(live.sdk).exchange_context(
        session_id=context.session_id, identity=identity, token_budget=256000
    )
    assert len(restored.messages) == len(messages)
    assert len(view["messages"]) >= 103
    assert [m.content for m in restored.messages[:102]] == [m.content for m in messages[:102]]
    validate_tool_pairs(restored.messages)


async def test_i05_shared_sdk_scopes_parallel_account_user_peer_sessions(live):
    originals = dict(live.pool.headers)
    contexts = await asyncio.gather(
        *[
            live.new(account=f"{live.prefix}-{a}", user=f"u{u}", peer=f"p{p}")
            for a in range(2)
            for u in range(2)
            for p in range(2)
        ]
    )

    async def execute(context, identity):
        marker = "/".join(context.owner)
        raw = [
            HumanMessage(id="u", content=marker),
            AIMessage(id="a", content="已保存"),
            HumanMessage(id="u2", content=f"继续处理{marker}"),
        ]
        engine = live.engine(keep=1)
        await engine.write(context, raw, identity=identity)
        assert await engine.commit(
            context, identity=identity, trigger="capacity", pending_tokens=1000
        )
        await engine.wait_summary(context, identity=identity)
        snapshot = await OVContextReader(live.sdk).exchange_context(
            session_id=context.session_id, identity=identity
        )
        assert marker in str(snapshot.messages)
        assert context.archive_id is not None
        return context, identity

    completed = await asyncio.gather(*(execute(c, i) for c, i in contexts))
    for context, identity in completed:
        another = next(i for _, i in completed if i.account_id != identity.account_id)
        with pytest.raises((NotFoundError, PermissionDeniedError)):
            await live.sdk.get_session(context.session_id, identity=another, auto_create=False)
        rows = [r for r in live.wire if context.session_id in r["path"] and r["method"] == "POST"]
        assert rows
        assert all(r["actor-peer"] == identity.actor_peer_id for r in rows)
        assert all(
            (r["account"], r["user"]) == (identity.account_id, identity.user_id) for r in rows
        )
    assert dict(live.pool.headers) == originals


@pytest.mark.parametrize("keep", [0, 1, 3])
async def test_i06_real_archive_refresh_preserves_current_anchor_and_parallel_tail(live, keep):
    context, identity = await live.new()
    raw = history()
    engine = live.engine(keep=keep)
    await engine.write(context, raw, identity=identity)
    context.begin_run("run1", raw[0])
    context.inject_recall("本轮专用参考记忆XYZ", "success")
    plan = engine.assembler.plan_refresh(context, keep_recent_count=keep)
    assert await engine.commit(context, identity=identity, trigger="capacity", pending_tokens=1000)
    await engine.wait_summary(context, identity=identity)
    assert context.archive_id and context.messages[0].id.startswith("ov_summary:")
    assert sum(m.id == "current" for m in context.messages) == 1
    assert sum("本轮专用参考记忆XYZ" in str(m.content) for m in context.messages) == 1
    assert (
        [m.id for m in context.messages[-len(plan.recent_messages) :]]
        == [m.id for m in plan.recent_messages]
        if plan.recent_messages
        else True
    )
    validate_tool_pairs(context.messages)
    remote = await live.sdk.get_session_context(context.session_id, identity=identity)
    assert "本轮专用参考记忆XYZ" not in json.dumps(remote, ensure_ascii=False)
    assert remote["latest_finished_archive_id"] == context.archive_id
    model_config = ModelConfig.load(live.config.model_config)
    model = ArkChatOpenAI(
        model=model_config.chat_model, api_key=model_config.api_key, base_url=model_config.api_base
    )
    request = ModelRequest(
        model=model, messages=[], system_message=SystemMessage(content="SYS_FIXED")
    )
    prepared = engine.assembler.prepare(request, context)
    assert prepared.input_tokens <= prepared.budget.input_limit
    assert prepared.request.system_message.content.count("SYS_FIXED") == 1
    assert prepared.request.system_message.content.count("[OpenViking Context Guide]") == 1
    old = deepcopy(context.to_state())
    await engine.check_summary(context, identity=identity)
    assert context.to_state() == old


async def test_i07_real_extraction_then_recall_in_new_session(live):
    context, identity = await live.new(memory=True)
    city = "杭州"
    raw = [
        HumanMessage(
            id="u",
            content=f"请记住我的固定工作偏好：我常驻{city}，所有周报采用中文，金额保留两位小数。",
        ),
        AIMessage(id="a", content="已记住你的工作偏好：常驻杭州，周报中文，金额两位小数。"),
    ]
    await OVMessageWriter(live.sdk).write(context.session_id, raw, 0, identity=identity)
    receipt = await live.sdk.commit_session(
        context.session_id, keep_recent_count=0, identity=identity
    )
    await live.terminal(context.session_id, identity, receipt)
    another, _ = await live.new(
        account=identity.account_id, user=identity.user_id, peer=identity.actor_peer_id
    )
    reader = OVContextReader(
        live.sdk,
        MemoryReaderConfig(
            target_uris=("viking://user/memories",),
            score_threshold=0,
            recall_timeout=60,
            recall_max_tokens=2000,
            context_timeout=60,
            max_injected_chars=8000,
        ),
    )
    recall = await reader.recall(
        "我的周报使用什么语言，金额保留几位小数，常驻哪个城市？",
        identity=identity,
        session_id=another.session_id,
    )
    assert recall.status == "success" and recall.count > 0
    assert any(word in recall.content for word in (city, "中文", "小数")), recall.content
    assert len(recall.content) <= reader.config.max_injected_chars
    another.messages = [HumanMessage(id="new-user", content="请帮我写周报")]
    another.begin_run("new-run", another.messages[0])
    another.inject_recall(recall.content, recall.status)
    view = another.messages[0]
    another.inject_recall("不应再次注入", "success")
    assert another.messages[0] is view
    assert str(another.messages).count("[参考记忆]") == 1


@pytest.mark.parametrize("stage", ["read", "write", "commit"])
async def test_i08_real_network_outage_preserves_cursor_and_recovers(live, stage):
    context, identity = await live.new()
    async with NetworkGate(live.config.url) as gate:
        async with httpx.AsyncClient(timeout=1) as pool:
            sdk = AsyncHTTPClient(
                url=gate.url,
                api_key=live.config.key_file.read_text().strip(),
                auth_mode="trusted",
                profile_enabled=False,
                http_client=pool,
            )
            await sdk.initialize()
            engine = live.engine(sdk=sdk, keep=0)
            raw = [
                HumanMessage(id="u", content="网络中断前已保存"),
                AIMessage(id="a", content="收到"),
            ]
            await engine.write(context, raw, identity=identity)
            before = deepcopy(context.to_state())
            gate.cut()
            expected = {
                "read": ContextReadError,
                "write": IncompleteWriteError,
                "commit": (httpx.TransportError, UnavailableError),
            }[stage]
            with pytest.raises(expected):
                if stage == "read":
                    await engine.refresh(context, identity=identity)
                elif stage == "write":
                    await engine.write(
                        context,
                        raw + [HumanMessage(id="u2", content="待写增量")],
                        identity=identity,
                    )
                else:
                    await engine.commit(
                        context, identity=identity, trigger="capacity", pending_tokens=1000
                    )
            assert context.messages == before["messages"]
            assert context.write_index == before["write_index"]
            assert context.archive_id == before["archive_id"]
            gate.blocked = False
            raw.append(HumanMessage(id="u-new", content="服务恢复后以新Run继续"))
            replacement = live.engine(sdk=sdk)
            await replacement.write(context, raw, identity=identity)
            assert context.write_index == len(raw)
            remote = await live.sdk.get_session_context(context.session_id, identity=identity)
            assert "服务恢复后以新Run继续" in json.dumps(remote, ensure_ascii=False)
            await sdk.close()


async def test_i09_cancel_cleans_checkpoint_and_next_run_uses_new_instance(live):
    async with ChatProcess(live) as chat:
        session, independent = await asyncio.gather(chat.new(), chat.new(user="bob"))
        sid, other = session["session_id"], independent["session_id"]
        running = asyncio.create_task(
            chat.http.post(
                "/api/chat",
                json={
                    "session_id": sid,
                    "message": "请用中文详细解释分布式系统的十个重要设计原则。不要调用工具。",
                },
            )
        )
        try:
            for _ in range(100):
                response = await chat.http.get(f"/api/sessions/{sid}")
                if response.json()["busy"]:
                    break
                await asyncio.sleep(0.05)
            else:
                pytest.fail("Real Run did not enter busy state")
            conflict = await chat.http.post(
                "/api/chat", json={"session_id": sid, "message": "并发请求"}
            )
            assert conflict.status_code == 409
            separate = asyncio.create_task(chat.run(other, "只回复OK，不调用工具"))
            response = await chat.http.post(f"/api/sessions/{sid}/cancel")
            assert response.json()["cancelled"]
            cancelled = await running
            assert any(kind == "cancelled" for kind, _ in events(cancelled.text))
            await separate
        finally:
            if not running.done():
                running.cancel()
            await asyncio.gather(running, return_exceptions=True)
        state = await chat.state(sid)
        if "ov_context" in state:
            assert state["ov_context"]["run_id"] is None
            assert state["ov_context"]["original_user"] is None
        response = await chat.http.post(f"/api/sessions/{sid}/restore")
        response.raise_for_status()
        await chat.run(sid, "现在是新的一轮，只回复恢复成功，不调用工具。")
        assert (await chat.state(sid))["ov_context"]["run_id"] is None


async def test_i08_chat_fail_open_checks_real_model_input_budget(live):
    async with NetworkGate(live.config.url) as gate:
        async with ChatProcess(live, ov_url=gate.url) as chat:
            session = await chat.new()
            sid = session["session_id"]
            await live.sdk.create_session(
                sid,
                identity=live.sessions[-1][1],
                memory_policy={"self": {"enabled": False}, "peer": {"enabled": False}},
            )
            gate.cut()
            result = await chat.run(sid, "只回复OFFLINE_OK，不调用工具")
            assert result[-1][1]["context"]["ov_degraded"] is True
            inputs = [data for kind, data in result if kind == "model_input"]
            assert inputs
            assert all(data["estimated_input_tokens"] <= data["input_limit"] for data in inputs)
            gate.blocked = False
            restored = await chat.http.post(f"/api/sessions/{sid}/restore")
            restored.raise_for_status()
            await chat.run(sid, "连接已恢复，只回复ONLINE_OK，不调用工具")
            state = await chat.state(sid)
            assert state["ov_context"]["write_index"] == len(state["messages"])
