"""Actual SIGKILL recovery, repeated Hook capacity compression and sustained load."""

from __future__ import annotations

import asyncio
import json
import os
import re
import statistics
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from examples.deepagents_ov_context.chat.config import ModelConfig
from examples.deepagents_ov_context.chat.model import ArkChatOpenAI
from openviking.integrations.langchain.middleware import OpenVikingContextMiddleware
from openviking.integrations.langchain.ov_context import (
    Metrics,
    OVContext,
    OVContextConfig,
    RunIdentity,
)
from openviking.integrations.langchain.ov_context.assembler import validate_tool_pairs
from openviking.integrations.langchain.ov_context.commit import CommitConfig
from openviking.integrations.langchain.ov_context.reader import MemoryReaderConfig, OVContextReader
from openviking.integrations.langchain.ov_context.state import OVPluginState
from openviking.integrations.langchain.ov_context.writer import OVMessageWriter

from .support import ChatProcess
from .test_deployed import history

pytestmark = pytest.mark.skipif(
    os.getenv("OV_LIVE_EXTENDED") != "1",
    reason="Set OV_LIVE_EXTENDED=1 on an isolated Docker validation environment",
)


def save(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def percentile(values, fraction):
    return sorted(values)[min(len(values) - 1, int((len(values) - 1) * fraction))]


def model(live):
    config = ModelConfig.load(live.config.model_config)
    return ArkChatOpenAI(
        model=config.chat_model,
        api_key=config.api_key,
        base_url=config.api_base,
        max_tokens=512,
        extra_body={"thinking": {"type": "disabled"}},
    )


async def test_i03_sigkill_keeps_acknowledged_storage_and_chat_checkpoint(live, ov_process):
    marker = "CRASH_" + live.prefix
    async with ChatProcess(live) as chat:
        session = await chat.new()
        sid, identity = session["session_id"], live.sessions[-1][1]
        await chat.run(sid, f'调用debug_echo一次，text参数为"{marker}"。拿到结果后只回复已保存。')
        before = await chat.state(sid)
        original = await live.sdk.get_session_context(sid, token_budget=256000, identity=identity)
        assert before["ov_context"]["write_index"] == len(before["messages"])
        assert any(m["type"] == "tool" for m in before["messages"])
        await ov_process.kill()
        try:
            async with httpx.AsyncClient(timeout=1) as client:
                with pytest.raises(httpx.TransportError):
                    await client.get(live.config.url + "/health")
        finally:
            await ov_process.start()
        persisted = await live.sdk.get_session_context(sid, token_budget=256000, identity=identity)
        assert persisted == original
        assert (await chat.state(sid))["ov_context"] == before["ov_context"]
        await chat.run(sid, "刚才工具返回的校验码是什么？只回复校验码，不调用工具。")
        after = await chat.state(sid)
        assert marker in after["messages"][-1]["content"]
        assert after["ov_context"]["write_index"] == len(after["messages"])
        save(
            live.directory / "crash-storage.json",
            {
                "messages_before": len(before["messages"]),
                "messages_after": len(after["messages"]),
                "server_restarts": ov_process.events,
            },
        )


async def checkpoint(path, thread, payload=None):
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        graph = StateGraph(OVPluginState)
        graph.add_node("snapshot", lambda state: {})
        graph.add_edge(START, "snapshot")
        graph.add_edge("snapshot", END)
        app = graph.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": thread}}
        if payload is not None:
            await app.ainvoke(payload, config)
        return (await app.aget_state(config)).values


@pytest.mark.parametrize("receipt_saved", [False, True])
async def test_i03_sigkill_during_real_commit_recovers_same_task(live, ov_process, receipt_saved):
    context, identity = await live.new()
    raw = history(rounds=4, parallel=2)
    raw.insert(0, HumanMessage(id="durable", content="待归档的工作记录。" * 1500))
    engine = live.engine(keep=1)
    await engine.write(context, raw, identity=identity)
    path = live.directory / "commit-checkpoints.sqlite3"
    thread = "crash-thread-" + context.session_id
    await checkpoint(path, thread, {"messages": raw, "ov_context": context.to_state()})
    assert await engine.commit(context, identity=identity, trigger="capacity", pending_tokens=10000)
    pending = deepcopy(context.pending_commit)
    assert pending and pending["task_id"] and pending.get("target_archive_id")
    if receipt_saved:
        await checkpoint(path, thread, {"ov_context": context.to_state()})
    task = await live.sdk.get_task(pending["task_id"], identity=identity)
    assert task["status"] in {"pending", "running"}, (
        "Commit finished before the intended crash window"
    )
    await ov_process.kill()
    try:
        # Re-open the SQLite file and create a different engine after process loss.
        saved = await checkpoint(path, thread)
        recovered = OVContext.from_state(saved["ov_context"])
    finally:
        await ov_process.start()
    replacement = live.engine(keep=1)
    replacement.recover_commit(recovered, persisted=True)
    if recovered.pending_commit:
        assert recovered.pending_commit["target_archive_id"] == pending["target_archive_id"]
    await replacement.wait_summary(recovered, identity=identity)
    assert recovered.archive_id == pending["target_archive_id"]
    assert recovered.pending_commit is None
    meta = await live.sdk.get_session(context.session_id, identity=identity)
    assert meta["commit_count"] == 1
    task = await live.sdk.get_task(pending["task_id"], identity=identity)
    assert task["status"] in {"completed", "done"}
    tasks = await live.sdk.list_tasks(
        task_type="session_commit", resource_id=context.session_id, limit=100, identity=identity
    )
    assert len(tasks) == 1 and tasks[0]["task_id"] == pending["task_id"]
    view = await live.sdk.get_session_context(context.session_id, identity=identity)
    assert view["latest_archive_overview"]
    assert view["latest_finished_archive_id"] == recovered.archive_id
    validate_tool_pairs(recovered.messages)
    save(
        live.directory / "crash-commit.json",
        {
            "receipt_saved": receipt_saved,
            "task": task,
            "archive_id": recovered.archive_id,
            "commit_count": meta["commit_count"],
            "events": ov_process.events,
        },
    )


async def test_i06_three_capacity_crossings_in_one_run_use_real_archives(live):
    cache, identity = await live.new()
    observations = []
    mw = OpenVikingContextMiddleware(
        client=live.sdk,
        message_writer=OVMessageWriter(live.sdk),
        context_reader=OVContextReader(
            live.sdk, MemoryReaderConfig(enabled=False, context_timeout=60)
        ),
        context_config=OVContextConfig(
            context_window=12000,
            output_reserve=512,
            safety_margin=256,
            base_context_ratio=0.45,
            capacity_ratio=0.7,
            commit_wait_timeout=300,
            poll_interval=0.2,
        ),
        commit_config=CommitConfig(256000, 0.7, 1000, 3),
        memory_policy={"self": {"enabled": False}, "peer": {"enabled": False}},
        metrics=Metrics(lambda event, **data: observations.append({"event": event, **data})),
    )
    runtime = SimpleNamespace(
        context=SimpleNamespace(
            identity=RunIdentity(*identity.session_key(cache.session_id), "long-capacity-run")
        )
    )
    state = {
        "messages": [
            HumanMessage(
                id="task",
                content="For CURRENT_TASK_ABC, return only a JSON object for the most recent pair of ledger results: latest_batch is the actual batch identifier (for example 0-3), amounts contains both receipt totals sorted ascending. Use the latest actual results on every call.",
            )
        ]
    }
    state.update(await mw.abefore_agent(state, runtime))
    context = OVContext.from_state(state["ov_context"])
    context.inject_recall("RECALL_ONLY_ABC：本轮核对参考", "success")
    state["ov_context"] = context.to_state()
    llm = model(live)
    requests = []
    returned = []
    usage = []
    rounds = []
    try:
        for cycle in range(3):
            for group in range(4):
                index = f"{cycle}-{group}"
                calls = [
                    {"name": "ledger", "id": f"c{index}-{j}", "args": {"batch": index, "part": j}}
                    for j in range(2)
                ]
                body = (
                    f"Batch {index}: checked sales entries, amounts and delivery receipts. " * 150
                )
                state["messages"].append(AIMessage(id=f"a{index}", content=body, tool_calls=calls))
                state["messages"].extend(
                    ToolMessage(
                        id=f"t{index}-{j}",
                        tool_call_id=call["id"],
                        name="ledger",
                        content=f"TAIL_{index}_{j}: receipt total {cycle * 1000 + group * 100 + j} confirmed",
                    )
                    for j, call in enumerate(reversed(calls))
                )
            state.update(await mw.abefore_model(state, runtime) or {})
            current = OVContext.from_state(state["ov_context"])
            request = ModelRequest(
                model=llm,
                messages=state["messages"],
                state=state,
                runtime=runtime,
                system_message=SystemMessage(
                    content="FIXED_SYSTEM_ABC: Ledger tool interactions are complete. Answer the current user with only JSON containing latest_batch and amounts from the most recent tool results. No additional tool calls are needed."
                ),
            )
            assembler = mw._ov_session_engine.assembler
            budget = assembler.budget(request)
            initial = assembler._system_request(request).override(messages=current.messages)
            before_tokens = assembler._count(initial)
            assert before_tokens >= budget.capacity_limit, "History did not reach capacity"

            async def handler(prepared, input_limit=budget.input_limit, round_id=cycle):
                requests.append(prepared)
                assert mw._ov_session_engine.assembler.check_input_budget(prepared) <= input_limit
                recent_tools = [m for m in prepared.messages if isinstance(m, ToolMessage)]
                assert len(recent_tools) >= 2
                assert {m.content for m in recent_tools[-2:]} == {
                    f"TAIL_{round_id}-3_{j}: receipt total {round_id * 1000 + 300 + j} confirmed"
                    for j in range(2)
                }
                result = await llm.ainvoke([prepared.system_message, *prepared.messages])
                usage.append(result.usage_metadata)
                returned.append(result)
                assert result.usage_metadata and result.usage_metadata["total_tokens"] > 0
                assert result.usage_metadata["input_tokens"] <= input_limit
                assert result.response_metadata.get("finish_reason") == "stop"
                body = re.search(r"\{.*\}", str(result.content), re.S)
                assert body, result.content
                answer = json.loads(body[0])
                assert answer["latest_batch"] == f"{round_id}-3"
                assert sorted(answer["amounts"]) == [round_id * 1000 + 300, round_id * 1000 + 301]
                return result

            response = await mw.awrap_model_call(request, handler)
            state.update(response.command.update)
            current = OVContext.from_state(state["ov_context"])
            assert current.archive_id == f"archive_{cycle + 1:03d}"
            assert current.pending_commit is None
            assert current.write_index == len(state["messages"])
            assert sum(m.id == "task" for m in current.messages) == 1
            assert sum("RECALL_ONLY_ABC" in str(m.content) for m in current.messages) == 1
            assert len({m.id for m in current.messages}) == len(current.messages)
            validate_tool_pairs(current.messages)
            assert requests[-1].system_message.content.count("FIXED_SYSTEM_ABC") == 1
            remote = await live.sdk.get_session_context(cache.session_id, identity=identity)
            assert remote["latest_finished_archive_id"] == current.archive_id
            assert "RECALL_ONLY_ABC" not in json.dumps(remote, ensure_ascii=False)
            rounds.append(
                {
                    "archive_id": current.archive_id,
                    "before_tokens": before_tokens,
                    "after_tokens": mw._ov_session_engine.assembler.check_input_budget(
                        requests[-1]
                    ),
                    "capacity_limit": budget.capacity_limit,
                    "input_limit": budget.input_limit,
                    "write_index": current.write_index,
                    "cache_count": len(current.messages),
                }
            )
            state["messages"].append(returned[-1])
            state.update(await mw.aafter_model(state, runtime) or {})
        state.update(await mw.aafter_agent(state, runtime))
        assert state["ov_context"]["run_id"] is None
        assert state["ov_context"]["write_index"] == len(state["messages"])
        accepted = [
            o
            for o in observations
            if o["event"] == "context.commit" and o.get("status") == "accepted"
        ]
        assert len(accepted) == 3 and all(o["trigger"] == "capacity" for o in accepted)
        meta = await live.sdk.get_session(cache.session_id, identity=identity)
        assert meta["commit_count"] == 3
        assert len(requests) == 3
    finally:
        save(
            live.directory / "capacity-rounds.json",
            {
                "rounds": rounds,
                "usage": usage,
                "events": observations,
                "model_responses": [m.model_dump(mode="json") for m in returned],
                "model_inputs": [
                    {
                        "system": r.system_message.model_dump(mode="json"),
                        "messages": [m.model_dump(mode="json") for m in r.messages],
                    }
                    for r in requests
                ],
            },
        )
        await mw.aclose()


async def test_i05_ten_minute_sustained_eight_session_load(live, ov_process):
    duration = int(os.environ.get("OV_LIVE_STRESS_SECONDS", "600"))
    assert 60 <= duration <= 3600
    live.pool.timeout = httpx.Timeout(5)
    contexts = await asyncio.gather(
        *[
            live.new(account=f"{live.prefix}-a{a}", user=f"u{u}", peer=f"p{p}")
            for a in range(2)
            for u in range(2)
            for p in range(2)
        ]
    )
    writer = OVMessageWriter(live.sdk)
    records = []
    samples = []
    counts = [0] * len(contexts)
    errors = []
    probes = []
    started = time.monotonic()
    stop = asyncio.Event()

    async def observe():
        while not stop.is_set():
            samples.append(await ov_process.metrics())
            try:
                await asyncio.wait_for(stop.wait(), 5)
            except TimeoutError:
                pass

    async def worker(number, context, identity):
        raw = []
        cursor = 0
        while time.monotonic() - started < duration:
            index = counts[number]
            marker = f"{context.session_id}-BATCH-{index}"
            call = {"name": "ledger", "id": f"{number}-{index}", "args": {"batch": index}}
            raw.extend(
                [
                    HumanMessage(id=f"u{index}", content=marker),
                    AIMessage(id=f"a{index}", content="核对", tool_calls=[call]),
                    ToolMessage(
                        id=f"t{index}",
                        tool_call_id=call["id"],
                        name="ledger",
                        content=marker + " amount=123.45",
                    ),
                ]
            )
            tick = time.monotonic()
            try:
                result = await writer.write(context.session_id, raw, cursor, identity=identity)
                cursor = result.write_index
                meta = await live.sdk.get_session(context.session_id, identity=identity)
                assert meta["message_count"] == len(raw)
                if index % 20 == 0:
                    view = await live.sdk.get_session_context(
                        context.session_id, token_budget=4000, identity=identity
                    )
                    assert marker in json.dumps(view, ensure_ascii=False)
                counts[number] += 1
                records.append(
                    {
                        "worker": number,
                        "elapsed": time.monotonic() - started,
                        "seconds": time.monotonic() - tick,
                    }
                )
            except Exception as exc:
                errors.append(
                    {
                        "worker": number,
                        "index": index,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                raise
            await asyncio.sleep(max(0, 1 - (time.monotonic() - tick)))
        return raw, cursor

    async def agent_probes():
        async with ChatProcess(live) as chat:
            sessions = await asyncio.gather(
                *(chat.new(user=f"probe-u{i // 2}", peer=f"p{i % 2}") for i in range(8))
            )
            round_index = 0
            while time.monotonic() - started < duration:

                async def run_probe(session, index=round_index):
                    marker = f"PROBE_{session['session_id']}_{index}"
                    tick = time.monotonic()
                    result = await chat.run(
                        session["session_id"],
                        f'调用debug_echo一次，text参数为"{marker}"。拿到结果后仅回复该标记。',
                    )
                    assert not result[-1][1]["context"]["ov_degraded"]
                    state = await chat.state(session["session_id"])
                    assert marker in state["messages"][-1]["content"]
                    assert any(
                        m["type"] == "tool" and m["content"] == marker for m in state["messages"]
                    )
                    probes.append(
                        {
                            "session": session["session_id"],
                            "round": index,
                            "seconds": time.monotonic() - tick,
                            "statistics": result[-1][1]["statistics"],
                        }
                    )

                await asyncio.gather(*(run_probe(session) for session in sessions))
                round_index += 1
                try:
                    await asyncio.wait_for(
                        stop.wait(), min(120, max(0.01, duration - (time.monotonic() - started)))
                    )
                except TimeoutError:
                    pass

    baseline = await ov_process.metrics()
    monitor = asyncio.create_task(observe())
    tasks = [
        asyncio.create_task(worker(i, c, identity)) for i, (c, identity) in enumerate(contexts)
    ]
    tasks.append(asyncio.create_task(agent_probes()))
    results = []
    try:
        results = (await asyncio.gather(*tasks))[:-1]
        elapsed = time.monotonic() - started
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        stop.set()
        await monitor
        save(
            live.directory / "stress-progress.json",
            {
                "duration": duration,
                "elapsed": time.monotonic() - started,
                "counts": counts,
                "requests": records,
                "errors": errors,
                "resources": samples,
                "baseline": baseline,
                "agent_probes": probes,
            },
        )
    latencies = [r["seconds"] for r in records]
    first = [r["seconds"] for r in records if r["elapsed"] <= 60]
    last = [r["seconds"] for r in records if r["elapsed"] >= duration - 60]
    # Match the late history size: perform the same write/read cycle serially.
    # History keeps growing during sustained load, so empty-vs-full drift alone
    # cannot identify contention or resource leakage.
    controls = []
    for number, ((context, identity), (raw, cursor)) in enumerate(
        zip(contexts, results, strict=True)
    ):
        for index in range(5):
            marker = f"CONTROL_{context.session_id}_{index}"
            call = {"name": "ledger", "id": f"control-{number}-{index}", "args": {"batch": index}}
            raw.extend(
                [
                    HumanMessage(id=f"control-u{index}", content=marker),
                    AIMessage(id=f"control-a{index}", content="核对", tool_calls=[call]),
                    ToolMessage(
                        id=f"control-t{index}",
                        tool_call_id=call["id"],
                        name="ledger",
                        content=marker + " amount=123.45",
                    ),
                ]
            )
            tick = time.monotonic()
            result = await writer.write(context.session_id, raw, cursor, identity=identity)
            cursor = result.write_index
            meta = await live.sdk.get_session(context.session_id, identity=identity)
            assert meta["message_count"] == len(raw)
            controls.append(time.monotonic() - tick)
        results[number] = (raw, cursor)
    await asyncio.sleep(10)
    after = await ov_process.metrics()
    checks = {
        "zero_errors": not errors,
        "duration": elapsed >= duration,
        "minimum_progress": min(counts) >= max(10, int(duration / 6)),
        "fair_progress": min(counts) / max(counts) >= 0.5,
        "agent_probes": len(probes) >= 8 * max(1, duration // 150),
        "p95_io_limit": percentile(latencies, 0.95) < 5,
        "matched_history_contention": percentile(last, 0.95)
        <= max(1, 2 * len(contexts) * percentile(controls, 0.95)),
        "memory_peak": max(s["working_set"] for s in samples) < baseline["memory_limit"] * 0.85,
        "memory_after_idle": after["working_set"] <= baseline["working_set"] + 256 * 1024**2,
        "fds_after_idle": after["fds"] <= baseline["fds"] + 32,
    }
    summary = {
        "duration_seconds": duration,
        "elapsed_seconds": elapsed,
        "concurrent_sessions": 8,
        "cycles": sum(counts),
        "timed_messages": sum(counts) * 3,
        "messages_after_controls": sum(cursor for _, cursor in results),
        "errors": errors,
        "cycles_per_second": len(records) / elapsed,
        "p50_seconds": statistics.median(latencies),
        "p95_seconds": percentile(latencies, 0.95),
        "first_minute_p95": percentile(first, 0.95),
        "last_minute_p95": percentile(last, 0.95),
        "matched_serial_p95": percentile(controls, 0.95),
        "serial_controls": controls,
        "checks": checks,
        "integrity_complete": False,
        "baseline": baseline,
        "peak_working_set": max(s["working_set"] for s in samples),
        "after_idle": after,
        "agent_probe_runs": len(probes),
        "agent_model_calls": sum(p["statistics"]["model_calls"] for p in probes),
        "agent_reported_tokens": sum(p["statistics"]["total_tokens"] for p in probes),
    }
    save(live.directory / "stress-summary.json", summary)
    for (context, identity), (raw, cursor) in zip(contexts, results, strict=True):
        assert cursor == len(raw)
        view = await live.sdk.get_session_context(
            context.session_id, token_budget=1000000, identity=identity
        )
        assert len(view["messages"]) == len(raw)
        restored = await OVContextReader(live.sdk).exchange_context(
            session_id=context.session_id, identity=identity, token_budget=1000000
        )
        validate_tool_pairs(restored.messages)
        assert len(restored.messages) == len(raw)
        assert [m.content for m in restored.messages] == [m.content for m in raw]
    summary["integrity_complete"] = True
    save(live.directory / "stress-summary.json", summary)
    assert all(checks.values()), checks
