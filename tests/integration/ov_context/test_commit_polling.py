"""A pending commit has one status read per cycle and bounded polling backoff."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from openviking.integrations.langchain.ov_context import OVContext, OVContextConfig
from openviking.integrations.langchain.ov_context.errors import ContextBudgetExceeded
from openviking.integrations.langchain.ov_context.reader import ContextSnapshot
from openviking.integrations.langchain.ov_context.session_engine import SessionEngine


class PollSDK:
    _initialized = True

    def __init__(self, finish_after=5, status="completed", task_id="t"):
        self.finish_after, self.status, self.task_id = finish_after, status, task_id
        self.gets = self.tasks = self.reads = 0
        self.ready = False
        self.publish_boundary = True

    async def get_session(self, session_id, *, identity, **kwargs):
        self.gets += 1
        if self.task_id is None and self.gets >= self.finish_after:
            self.ready = True
        return {
            "session_id": session_id,
            "pending_tokens": 0,
            "commit_count": 1,
            "latest_finished_archive_id": "archive_001" if self.ready else None,
        }

    async def get_task(self, task_id, *, identity):
        assert task_id == self.task_id and identity.actor_peer_id == "p"
        self.tasks += 1
        if self.tasks < self.finish_after:
            return {"status": "running"}
        self.ready = self.status == "completed"
        return {"status": self.status, "result": {"archive_id": "archive_001"}}

    async def exchange_context(self, **kwargs):
        self.reads += 1
        assert self.ready
        return ContextSnapshot(
            messages=[
                HumanMessage(
                    id="ov_summary:archive_001", content="[Session History Summary] summary"
                )
            ],
            archive_id="archive_001" if self.publish_boundary else None,
        )


def pending(sdk):
    config = OVContextConfig(
        context_window=100000, poll_interval=0.001, poll_max_interval=0.004, commit_wait_timeout=1
    )
    engine = SessionEngine(SimpleNamespace(client=sdk), config, None, None, context_reader=sdk)
    context = OVContext(
        session_id="s",
        owner=("a", "u", "p", "s"),
        commit_count=1,
        pending_commit={
            "task_id": sdk.task_id,
            "archive_id": None,
            "target_archive_id": "archive_001",
        },
    )
    return engine, context


@pytest.mark.asyncio
@pytest.mark.parametrize("task_id", ["t", None])
async def test_one_status_read_per_cycle_and_capped_backoff(monkeypatch, task_id):
    sdk = PollSDK(task_id=task_id)
    engine, ctx = pending(sdk)
    delays = []

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(
        "openviking.integrations.langchain.ov_context.session_engine.asyncio.sleep", sleep
    )
    await engine.wait_summary(ctx, identity=ctx.identity)
    assert sdk.gets == 5 and sdk.tasks == (5 if task_id else 0)
    assert sdk.reads == 1 and ctx.pending_commit is None and ctx.archive_id == "archive_001"
    assert delays == [0.001, 0.002, 0.004, 0.004]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "cancelled", "canceled"])
async def test_terminal_failure_propagates_without_second_task_read(status):
    sdk = PollSDK(finish_after=1, status=status)
    engine, ctx = pending(sdk)
    with pytest.raises(ContextBudgetExceeded, match="did not complete"):
        await engine.wait_summary(ctx, identity=ctx.identity)
    assert sdk.tasks == 1 and sdk.reads == 0 and ctx.archive_id is None


@pytest.mark.asyncio
async def test_timeout_retains_receipt_for_recovery():
    sdk = PollSDK(finish_after=10000)
    engine, ctx = pending(sdk)
    engine.config = replace(engine.config, commit_wait_timeout=0.01)
    with pytest.raises(TimeoutError):
        await engine.wait_summary(ctx, identity=ctx.identity)
    assert ctx.pending_commit["task_id"] == "t" and sdk.reads == 0


@pytest.mark.asyncio
async def test_completed_task_cannot_install_unpublished_archive():
    sdk = PollSDK(finish_after=1)
    sdk.publish_boundary = False
    engine, ctx = pending(sdk)
    with pytest.raises(ContextBudgetExceeded):
        await engine.wait_summary(ctx, identity=ctx.identity)
    assert ctx.archive_id is None and ctx.pending_commit["task_id"] == "t"


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_backoff_cap_must_be_finite_positive(value):
    with pytest.raises(ValueError):
        OVContextConfig(poll_max_interval=value)


def test_polling_settings_load_from_env_file(tmp_path, monkeypatch):
    from examples.deepagents_ov_context.chat.config import load_context_config

    monkeypatch.delenv("OV_CONTEXT__POLL_INTERVAL", raising=False)
    monkeypatch.delenv("OV_CONTEXT__POLL_MAX_INTERVAL", raising=False)
    path = tmp_path / "chat.env"
    path.write_text("OV_CONTEXT__POLL_INTERVAL=0.25\nOV_CONTEXT__POLL_MAX_INTERVAL=0.75\n")
    cfg = load_context_config(path)
    assert cfg.poll_interval == 0.25 and cfg.poll_max_interval == 0.75
