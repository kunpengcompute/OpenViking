"""Poll checkpointed task IDs until their archive is observed; never rediscover tasks."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage

from openviking.integrations.langchain.client import OpenVikingConnection
from openviking.integrations.langchain.ov_context import OVContext, OVContextConfig
from openviking.integrations.langchain.ov_context.identity import OVIdentity
from openviking.integrations.langchain.ov_context.reader import ContextSnapshot
from openviking.integrations.langchain.ov_context.session_engine import SessionEngine


@pytest.mark.parametrize("commit_count", [1, 2])
async def test_checkpointed_task_is_preserved_until_its_archive_is_applied(commit_count):
    archive = f"archive_{commit_count:03d}"
    meta = {"session_id": "s", "commit_count": commit_count, "latest_finished_archive_id": None}
    task = {"status": "running"}
    calls = []

    class SDK:
        _initialized = True

        async def get_session(self, session_id, *, identity, auto_create=False):
            return meta

        async def get_task(self, task_id, *, identity):
            calls.append(task_id)
            return task

        async def list_tasks(self, **kwargs):
            pytest.fail("Task discovery must not be used")

    identity = OVIdentity("a", "u", "p")
    pending = {"task_id": "known-task", "archive_id": None, "target_archive_id": "archive_001"}
    context = OVContext(
        "s", owner=identity.session_key("s"), commit_count=0, pending_commit=pending
    )
    read = AsyncMock(return_value=ContextSnapshot([AIMessage(id="h", content="history")], archive))
    engine = SessionEngine(
        OpenVikingConnection(client=SDK()),
        OVContextConfig(),
        None,
        None,
        context_reader=SimpleNamespace(exchange_context=read),
    )
    engine.recover_commit(context, persisted=True)
    await engine.check_summary(context, identity=identity)
    assert context.pending_commit == pending
    assert context.commit_count == commit_count
    read.assert_not_awaited()

    meta["latest_finished_archive_id"] = archive
    task.update(status="completed", result={"archive_uri": f"viking://s/history/{archive}"})
    await engine.check_summary(context, identity=identity)
    assert calls == ["known-task", "known-task"]
    assert context.archive_id == archive and context.pending_commit is None
    read.assert_awaited_once()
