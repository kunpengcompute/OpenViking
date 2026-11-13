"""Task reads distinguish missing records from backend failure and corrupted JSON."""

from unittest.mock import AsyncMock

import pytest

from openviking.pyagfs import AsyncAGFSClient
from openviking.pyagfs.exceptions import AGFSNetworkError, AGFSNotFoundError
from openviking.service.task_store import PersistentTaskStore


@pytest.fixture
def backend():
    client = AsyncMock(spec=AsyncAGFSClient)
    client.ls.return_value = [{"name": "task.json"}]
    client.read.return_value = '{"task_id": "task"}'
    return client


async def read_at(store, phase):
    if phase == "get":
        return await store.get("task", account_id="a", user_id="u")
    return await store.list("a", user_id="u")


def fail_at(backend, phase, error):
    (backend.ls if phase == "directory" else backend.read).side_effect = error


@pytest.mark.parametrize("phase", ["get", "directory", "record"])
@pytest.mark.parametrize(
    "error",
    [
        AGFSNetworkError("Upstream endpoint not found; network unavailable"),
        OSError("temporary backend failure"),
        PermissionError("access denied"),
    ],
)
async def test_storage_failure_is_propagated_instead_of_missing(backend, phase, error):
    fail_at(backend, phase, error)
    store = PersistentTaskStore(backend)
    with pytest.raises(type(error)) as caught:
        await read_at(store, phase)
    assert caught.value is error


@pytest.mark.parametrize("phase", ["get", "directory", "record"])
@pytest.mark.parametrize("error", [AGFSNotFoundError("gone"), FileNotFoundError("gone")])
async def test_explicit_missing_record_has_empty_result(backend, phase, error):
    fail_at(backend, phase, error)
    result = await read_at(PersistentTaskStore(backend), phase)
    assert result == (None if phase == "get" else [])


@pytest.mark.parametrize("phase", ["get", "record"])
async def test_corrupted_task_json_is_reported(backend, phase):
    import json

    backend.read.return_value = '{"task_id":'
    with pytest.raises(json.JSONDecodeError):
        await read_at(PersistentTaskStore(backend), phase)


async def test_vanished_task_does_not_hide_remaining_records(backend):
    backend.ls.return_value = [{"name": "gone.json"}, {"name": "kept.json"}]
    backend.read.side_effect = [AGFSNotFoundError("gone"), '{"task_id": "kept"}']
    result = await PersistentTaskStore(backend).list("a", user_id="u")
    assert result == [{"task_id": "kept"}]
    assert backend.read.await_count == 2
