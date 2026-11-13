"""Offline HTTP fixtures with simulated OV service responses."""

import json
from collections import Counter
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import httpx
import pytest_asyncio
from langchain.agents.middleware import ModelRequest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage

from openviking.integrations.langchain.client import OpenVikingConnection
from openviking.integrations.langchain.ov_context import BudgetSpec, Metrics, RunIdentity
from openviking.integrations.langchain.ov_context.reader import MemoryReaderConfig
from openviking.integrations.langchain.ov_context.recall_assembly import _render
from openviking.integrations.langchain.ov_context.sdk import AsyncClientHandle
from openviking.storage.transaction import LockManager
from tests.ov_context_support.mock_reader import MockContextReader
from tests.ov_context_support.mock_writer import MockMessageWriter
from tests.ov_context_support.models import RecallPolicy, ResolvedCaller
from tests.ov_context_support.transport import NativeHTTPTransport
from tests.transaction.conftest import MemoryAgfs


@pytest_asyncio.fixture(autouse=True)
async def session_path_locks(monkeypatch):
    """Exercise native path locking without requiring an external AGFS service."""
    manager = LockManager(MemoryAgfs(), lock_timeout=1, redo_recovery_enabled=False)
    monkeypatch.setattr("openviking.storage.transaction.get_lock_manager", lambda: manager)
    yield manager
    await manager.stop()


def mock_io(
    client: Any, *, reader_config: MemoryReaderConfig | None = None, peer_id: str | None = None
) -> dict[str, Any]:
    """Test-only adapters for both external I/O contracts."""
    handle = AsyncClientHandle(OpenVikingConnection(client=client))
    # Offline adapters share fake OV storage and record IDs, outside the HTTP contract.
    metadata = getattr(client.transport, "test_metadata", {})
    client.transport.test_metadata = metadata
    return {
        "message_writer": MockMessageWriter(handle, peer_id, metadata),
        "context_reader": MockContextReader(handle, reader_config, metadata),
    }


class NativeServer:
    def __init__(self):
        self.requests = []
        self.counts = Counter()
        self.messages = []
        self.sessions = set()
        self.session_messages = {}
        self.native_payloads = {}
        self.native_host_ids = {"_native_payloads": self.native_payloads}
        self.history = []
        self.overview = ""
        self.archive_id = None
        self.context_hook = None
        self.session_meta_hook = None
        self.error = {}
        self.search_hook = None
        self.batch_hook = None
        self.memory = {
            "uri": "viking://user/u/peers/p/memories/preferences/style.md",
            "context_type": "memory",
            "category": "preferences",
            "score": 0.8,
            "abstract": "short answers",
            "overview": "Prefer concise answers",
        }
        self.recalled = [self.memory]

    async def handle(self, request):
        path = request.url.path
        body = json.loads(request.content) if request.content else None
        self.requests.append((request, body))
        event = path.rsplit("/", 1)[-1]
        self.counts[event] += 1
        if event in self.error:
            return httpx.Response(self.error[event])
        if event == "sessions":
            assert set(body) <= {"session_id", "memory_policy"}
            self.sessions.add(body["session_id"])
            result = {"session_id": body["session_id"]}
        elif event in ("search", "find"):
            if self.search_hook:
                await self.search_hook()
            if body.get("mode") == "context":
                assert event == "search" and body["query_expansion"] == "off"
                entries = [
                    {
                        "uri": item["uri"],
                        "category": item["category"],
                        "score": item["score"],
                        "detail": "abstract",
                        "text": item["abstract"],
                    }
                    for item in self.recalled
                ]
                result = {"entries": entries, "rendered": _render(entries), "stats": {}}
            else:
                assert set(body) - {"level", "filter", "telemetry"} == {
                    "query",
                    "target_uri",
                    "context_type",
                    "limit",
                    "score_threshold",
                }
                assert body.get("filter") is None
                assert not body.get("telemetry", False)
                result = {"memories": self.recalled, "resources": [], "skills": []}
        elif event == "context":
            assert set(request.url.params) == {"token_budget"}
            if self.context_hook:
                await self.context_hook()
            sid = path.split("/")[-2]
            result = {
                "messages": deepcopy(self.history + self.session_messages.get(sid, [])),
                "latest_archive_overview": self.overview,
                "stats": {"totalArchives": int(bool(self.overview)), "failedArchives": 0},
                "refresh": self.counts[event],
            }
        elif event == "read":
            assert set(request.url.params) == {"uri", "offset", "limit"}
            result = "Full preference text"
        elif event == "batch":
            assert set(body) == {"messages"}
            assert 1 <= len(body["messages"]) <= 100
            assert all(
                set(message)
                <= {
                    "role",
                    "content",
                    "parts",
                    "peer_id",
                    "message_kind",
                    "source_message_ids",
                }
                for message in body["messages"]
            )
            if self.batch_hook:
                await self.batch_hook()
            sid = path.split("/")[-3]
            native = self.session_messages.setdefault(sid, [])
            for message in body["messages"]:
                native.append({**deepcopy(message), "id": f"native-{len(self.messages)}"})
                self.messages.append(deepcopy(message))
            result = {
                "session_id": sid,
                "added": len(body["messages"]),
                "message_count": len(native),
            }
        elif event == "commit":
            assert set(body) - {"telemetry"} == {"keep_recent_count"}
            assert not body.get("telemetry", False)
            result = {"archived": True, "status": "accepted", "task_id": "t1"}
        elif path == "/api/v1/tasks":
            result = []
        elif path.startswith("/api/v1/tasks/"):
            result = {"task_id": event, "status": "completed"}
        elif request.method == "GET":
            if event not in self.sessions:
                return httpx.Response(404)
            result = {
                "session_id": event,
                "pending_tokens": 0,
                "latest_finished_archive_id": self.archive_id
                or ("archive_001" if self.overview else None),
            }
            if self.session_meta_hook:
                result.update(self.session_meta_hook(event))
        else:
            raise AssertionError(f"Unexpected route: {request.method} {path}")
        return httpx.Response(200, json={"status": "ok", "result": result})


@pytest_asyncio.fixture
async def env():
    server = NativeServer()
    http = httpx.AsyncClient(transport=httpx.MockTransport(server.handle))
    samples = []

    metrics = Metrics(lambda event, **fields: samples.append((event, fields)))
    caller = ResolvedCaller("ov-local", "a", "u", "permissions-v1")
    transport = NativeHTTPTransport(
        url="http://ov.test", caller=caller, api_key="test-only", http_client=http, metrics=metrics
    )
    transport.test_metadata = server.native_host_ids
    transport.test_metadata["_published_archive"] = lambda: (
        server.archive_id or ("archive_001" if server.overview else None)
    )
    options = SimpleNamespace(context_failure="raise", metrics=metrics)
    identity = RunIdentity("a", "u", "p", "s", "r")
    request = ModelRequest(
        model=FakeListChatModel(responses=["done"]),
        messages=[HumanMessage(id="m1", content="Read the mail")],
        tools=[],
        model_settings={},
    )
    yield server, transport, options, identity, request, samples
    await http.aclose()


BUDGET = BudgetSpec(100000, 1000)
POLICY = RecallPolicy(content_mode="full")


def counter(request):
    # A deterministic test counter that counts the whole request; not a production tokenizer.
    return (
        len(str(request.system_message))
        + len(str(request.messages))
        + len(str(request.tools))
        + len(str(request.response_format))
    )


def publish(server, identity, ids, text="Read the target mail successfully"):
    """Publish a fake external Reader snapshot with its completed archive ID."""
    # IDs are provided by this fixture's out-of-band adapter metadata.
    mapping = server.native_host_ids
    server.session_messages[identity.session_id] = [
        record
        for record in server.session_messages[identity.session_id]
        if not set(mapping.get(record["id"], [])) & ids
    ]
    server.overview = text
    index = int(server.archive_id.split("_")[1]) + 1 if server.archive_id else 1
    server.archive_id = f"archive_{index:03d}"
