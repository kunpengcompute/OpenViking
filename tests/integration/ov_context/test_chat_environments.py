"""One Chat API owns daily and isolated experiment runs and their history."""

from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from fastapi import HTTPException

from examples.deepagents_ov_context.chat.environments import ChatEnvironment, load_environments
from examples.deepagents_ov_context.chat.session import SessionStore

from .chat_model_support import NoToolModel
from .test_chat_api import parse_events


def configure(rt, tmp_path):
    key = tmp_path / "experiment.key"
    key.write_text("private-experiment-key")
    env = tmp_path / "experiment.env"
    env.write_text(
        Path("examples/deepagents_ov_context/config/chat.example.env")
        .read_text()
        .replace("CHAT_AGENT__ENABLE_TODOS=true", "CHAT_AGENT__ENABLE_TODOS=false")
        .replace("TOKEN_BUDGET=256000", "TOKEN_BUDGET=350000")
    )
    rt.settings = replace(
        rt.settings,
        environments=(
            ChatEnvironment(
                id="experiment",
                label="隔离实验",
                ov_url="http://experiment.test:19461",
                account_id="experiment-account",
                api_key_file=key,
                env_file=env,
            ),
        ),
    )


async def test_environment_selection_and_snapshot_roundtrip(chat, tmp_path):
    client, rt = chat
    configure(rt, tmp_path)
    choices = (await client.get("/api/environments")).json()
    assert [e["id"] for e in choices] == ["daily", "experiment"]
    assert "api_key_file" not in str(choices) and "private-experiment-key" not in str(choices)
    for environment_id, account, enabled in [
        ("daily", "chat-account", True),
        ("experiment", "experiment-account", False),
    ]:
        defaults = (
            await client.get("/api/execution-profiles", params={"environment_id": environment_id})
        ).json()
        assert defaults["account_id"] == account
        assert defaults["agent_defaults"]["enable_todos"] is enabled
        created = await client.post(
            "/api/sessions", json={"environment_id": environment_id, "experiment_id": "paired-01"}
        )
        assert created.status_code == 201
        session = rt.store.get(created.json()["session_id"])
        store = SessionStore(rt.settings.data_dir / "sessions.sqlite3")
        try:
            restored = store.get(session.session_id)
            assert restored == session
            assert restored.config_snapshot.environment.id == environment_id
            assert restored.config_snapshot.execution["profile"]["enable_todos"] is enabled
            assert restored.account_id == account
            assert "private-experiment-key" not in restored.model_dump_json()
        finally:
            store.close()
    listing = (await client.get("/api/sessions")).json()
    assert len(listing) == 2 and all(s["experiment_id"] == "paired-01" for s in listing)
    assert (
        await client.post("/api/sessions", json={"environment_id": "unknown"})
    ).status_code == 409
    assert (
        await client.post("/api/sessions", json={"ov_url": "http://unregistered"})
    ).status_code == 422


async def test_session_binding_cannot_silently_move_and_metadata_is_immutable(chat, tmp_path):
    client, rt = chat
    configure(rt, tmp_path)
    sid = (await client.post("/api/sessions", json={"environment_id": "experiment"})).json()[
        "session_id"
    ]
    saved = rt.store.get(sid)
    for updates in ({"environment_id": "daily"}, {"experiment_id": "other"}):
        with pytest.raises(HTTPException, match="已固定"):
            rt.store.save(saved.model_copy(update=updates))
    rt.settings = replace(
        rt.settings,
        environments=(
            rt.settings.environments[0].model_copy(update={"ov_url": "http://changed.test"}),
        ),
    )
    response = await client.post("/api/chat", json={"session_id": sid, "message": "hello"})
    assert response.status_code == 409
    assert (await client.get(f"/api/sessions/{sid}/runs")).json() == []


async def test_sdk_read_commit_task_restore_all_use_selected_backend(chat, tmp_path):
    client, rt = chat
    configure(rt, tmp_path)
    requests = []
    sid = (await client.post("/api/sessions", json={"environment_id": "experiment"})).json()[
        "session_id"
    ]

    def respond(request):
        requests.append(request)
        path = request.url.path
        result = (
            {"archived": True, "task_id": "task-1"}
            if request.method == "POST"
            else {
                "task_type": "session_commit",
                "resource_id": sid,
                "status": "completed",
                "result": {},
            }
            if "/tasks/" in path
            else {"messages": [], "latest_finished_archive_id": None}
            if path.endswith("/context")
            else {"message_count": 2}
        )
        return httpx.Response(200, json={"result": result})

    await rt.ov_http.aclose()
    rt.ov_http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    assert (await client.get(f"/api/sessions/{sid}")).status_code == 200
    assert (await client.post(f"/api/sessions/{sid}/memory")).status_code == 200
    assert (await client.get(f"/api/sessions/{sid}/memory")).json()["status"] == "completed"
    assert (await client.post(f"/api/sessions/{sid}/restore")).status_code == 200
    assert requests and all(r.url.host == "experiment.test" for r in requests)
    assert all(r.headers["x-openviking-account"] == "experiment-account" for r in requests)
    assert all(r.headers["x-api-key"] == "private-experiment-key" for r in requests)


async def test_baseline_experiment_is_visible_with_trace_but_never_uses_ov(chat, tmp_path):
    client, rt = chat
    configure(rt, tmp_path)
    rt.model = NoToolModel()
    await rt.ov_http.aclose()

    def forbidden(request):
        pytest.fail("Baseline must not request OV")

    rt.ov_http = httpx.AsyncClient(transport=httpx.MockTransport(forbidden))
    sid = (
        await client.post(
            "/api/sessions",
            json={
                "environment_id": "experiment",
                "experiment_id": "suite-01",
                "execution": {"mode": "baseline", "enable_todos": False},
            },
        )
    ).json()["session_id"]
    response = await client.post("/api/chat", json={"session_id": sid, "message": "hello"})
    events = parse_events(response.text)
    assert events[-1][0] == "done"
    assert events[0][1]["environment_id"] == "experiment"
    assert (await client.get(f"/api/sessions/{sid}")).status_code == 200
    runs = (await client.get(f"/api/sessions/{sid}/runs")).json()
    assert len(runs) == 1
    trace = (await client.get(f"/api/sessions/{sid}/runs/{runs[0]['run_id']}")).json()
    assert any(e["event"] == "model_input" for e in trace["events"])


@pytest.mark.parametrize(
    "text",
    ["{}", '[{"id":"daily","label":"reserved","ov_url":"http://localhost","account_id":"a"}]'],
)
def test_environment_file_rejects_invalid_registry(tmp_path, text):
    path = tmp_path / "environments.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_environments(path)


async def test_imported_history_is_read_only(chat):
    client, rt = chat
    created = await client.post("/api/sessions", json={"execution": {"mode": "baseline"}})
    sid = created.json()["session_id"]
    # Offline import sets this field before saving; normal HTTP clients cannot request it.
    rt.store.db.execute(
        "UPDATE sessions SET data=json_set(data,'$.history_only',json('true')) WHERE id=?", (sid,)
    )
    rt.store.db.commit()
    for path, body in [
        ("/api/chat", {"session_id": sid, "message": "hi"}),
        (f"/api/sessions/{sid}/restore", None),
        (f"/api/sessions/{sid}/memory", None),
    ]:
        response = await client.post(path, json=body)
        assert response.status_code == 409
    assert (await client.get(f"/api/sessions/{sid}/runs")).status_code == 200


async def test_daily_and_experiment_runs_do_not_cross_sdk_destinations(chat, tmp_path):
    import asyncio

    from tests.ov_context_support.fixtures import NativeServer

    client, rt = chat
    configure(rt, tmp_path)
    servers = {"127.0.0.1": NativeServer(), "experiment.test": NativeServer()}
    requests = []

    async def respond(request):
        requests.append(request)
        response = await servers[request.url.host].handle(request)
        if response.status_code == 404:
            return httpx.Response(
                404, json={"status": "error", "error": {"code": "NOT_FOUND", "message": "missing"}}
            )
        if request.url.path.endswith("/context"):
            body = response.json()
            body["result"]["latest_finished_archive_id"] = None
            return httpx.Response(200, json=body)
        return response

    await rt.ov_http.aclose()
    rt.ov_http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    rt.model = NoToolModel()
    sessions = []
    for environment_id in ["daily", "experiment"]:
        response = await client.post(
            "/api/sessions",
            json={
                "environment_id": environment_id,
                "user_id": "same-user",
                "agent_id": "same-agent",
                "execution": {"mode": "ordinary"},
            },
        )
        assert response.status_code == 201
        sessions.append(response.json()["session_id"])

    async def execute(sid):
        response = await client.post("/api/chat", json={"session_id": sid, "message": "hello"})
        events = parse_events(response.text)
        assert events[-1][0] == "done", events
        assert not events[-1][1]["context"]["ov_degraded"]

    await asyncio.gather(*(execute(sid) for sid in sessions))
    assert set(servers["127.0.0.1"].sessions) == {sessions[0]}
    assert set(servers["experiment.test"].sessions) == {sessions[1]}
    assert all(
        r.headers["x-openviking-account"]
        == ("chat-account" if r.url.host == "127.0.0.1" else "experiment-account")
        for r in requests
    )
    for sid in sessions:
        assert (await client.get(f"/api/sessions/{sid}/state")).json()["ov_context"][
            "write_index"
        ] == 2
