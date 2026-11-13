"""Real HTTP, process and SQLite helpers; no simulated OV or model responses."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from openviking.integrations.langchain.client import OpenVikingConnection
from openviking.integrations.langchain.ov_context import OVContext, OVContextConfig
from openviking.integrations.langchain.ov_context.commit import CommitConfig, OpenVikingCommitPolicy
from openviking.integrations.langchain.ov_context.identity import OVIdentity
from openviking.integrations.langchain.ov_context.reader import MemoryReaderConfig, OVContextReader
from openviking.integrations.langchain.ov_context.session_engine import SessionEngine
from openviking.integrations.langchain.ov_context.writer import OVMessageWriter
from openviking_cli.client.http import AsyncHTTPClient

REPO = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class LiveEnvironment:
    url: str
    key_file: Path = field(repr=False)
    model_config: Path = field(repr=False)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def events(text):
    result = []
    for frame in text.split("\n\n"):
        rows = frame.splitlines()
        kind = next((r[7:] for r in rows if r.startswith("event: ")), None)
        data = next((r[6:] for r in rows if r.startswith("data: ")), None)
        if kind and data:
            result.append((kind, json.loads(data)))
    return result


class LiveService:
    def __init__(self, config, directory):
        self.config = config
        self.directory = directory
        self.prefix = "it-" + uuid.uuid4().hex[:12]
        self.sessions = []
        self.wire = []

    async def __aenter__(self):
        async def observe(request):
            row = {"method": request.method, "path": request.url.path}
            for name in ("account", "user", "actor-peer"):
                row[name] = request.headers.get("x-openviking-" + name)
            self.wire.append(row)

        self.pool = httpx.AsyncClient(timeout=65, event_hooks={"request": [observe]})
        self.sdk = AsyncHTTPClient(
            url=self.config.url,
            api_key=self.config.key_file.read_text().strip(),
            auth_mode="trusted",
            profile_enabled=False,
            http_client=self.pool,
        )
        await self.sdk.initialize()
        return self

    async def __aexit__(self, *args):
        # Only remove sessions created by this invocation; retain evidence on disk.
        errors = []
        for sid, identity in self.sessions:
            try:
                await self.sdk.delete_session(sid, identity=identity)
            except Exception as exc:
                errors.append({"session": sid, "error_type": type(exc).__name__})
        (self.directory / "http-identity.json").write_text(json.dumps(self.wire, indent=2))
        (self.directory / "cleanup.json").write_text(json.dumps(errors))
        await self.sdk.close()
        await self.pool.aclose()

    async def new(self, *, account=None, user="alice", peer="mail", memory=False):
        identity = OVIdentity(account or self.prefix, user, peer)
        sid = self.prefix + "-" + uuid.uuid4().hex[:8]
        await self.sdk.create_session(
            sid,
            identity=identity,
            memory_policy=None
            if memory
            else {"self": {"enabled": False}, "peer": {"enabled": False}},
        )
        self.sessions.append((sid, identity))
        return OVContext(sid, owner=identity.session_key(sid)), identity

    def engine(self, *, sdk=None, keep=2, context_failure="raise"):
        sdk = sdk or self.sdk
        return SessionEngine(
            OpenVikingConnection(client=sdk),
            OVContextConfig(
                context_window=256000,
                context_failure=context_failure,
                commit_wait_timeout=300,
                poll_interval=0.2,
            ),
            None,
            None,
            message_writer=OVMessageWriter(sdk),
            context_reader=OVContextReader(
                sdk, MemoryReaderConfig(enabled=False, context_timeout=60)
            ),
            session_commit_policy=OpenVikingCommitPolicy(
                ov_client=sdk, config=CommitConfig(256000, 0.7, 100, keep)
            ),
        )

    async def terminal(self, sid, identity, receipt):
        target = receipt.get("archive_uri", "").rstrip("/").split("/")[-1]
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            task = await self.sdk.get_task(receipt["task_id"], identity=identity)
            if task["status"] in ("failed", "cancelled"):
                raise AssertionError("Real archive task failed: " + json.dumps(task))
            meta = await self.sdk.get_session(sid, identity=identity)
            if (
                task["status"] in ("completed", "done")
                and meta.get("latest_finished_archive_id") == target
            ):
                return target
            await asyncio.sleep(0.3)
        raise TimeoutError("Archive did not finish within 300 seconds")


class ChatProcess:
    """Start the actual Chat HTTP/SSE entry point; restart retains its SQLite files."""

    def __init__(self, live, *, ov_url=None):
        self.live = live
        self.ov_url = ov_url or live.config.url
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.data = live.directory / "chat"
        self.generation = 0
        self.process = None

    async def __aenter__(self):
        await self.start()
        self.http = httpx.AsyncClient(base_url=self.url, timeout=360)
        return self

    async def start(self, *, ratio="0.3"):
        self.generation += 1
        env = dict(os.environ)
        env.update(
            {
                "OV_SERVICE_API_KEY": self.live.config.key_file.read_text().strip(),
                "OV_CONTEXT__BASE_CONTEXT_RATIO": ratio,
                "OV_COMMIT__COMPLETED_RUN_THRESHOLD": "100",
                "OV_READER__ENABLED": "false",
            }
        )
        self.log = (self.live.directory / f"chat-{self.generation}.log").open("w")
        self.process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "examples.deepagents_ov_context.chat.server",
            "--model-config",
            str(self.live.config.model_config),
            "--data-dir",
            str(self.data),
            "--ov-url",
            self.ov_url,
            "--account-id",
            self.live.prefix,
            "--env-file",
            str(REPO / "examples/deepagents_ov_context/config/chat.example.env"),
            "--port",
            str(self.port),
            "--max-tokens",
            "8192",
            cwd=REPO,
            env=env,
            stdout=self.log,
            stderr=asyncio.subprocess.STDOUT,
        )
        async with httpx.AsyncClient() as client:
            for _ in range(150):
                if self.process.returncode is not None:
                    raise RuntimeError(f"Chat exited; inspect {self.log.name}")
                try:
                    response = await client.get(self.url + "/api/health")
                    if response.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.2)
        raise TimeoutError(f"Chat did not start; inspect {self.log.name}")

    async def stop(self, *, kill=False):
        if self.process and self.process.returncode is None:
            self.process.kill() if kill else self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), 20)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()
        self.log.close()

    async def __aexit__(self, *args):
        await self.http.aclose()
        await self.stop()

    async def new(self, user="alice", peer="mail"):
        response = await self.http.post(
            "/api/sessions",
            json={"user_id": user, "agent_id": peer, "execution": {"mode": "context_only"}},
        )
        response.raise_for_status()
        session = response.json()
        self.live.sessions.append((session["session_id"], OVIdentity(self.live.prefix, user, peer)))
        return session

    async def state(self, sid):
        response = await self.http.get(f"/api/sessions/{sid}/state")
        response.raise_for_status()
        return response.json()

    async def run(self, sid, message):
        response = await self.http.post("/api/chat", json={"session_id": sid, "message": message})
        response.raise_for_status()
        result = events(response.text)
        (self.live.directory / f"sse-{sid}-{uuid.uuid4().hex[:6]}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2)
        )
        assert result and result[-1][0] == "done", result[-1:]
        assert result[-1][1]["statistics"]["model_calls"] > 0
        assert result[-1][1]["statistics"]["total_tokens"] > 0
        return result


class NetworkGate:
    """Real TCP forwarding; cutting sockets injects an outage without stopping OV."""

    def __init__(self, url):
        self.target = urlsplit(url)
        if self.target.scheme != "http":
            raise ValueError("Fault injection gate requires a test HTTP OV endpoint")
        self.blocked = False
        self.connections = set()
        self.tasks = set()

    async def __aenter__(self):
        self.server = await asyncio.start_server(self.forward, "127.0.0.1", 0)
        self.url = f"http://127.0.0.1:{self.server.sockets[0].getsockname()[1]}"
        return self

    async def forward(self, reader, writer):
        self.connections.add(writer)
        task = asyncio.current_task()
        self.tasks.add(task)
        upstream = None
        try:
            if self.blocked:
                return
            remote, upstream = await asyncio.open_connection(
                self.target.hostname, self.target.port or 80
            )
            self.connections.add(upstream)

            async def pipe(source, destination):
                while data := await source.read(65536):
                    destination.write(data)
                    await destination.drain()

            pair = [
                asyncio.create_task(pipe(reader, upstream)),
                asyncio.create_task(pipe(remote, writer)),
            ]
            try:
                await asyncio.wait(pair, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for t in pair:
                    t.cancel()
                await asyncio.gather(*pair, return_exceptions=True)
        except (OSError, asyncio.CancelledError):
            pass
        finally:
            for stream in (writer, upstream):
                if stream:
                    self.connections.discard(stream)
                    stream.close()
            self.tasks.discard(task)

    def cut(self):
        self.blocked = True
        for writer in list(self.connections):
            writer.close()

    async def __aexit__(self, *args):
        self.cut()
        self.server.close()
        await self.server.wait_closed()
        for task in list(self.tasks):
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
