"""Control only an explicitly labelled validation OV, outside its process/container."""

from __future__ import annotations

import asyncio
import json
import os
import struct
import time
from urllib.parse import urlsplit

import httpx


def _decode_exec_output(raw: bytes) -> tuple[bytes, bytes]:
    streams = {1: bytearray(), 2: bytearray()}
    offset = 0
    while offset < len(raw):
        if len(raw) - offset < 8:
            raise RuntimeError("Truncated Docker exec stream header")
        header = raw[offset : offset + 8]
        stream = header[0]
        if stream not in streams or header[1:4] != b"\0\0\0":
            raise RuntimeError("Invalid Docker exec stream header")
        size = struct.unpack(">I", header[4:8])[0]
        end = offset + 8 + size
        if end > len(raw):
            raise RuntimeError("Truncated Docker exec stream payload")
        streams[stream].extend(raw[offset + 8 : end])
        offset = end
    return bytes(streams[1]), bytes(streams[2])


class DockerOV:
    def __init__(self, environment):
        self.url = environment.url
        self.name = os.environ.get("OV_LIVE_CONTAINER", "")
        if not self.name.startswith("ov-middleware-integration-"):
            raise ValueError("Use an isolated ov-middleware-integration-* container")
        self.http = httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds="/var/run/docker.sock"),
            base_url="http://docker",
            timeout=30,
        )
        self.events = []

    async def inspect(self):
        response = await self.http.get(f"/containers/{self.name}/json")
        response.raise_for_status()
        data = response.json()
        if data["Config"].get("Labels", {}).get("ov.live-test") != "true":
            raise ValueError("Refusing to control a container without ov.live-test=true")
        command = data["Config"]["Cmd"]
        if (
            "--port" not in command
            or int(command[command.index("--port") + 1]) != urlsplit(self.url).port
        ):
            raise ValueError("Controlled container does not match the configured test OV port")
        return data

    async def kill(self):
        before = await self.inspect()
        response = await self.http.post(f"/containers/{self.name}/kill", params={"signal": "KILL"})
        response.raise_for_status()
        for _ in range(100):
            state = (await self.inspect())["State"]
            if not state["Running"]:
                assert state["ExitCode"] == 137 and not state["OOMKilled"]
                self.events.append(
                    {
                        "event": "sigkill",
                        "old_pid": before["State"]["Pid"],
                        "exit_code": state["ExitCode"],
                    }
                )
                return
            await asyncio.sleep(0.05)
        raise TimeoutError("Validation OV did not exit after SIGKILL")

    async def start(self):
        state = (await self.inspect())["State"]
        if not state["Running"]:
            response = await self.http.post(f"/containers/{self.name}/start")
            response.raise_for_status()
        async with httpx.AsyncClient(timeout=2) as client:
            for _ in range(180):
                try:
                    response = await client.get(self.url + "/health")
                    if response.status_code == 200:
                        current = await self.inspect()
                        self.events.append({"event": "ready", "pid": current["State"]["Pid"]})
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.5)
        raise TimeoutError("Validation OV did not recover within 90 seconds")

    async def metrics(self):
        response = await self.http.get(
            f"/containers/{self.name}/stats", params={"stream": "false", "one-shot": "true"}
        )
        response.raise_for_status()
        stats = response.json()
        mem = stats.get("memory_stats", {})
        detail = mem.get("stats", {})
        inactive = detail.get("inactive_file", detail.get("total_inactive_file", 0))
        result = {
            "time": time.time(),
            "working_set": max(0, mem.get("usage", 0) - inactive),
            "memory_limit": mem.get("limit", 0),
            "pids": stats.get("pids_stats", {}).get("current", 0),
            "cpu_total_ns": stats.get("cpu_stats", {}).get("cpu_usage", {}).get("total_usage", 0),
        }
        # Read only this container's actual server process, excluding the inspection child.
        script = """import glob,json,os
for p in glob.glob("/proc/[0-9]*"):
 try:
  args=open(p+"/cmdline","rb").read().split(b"\\0")
  if b"openviking.server.bootstrap" in args and args[0].rsplit(b"/",1)[-1].startswith(b"python"):
   status=dict(line.split(":",1) for line in open(p+"/status") if ":" in line)
   print(json.dumps({"fds":len(os.listdir(p+"/fd")),"rss_kib":int(status["VmRSS"].split()[0])}))
   break
 except (FileNotFoundError,PermissionError): pass
else:
 raise SystemExit("OV server process not found")
"""
        create = await self.http.post(
            f"/containers/{self.name}/exec",
            json={
                "AttachStdout": True,
                "AttachStderr": True,
                "Cmd": ["python", "-c", script],
            },
        )
        create.raise_for_status()
        exec_id = create.json()["Id"]
        response = await self.http.post(
            f"/exec/{exec_id}/start", json={"Detach": False, "Tty": False}
        )
        response.raise_for_status()
        status = await self.http.get(f"/exec/{exec_id}/json")
        status.raise_for_status()
        execution = status.json()
        exit_code = execution.get("ExitCode")
        if execution.get("Running") or exit_code is None:
            raise RuntimeError("OV metrics command has not finished")
        if exit_code != 0:
            try:
                stdout, stderr = _decode_exec_output(response.content)
                error = stderr or stdout
            except RuntimeError:
                error = response.content
            detail = error.decode("utf-8", errors="replace").strip()[:1000]
            raise RuntimeError(f"OV metrics command failed (exit={exit_code}): {detail}")
        stdout, _ = _decode_exec_output(response.content)
        if not stdout.strip():
            raise RuntimeError("OV metrics command returned no process metrics")
        try:
            process = json.loads(stdout)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError("OV metrics command returned invalid JSON") from exc
        if not isinstance(process, dict) or any(
            type(process.get(key)) is not int or process[key] < 0 for key in ("fds", "rss_kib")
        ):
            raise RuntimeError("OV metrics command returned invalid process metrics")
        result.update(process)
        return result

    async def close(self):
        await self.http.aclose()
