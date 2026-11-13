"""Control only an explicitly labelled validation OV, outside its process/container."""

from __future__ import annotations

import asyncio
import json
import os
import struct
import time
from urllib.parse import urlsplit

import httpx


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
"""
        create = await self.http.post(
            f"/containers/{self.name}/exec",
            json={
                "AttachStdout": True,
                "AttachStderr": True,
                "Cmd": ["/validation/OpenViking-0.4.1/.venv/bin/python", "-c", script],
            },
        )
        create.raise_for_status()
        response = await self.http.post(
            f"/exec/{create.json()['Id']}/start", json={"Detach": False, "Tty": False}
        )
        response.raise_for_status()
        raw, out = response.content, b""
        while len(raw) >= 8:
            size = struct.unpack(">I", raw[4:8])[0]
            out += raw[8 : 8 + size]
            raw = raw[8 + size :]
        result.update(json.loads(out))
        return result

    async def close(self):
        await self.http.aclose()
