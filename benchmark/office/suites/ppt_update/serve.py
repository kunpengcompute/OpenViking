"""Run an isolated standard Chat with a read-only PPT evidence observer."""

from __future__ import annotations

import os
from pathlib import Path

from examples.deepagents_ov_context.chat import server as chat_server
from examples.deepagents_ov_context.chat.stream import InputTrace

from .observer import RevisionAudit


class AuditedInputTrace(InputTrace):
    def __init__(self, run, session, store):
        super().__init__(run, session, store)
        root = Path(os.environ["PPT_AUDIT_CHAT_DATA"])
        self.audit = RevisionAudit(
            root / "artifacts" / session.session_id / run.run_id,
            root / "ppt-audit" / session.session_id / run.run_id,
        )
        self.audit_calls = {}

    async def on_tool_start(self, serialized, input_str, *, run_id, inputs=None, **kwargs):
        await super().on_tool_start(serialized, input_str, run_id=run_id, inputs=inputs, **kwargs)
        tool = kwargs.get("name") or (serialized or {}).get("name", "unknown")
        call_id = kwargs.get("tool_call_id") or str(run_id)
        args = inputs if inputs is not None else input_str
        self.audit_calls[run_id] = (call_id, tool, args)
        self.audit.capture("start", call_id, tool, args)

    async def on_tool_end(self, output, *, run_id, **kwargs):
        await super().on_tool_end(output, run_id=run_id, **kwargs)
        call_id, tool, args = self.audit_calls.pop(run_id, (str(run_id), "unknown", None))
        self.audit.capture("end", call_id, tool, args)

    async def on_tool_error(self, error, *, run_id, **kwargs):
        await super().on_tool_error(error, run_id=run_id, **kwargs)
        call_id, tool, args = self.audit_calls.pop(run_id, (str(run_id), "unknown", None))
        self.audit.capture("error", call_id, tool, args)


if __name__ == "__main__":
    chat_server.InputTrace = AuditedInputTrace
    chat_server.main()
