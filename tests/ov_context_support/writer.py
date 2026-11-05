"""Real Session/ToolResultStore with temporary files, without model services."""

from pathlib import Path
from typing import Any

from openviking.message.part import part_from_dict
from openviking.server.config import ToolOutputExternalizationConfig
from openviking.session import Session
from openviking.session.tool_views import add_messages_with_tool_views_async
from openviking_cli.exceptions import NotFoundError


class FileVikingFS:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.fail_tool_store = False

    def path(self, uri: str) -> Path:
        return self.root / uri.removeprefix("viking://")

    def _uri_to_path(self, uri: str, *, ctx: Any = None) -> str:
        # The fixture's lock manager uses AGFS POSIX paths even when its real
        # message files live on Windows. Keep the same per-directory identity.
        return "/" + self.path(uri).resolve().as_posix().lstrip("/")

    async def write_file(self, uri: str, content: str, *, ctx: Any = None) -> None:
        if self.fail_tool_store and "/tool-results/" in uri:
            raise OSError("test tool-store write failure")
        path = self.path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    async def append_file(self, uri: str, content: str, *, ctx: Any = None) -> None:
        path = self.path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(content)

    async def read_file(self, uri: str, *, ctx: Any = None) -> str:
        path = self.path(uri)
        if not path.exists():
            raise NotFoundError(uri, "file")
        return path.read_text(encoding="utf-8")

    async def ls(
        self, uri: str, *, output: str = "original", node_limit: int = 1000, ctx: Any = None
    ) -> list[dict[str, Any]]:
        path = self.path(uri)
        if not path.exists():
            raise NotFoundError(uri, "directory")
        return [{"name": entry.name, "isDir": entry.is_dir()} for entry in sorted(path.iterdir())][
            :node_limit
        ]


def make_session(root: Path, **overrides: Any) -> Session:
    values = {
        "threshold_chars": 2000,
        "preview_chars": 100,
        "assistant_turn_inline_budget_chars": 10000,
        "assistant_turn_preview_budget_chars": 500,
        "min_preview_chars": 32,
    }
    values.update(overrides)
    return Session(
        FileVikingFS(root),
        session_id="writer-test",
        tool_output_externalization_config=ToolOutputExternalizationConfig(**values),
    )


class SessionBatchClient:
    """Use native asynchronous writes and compression receipts."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.requests: list[list[dict[str, Any]]] = []

    async def get_session(
        self, session_id: str, auto_create: bool = False, *, identity: Any = None
    ) -> dict[str, Any]:
        assert session_id == self.session.session_id
        return {"session_id": session_id}

    async def batch_add_messages(
        self, session_id: str, messages: list[dict[str, Any]], *, identity: Any = None
    ) -> dict[str, Any]:
        assert session_id == self.session.session_id
        self.requests.append(messages)
        specs = [
            {**spec, "parts": [part_from_dict(part) for part in spec["parts"]]} for spec in messages
        ]
        return await add_messages_with_tool_views_async(self.session, specs)

    async def list_tasks(self, *, identity: Any = None, **filters: Any) -> list[dict[str, Any]]:
        return []
