import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from openviking.message import Message, TextPart
from openviking.server.routers.sessions import AutoCommitPolicyRequest
from openviking.service.session_service import SessionService
from openviking.session.auto_commit_policy import (
    AutoCommitPolicy,
    resolve_session_auto_commit_policy,
)
from openviking.session.session import Session, SessionMeta
from openviking_cli.utils.config.memory_config import SessionAutoCommitConfig


def test_global_policy_is_idle_only_and_can_be_disabled() -> None:
    config = SessionAutoCommitConfig(enabled=True, idle_timeout_seconds=3600)

    policy = resolve_session_auto_commit_policy({}, config)

    assert policy == AutoCommitPolicy.global_idle_default(3600)
    assert policy.pending_token_threshold == 0
    assert policy.message_count_threshold == 0
    assert resolve_session_auto_commit_policy({}, SessionAutoCommitConfig()) is None


def test_explicit_policy_overrides_global_config_and_null_disables() -> None:
    disabled_global_config = SessionAutoCommitConfig(enabled=False)
    configured = {
        "auto_commit_policy_configured": True,
        "auto_commit_policy": {"idle_timeout_seconds": 7200},
    }

    policy = resolve_session_auto_commit_policy(configured, disabled_global_config)

    assert policy is not None
    assert policy.idle_timeout_seconds == 7200
    assert policy.pending_token_threshold == 150_000
    assert (
        resolve_session_auto_commit_policy(
            {"auto_commit_policy_configured": True, "auto_commit_policy": None},
            SessionAutoCommitConfig(enabled=True),
        )
        is None
    )


def test_session_meta_persists_explicit_disabled_policy() -> None:
    meta = SessionMeta(auto_commit_policy_configured=True, auto_commit_policy=None)

    restored = SessionMeta.from_dict(meta.to_dict())

    assert restored.auto_commit_policy_configured is True
    assert restored.auto_commit_policy is None


@pytest.mark.parametrize("value", [1, 299])
def test_policy_rejects_idle_timeout_below_minimum(value: int) -> None:
    with pytest.raises(ValidationError):
        AutoCommitPolicyRequest(idle_timeout_seconds=value)


@pytest.mark.parametrize("value", [0, 300, 604800])
def test_policy_accepts_idle_timeout_bounds(value: int) -> None:
    assert AutoCommitPolicyRequest(idle_timeout_seconds=value).idle_timeout_seconds == value


def test_message_threshold_triggers_only_for_explicit_session_policy() -> None:
    service = SessionService()
    configured_session = SimpleNamespace(
        meta=SessionMeta(
            pending_tokens=101,
            message_count=0,
            auto_commit_policy_configured=True,
            auto_commit_policy=AutoCommitPolicy(
                pending_token_threshold=100,
                message_count_threshold=0,
            ).to_dict(),
        )
    )
    inherited_session = SimpleNamespace(
        meta=SessionMeta(
            pending_tokens=101,
            message_count=0,
        )
    )
    service.set_session_auto_commit_config(
        SessionAutoCommitConfig(enabled=True, idle_timeout_seconds=3600)
    )

    assert service._should_run_auto_commit(configured_session, "message_write") is True
    assert service._should_run_auto_commit(inherited_session, "message_write") is False


def test_idle_timeout_uses_each_sessions_effective_policy() -> None:
    service = SessionService()
    service.set_session_auto_commit_config(
        SessionAutoCommitConfig(enabled=True, idle_timeout_seconds=86_400)
    )
    old_timestamp = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    session = SimpleNamespace(
        meta=SessionMeta(
            pending_tokens=20,
            message_count=1,
            last_message_at=old_timestamp,
            auto_commit_policy_configured=True,
            auto_commit_policy=AutoCommitPolicy(
                pending_token_threshold=0,
                message_count_threshold=0,
                idle_timeout_seconds=3600,
            ).to_dict(),
        )
    )

    assert service._should_run_auto_commit(session, "idle_timeout") is True


class _InMemoryMetaFS:
    def __init__(self, meta: dict) -> None:
        self.meta_content = json.dumps(meta)
        self.messages_content = ""
        self.lock = asyncio.Lock()

    def _uri_to_path(self, _uri: str, *, ctx: object) -> str:
        return "/sessions/session-1"

    async def ls(self, uri: str, *, ctx: object, **kwargs):
        return []

    async def read_file(self, uri: str, *, ctx: object) -> str:
        if uri.endswith("/messages.jsonl"):
            return self.messages_content
        return self.meta_content

    async def write_file(self, *, uri: str, content: str, ctx: object) -> None:
        if uri.endswith("/messages.jsonl"):
            self.messages_content = content
        else:
            self.meta_content = content


class _FakeSessionLock:
    def __init__(self, lock: asyncio.Lock, on_acquire=None) -> None:
        self._lock = lock
        self._on_acquire = on_acquire

    async def __aenter__(self):
        await self._lock.acquire()
        if self._on_acquire is not None:
            self._on_acquire()
        return self

    async def __aexit__(self, *_exc_info):
        self._lock.release()


def _session_with_meta(viking_fs: _InMemoryMetaFS, ctx: object) -> Session:
    session = Session(
        viking_fs=viking_fs,
        session_id="session-1",
        session_uri="viking://session/session-1",
        ctx=ctx,
    )
    session._meta = SessionMeta.from_dict(json.loads(viking_fs.meta_content))
    return session


async def test_concurrent_partial_policy_updates_merge_under_lock(monkeypatch) -> None:
    import openviking.storage.transaction as transaction

    initial_policy = AutoCommitPolicy(
        idle_timeout_seconds=3600,
        message_count_threshold=100,
    ).to_dict()
    initial_meta = SessionMeta(
        session_id="session-1",
        auto_commit_policy=initial_policy,
        auto_commit_policy_configured=True,
    ).to_dict()
    viking_fs = _InMemoryMetaFS(initial_meta)
    monkeypatch.setattr(transaction, "get_lock_manager", lambda: None)
    monkeypatch.setattr(
        transaction,
        "LockContext",
        lambda *_args, **_kwargs: _FakeSessionLock(viking_fs.lock),
    )

    service = SessionService(viking_fs=viking_fs)
    ctx = SimpleNamespace(account_id="account-1", user=SimpleNamespace(user_id="user-1"))
    barrier = asyncio.Barrier(2)
    initial_reads = 0
    read_counter_lock = asyncio.Lock()

    async def get_session(_session_id: str, _ctx: object, *, auto_create: bool = False):
        nonlocal initial_reads
        snapshot = _session_with_meta(viking_fs, ctx)
        async with read_counter_lock:
            initial_reads += 1
            should_wait = initial_reads <= 2
        if should_wait:
            await barrier.wait()
        return snapshot

    monkeypatch.setattr(service, "get", get_session)
    monkeypatch.setattr(service, "open", get_session, raising=False)

    await asyncio.gather(
        service.update_auto_commit_policy("session-1", ctx, policy={"idle_timeout_seconds": 600}),
        service.update_auto_commit_policy("session-1", ctx, policy={"message_count_threshold": 5}),
    )

    stored = SessionMeta.from_dict(json.loads(viking_fs.meta_content))
    assert stored.auto_commit_policy["idle_timeout_seconds"] == 600
    assert stored.auto_commit_policy["message_count_threshold"] == 5


async def test_idle_commit_rechecks_condition_inside_commit_lock(monkeypatch) -> None:
    import openviking.service.session_service as session_service_module
    import openviking.storage.transaction as transaction

    old_timestamp = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    meta = SessionMeta(
        session_id="session-1",
        pending_tokens=20,
        message_count=1,
        last_message_at=old_timestamp,
        auto_commit_policy_configured=True,
        auto_commit_policy=AutoCommitPolicy(
            pending_token_threshold=0,
            message_count_threshold=0,
            idle_timeout_seconds=3600,
        ).to_dict(),
    )
    viking_fs = _InMemoryMetaFS(meta.to_dict())
    ctx = SimpleNamespace(account_id="account-1", user=SimpleNamespace(user_id="user-1"))
    session = _session_with_meta(viking_fs, ctx)
    message = Message(id="message-1", role="user", parts=[TextPart(text="hello")])
    session._messages = [message]
    viking_fs.messages_content = message.to_jsonl() + "\n"
    session._usage_records = []
    session._compression = SimpleNamespace(compression_index=0)

    def record_new_message_activity() -> None:
        session._inside_test_lock = True
        updated_meta = SessionMeta.from_dict(json.loads(viking_fs.meta_content))
        updated_meta.last_message_at = datetime.now(timezone.utc).isoformat()
        viking_fs.meta_content = json.dumps(updated_meta.to_dict())

    monkeypatch.setattr(
        transaction,
        "get_lock_manager",
        lambda: None,
    )
    monkeypatch.setattr(
        transaction,
        "LockContext",
        lambda *_args, **_kwargs: _FakeSessionLock(
            viking_fs.lock,
            on_acquire=record_new_message_activity,
        ),
    )

    monkeypatch.setattr(
        session_service_module,
        "get_task_tracker",
        lambda: SimpleNamespace(has_running=_async_false),
    )
    service = SessionService(viking_fs=viking_fs)
    service.set_session_auto_commit_config(
        SessionAutoCommitConfig(enabled=True, idle_timeout_seconds=86_400)
    )

    async def get_session(_session_id: str, _ctx: object, *, auto_create: bool = False):
        return session

    monkeypatch.setattr(service, "get", get_session)
    monkeypatch.setattr(service, "open", get_session, raising=False)
    checked_under_lock = False
    original_check = service._should_run_auto_commit

    def check_with_lock_observation(candidate: Session, reason: str) -> bool:
        nonlocal checked_under_lock
        if getattr(candidate, "_inside_test_lock", False):
            checked_under_lock = True
        return original_check(candidate, reason)

    monkeypatch.setattr(service, "_should_run_auto_commit", check_with_lock_observation)

    result = await service._run_auto_commit("session-1", ctx, reason="idle_timeout")

    assert checked_under_lock is True
    assert result is None
    assert (
        SessionMeta.from_dict(json.loads(viking_fs.meta_content)).last_message_at != old_timestamp
    )
    assert session.meta.commit_count == 0


async def _async_false(*_args, **_kwargs) -> bool:
    return False
