"""Budget formulas and bounded operation observations."""

import logging
from copy import deepcopy

import pytest

from openviking.integrations.langchain.ov_context import BudgetSpec, LatencySamples, Metrics
from openviking.integrations.langchain.ov_context.token_estimation import (
    PayloadUsageCounter,
    payload_text_budget,
)


def test_budget_derives_distinct_history_capacity_and_input_limits():
    budget = BudgetSpec(256000, 8192, 1024, base_context_ratio=0.3, capacity_ratio=0.9)
    assert budget.base_context_limit == 76800
    assert budget.capacity_limit == 230400
    assert budget.input_limit == 246784
    assert BudgetSpec(256000, 8192, 1024, max_input_tokens=240000).input_limit == 240000


@pytest.mark.parametrize(
    "overrides",
    [
        {"context_window": True},
        {"output_reserve": -1},
        {"safety_margin": 0.5},
        {"base_context_ratio": 0.95},
        {"capacity_ratio": float("nan")},
        {"max_input_tokens": 1000},
    ],
)
def test_inconsistent_budget_rejected(overrides):
    with pytest.raises(ValueError):
        BudgetSpec(**{**{"context_window": 256000, "output_reserve": 8192}, **overrides})


def test_usage_calibration_applies_only_to_matching_payload_prefix():
    payload = {"model": "test", "messages": [{"role": "user", "content": "中文" * 1000}]}
    counter = PayloadUsageCounter()
    assert counter.estimate(payload) == payload_text_budget(payload)
    counter.observe(payload, 1800)
    grown = deepcopy(payload)
    grown["messages"].append({"role": "assistant", "content": "done"})
    assert (
        counter.estimate(grown)
        == 1800 + payload_text_budget(grown) - payload_text_budget(payload) + 256
    )
    grown["messages"][0]["content"] = "new summary"
    assert counter.estimate(grown) == payload_text_budget(grown)


def test_latency_is_bounded_and_partitioned_by_status():
    samples = LatencySamples(max_samples=2)
    for duration in (1, 2, 3):
        samples("write", duration_ms=duration)
    for invalid in (-1, float("inf"), float("nan"), "3"):
        samples("write", duration_ms=invalid)
    samples("write", status="error", duration_ms=4)
    rows = {r["status"]: r for r in samples.summary()}
    assert rows["ok"] == {
        "event": "write",
        "status": "ok",
        "count": 2,
        "sum_ms": 5,
        "avg_ms": 2.5,
        "p90_ms": 3,
        "max_ms": 3,
    }
    assert rows["error"]["count"] == 1


@pytest.mark.parametrize("error_type", [OSError, TypeError])
def test_observation_sink_failure_does_not_break_the_operation(caplog, monkeypatch, error_type):
    logger = logging.getLogger("openviking.integrations.langchain.ov_context.metrics")
    caplog.set_level(logging.WARNING, logger=logger.name)
    monkeypatch.setattr(logger, "disabled", False)
    monkeypatch.setattr(logger, "handlers", [caplog.handler])

    def broken(event, **fields):
        raise error_type("sink failed")

    Metrics(broken).emit(
        "write",
        status="ok",
        session_id="session-1",
        run_id="run-1",
        account_id="account-1",
        user_id="user-1",
        actor_peer_id="peer-1",
        model_call_id="call-1",
        prompt="private-message-body",
        api_key="private-credential",
        payload=object(),
    )
    record = next(record for record in caplog.records if "metrics_sink_failed" in record.message)
    assert "event=write" in record.message
    for value in ("session-1", "run-1", "account-1", "user-1", "peer-1", "call-1"):
        assert value in record.message
    assert f"error_type={error_type.__name__}" in record.message
    assert record.exc_info[0] is error_type
    assert record.exc_info[2] is not None
    assert "broken" in caplog.text
    assert "private-message-body" not in caplog.text
    assert "private-credential" not in caplog.text


def test_metrics_does_not_intercept_cancellation():
    from asyncio import CancelledError

    def cancelled(event, **fields):
        raise CancelledError

    with pytest.raises(CancelledError):
        Metrics(cancelled).emit("write")
