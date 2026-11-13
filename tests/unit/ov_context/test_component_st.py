"""Component acceptance shares the ST configuration and pytest failure contract."""

from argparse import Namespace

import pytest

from tests.st.ov_context import components


@pytest.mark.parametrize("compression_failed", [False, True])
async def test_compression_checks_index_only_after_success(
    tmp_path, monkeypatch, compression_failed
):
    calls = []

    class Suite:
        def __init__(self, args):
            self.results = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def case(self, name, action):
            calls.append(name)
            self.results.append(
                {"status": "failed" if compression_failed and name == "compression" else "passed"}
            )

        def __getattr__(self, name):
            async def action():
                return {}

            return action

    monkeypatch.setattr(components, "Suite", Suite)
    args = Namespace(output=tmp_path, parallelism=2, timeout=300, command="compression")
    result = await components.execute(args)
    assert result == int(compression_failed)
    assert calls == (["compression"] if compression_failed else ["compression", "index"])


def test_restart_worker_uses_explicit_st_configuration(tmp_path, monkeypatch):
    from tests.st.ov_context import component_worker

    captured = []

    async def execute(args):
        captured.append(args)
        return 0

    monkeypatch.setattr(component_worker, "execute", execute)
    args = [
        "--model-config",
        str(tmp_path / "model.json"),
        "--api-key-file",
        str(tmp_path / "separate.key"),
        "--ov-url",
        "http://localhost:1234",
        "--output",
        str(tmp_path),
        "--prefix",
        "unit",
        "--worker-phase",
        "resume",
        "--timeout",
        "420",
    ]
    assert component_worker.main(args) == 0
    assert len(captured) == 1
    configured = captured[0]
    assert configured.api_key_file == tmp_path / "separate.key"
    assert configured.model_config == tmp_path / "model.json"
    assert configured.ov_url == "http://localhost:1234"
    assert configured.command == "worker" and configured.worker_phase == "resume"
    assert configured.timeout == 420


@pytest.mark.parametrize("failed", [False, True])
async def test_pytest_component_uses_shared_environment_and_propagates_failure(
    tmp_path, monkeypatch, failed
):
    from types import SimpleNamespace

    from tests.st.ov_context.live import test_components

    seen = []

    async def execute(args):
        seen.append(args)
        return int(failed)

    monkeypatch.setattr(test_components, "execute", execute)
    monkeypatch.setenv("OV_LIVE_COMPONENT_PARALLELISM", "3")
    monkeypatch.setenv("OV_LIVE_COMPONENT_TIMEOUT", "420")
    environment = SimpleNamespace(
        url="http://localhost:9876",
        key_file=tmp_path / "elsewhere.key",
        model_config=tmp_path / "model.json",
    )
    if failed:
        with pytest.raises(AssertionError, match="writer failed"):
            await test_components.test_component_acceptance(environment, tmp_path, "writer")
    else:
        await test_components.test_component_acceptance(environment, tmp_path, "writer")
    args = seen[0]
    assert args.api_key_file == environment.key_file
    assert args.ov_url == environment.url and args.model_config == environment.model_config
    assert args.parallelism == 3 and args.timeout == 420
