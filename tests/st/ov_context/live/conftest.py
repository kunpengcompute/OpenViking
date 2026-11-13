"""Live tests are explicitly enabled; configured service failures fail the run."""

import json
import os
from pathlib import Path

import pytest
import pytest_asyncio

from .support import LiveEnvironment, LiveService


@pytest.fixture(scope="session")
def live_environment():
    if os.environ.get("OV_LIVE_ENABLED") != "1":
        pytest.skip("Set OV_LIVE_ENABLED=1 for deployed OV/model integration tests")
    names = ("OV_LIVE_URL", "OV_LIVE_API_KEY_FILE", "OV_LIVE_MODEL_CONFIG")
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        pytest.fail("Live configuration missing: " + ", ".join(missing))
    return LiveEnvironment(
        os.environ["OV_LIVE_URL"],
        Path(os.environ["OV_LIVE_API_KEY_FILE"]),
        Path(os.environ["OV_LIVE_MODEL_CONFIG"]),
    )


@pytest_asyncio.fixture
async def live(live_environment, tmp_path):
    async with LiveService(live_environment, tmp_path) as service:
        yield service


@pytest_asyncio.fixture
async def ov_process(live_environment, tmp_path):
    from .extended_support import DockerOV

    control = DockerOV(live_environment)
    await control.inspect()
    try:
        yield control
    finally:
        try:
            await control.start()
            (tmp_path / "process-events.json").write_text(json.dumps(control.events))
        finally:
            await control.close()
