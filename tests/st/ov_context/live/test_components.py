"""Writer, context and tenant acceptance under the common deployed-service ST run."""

from __future__ import annotations

import argparse
import os
import uuid

import pytest

from tests.st.ov_context.components import execute


@pytest.mark.parametrize(
    "component",
    ["writer", "matrix", "compression", "restart", "collision", "logs"],
    ids=["writer", "matrix", "compression-index", "restart", "collision", "logs"],
)
async def test_component_acceptance(live_environment, tmp_path, component):
    args = argparse.Namespace(
        command=component,
        output=tmp_path,
        model_config=live_environment.model_config,
        api_key_file=live_environment.key_file,
        ov_url=live_environment.url,
        prefix="st-" + uuid.uuid4().hex[:12],
        parallelism=int(os.environ.get("OV_LIVE_COMPONENT_PARALLELISM", "4")),
        timeout=float(os.environ.get("OV_LIVE_COMPONENT_TIMEOUT", "300")),
        source="working-tree",
        worker_phase="pause",
    )
    result = await execute(args)
    assert result == 0, f"{component} failed; see {tmp_path}/results-{component}.json"
