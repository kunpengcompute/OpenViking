"""Minimal assembly fixture independent of Reader, Commit and Hook orchestration."""

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage

from openviking.integrations.langchain.ov_context import RunIdentity


@pytest.fixture
def env():
    request = ModelRequest(
        model=FakeListChatModel(responses=["done"]),
        messages=[HumanMessage(id="m1", content="Read the mail")],
        tools=[],
        model_settings={},
    )
    return None, None, None, RunIdentity("a", "u", "p", "s", "r"), request, []
