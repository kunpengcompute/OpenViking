# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""VLM streaming is not a supported configuration contract."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from openviking.models.vlm.backends.openai_vlm import OpenAIVLM
from openviking_cli.utils.config.vlm_config import VLMConfig


@pytest.mark.parametrize(
    "config",
    [
        {
            "provider": "openai",
            "model": "test-model",
            "api_key": "test-key",
            "stream": False,
        },
        {
            "model": "test-model",
            "providers": {"openai": {"api_key": "test-key", "stream": True}},
        },
        {
            "model": "test-model",
            "credentials": [
                {
                    "provider": "openai",
                    "model": "test-model",
                    "api_key": "test-key",
                    "stream": True,
                }
            ],
        },
        {
            "provider": "openai",
            "model": "test-model",
            "api_key": "test-key",
            "extra_request_body": {"stream": True},
        },
        {
            "model": "test-model",
            "providers": {
                "openai": {
                    "api_key": "test-key",
                    "extra_request_body": {"stream": True},
                }
            },
        },
    ],
)
def test_vlm_config_rejects_stream(config):
    with pytest.raises(ValidationError, match="stream.*not supported"):
        VLMConfig(**config)


class MockUsage:
    """Mock usage object."""

    def __init__(self, prompt_tokens=0, completion_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


@patch("openviking.models.vlm.backends.openai_vlm.openai.OpenAI")
def test_default_sync_completion_returns_full_response(mock_openai_class):
    """Default VLM calls return a complete response without a stream parameter."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Non-streaming response"
    mock_response.usage = MockUsage(prompt_tokens=5, completion_tokens=10)
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_class.return_value = mock_client

    vlm = OpenAIVLM(
        {
            "api_key": "sk-test",
            "api_base": "https://api.openai.com/v1",
        }
    )

    result = vlm.get_completion("test prompt")

    assert result == "Non-streaming response"
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert "stream" not in call_kwargs


@patch("openviking.models.vlm.backends.openai_vlm.openai.AsyncOpenAI")
async def test_default_async_completion_returns_full_response(mock_async_openai_class):
    """Default async VLM calls return a complete response."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Async non-streaming"
    mock_response.usage = MockUsage(prompt_tokens=5, completion_tokens=5)

    async def mock_create(*args, **kwargs):
        return mock_response

    mock_client.chat.completions.create = mock_create
    mock_async_openai_class.return_value = mock_client

    vlm = OpenAIVLM(
        {
            "api_key": "sk-test",
            "api_base": "https://api.openai.com/v1",
        }
    )

    result = await vlm.get_completion_async("test prompt")

    assert result == "Async non-streaming"


def test_vlm_config_max_retries_defaults_to_three():
    """VLMConfig should default max_retries to 3."""
    from openviking_cli.utils.config.vlm_config import VLMConfig

    config = VLMConfig(
        model="gpt-4o",
        provider="openai",
        providers={
            "openai": {
                "api_key": "sk-test",
            }
        },
    )

    assert config.max_retries == 3
    assert config._build_vlm_config_dict()["max_retries"] == 3
