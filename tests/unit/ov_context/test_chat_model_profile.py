"""Session model limits do not mutate shared model profiles."""

from copy import deepcopy

from langchain_openai import ChatOpenAI

from examples.deepagents_ov_context.chat.model import windowed_model


def test_window_is_local_to_each_graph():
    shared = ChatOpenAI(api_key="offline", model="offline-model")
    original = deepcopy(shared.profile)
    first = windowed_model(shared, 256000)
    second = windowed_model(shared, 128000)
    assert shared.profile == original
    assert first is not shared and second is not shared
    assert first.profile["max_input_tokens"] == 256000
    assert second.profile["max_input_tokens"] == 128000
    first.profile["max_input_tokens"] = 64000
    assert second.profile["max_input_tokens"] == 128000
    assert shared.profile == original


def test_session_window_respects_provider_limit_and_preserves_capabilities():
    shared = ChatOpenAI(
        api_key="offline", profile={"max_input_tokens": 128000, "tool_calling": True}
    )
    local = windowed_model(shared, 256000)
    assert local is not shared
    assert local.profile == {"max_input_tokens": 128000, "tool_calling": True}
    assert windowed_model(local, 64000).profile["max_input_tokens"] == 64000
    assert shared.profile["max_input_tokens"] == 128000
