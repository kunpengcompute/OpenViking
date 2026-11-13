"""The checked-in .env template supplies policy settings for contract tests."""

from pathlib import Path

from examples.deepagents_ov_context.chat.config import load_commit_defaults

EXAMPLE_ENV = Path(__file__).parents[2] / "examples/deepagents_ov_context/config/chat.example.env"
EXAMPLE_COMMIT_CONFIG = load_commit_defaults(EXAMPLE_ENV)
