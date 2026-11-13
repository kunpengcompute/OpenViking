"""Known CLI documentation corrections; upstream Skill bytes stay unchanged."""

BATCH_GUIDE_PATCH = "agent-browser-0.27.1-batch-stdin"
# Exact example returned by the installed 0.27.1 `skills get core --full`.
_OLD_BATCH_EXAMPLE = 'agent-browser batch \\\n  \'["open"]\' \\\n  \'["network","route","*","--abort","--resource-type","script"]\' \\\n  \'["cookies","set","--curl","cookies.curl","--domain","localhost"]\' \\\n  \'["navigate","http://localhost:3000/target"]\''
_CORRECT_BATCH_EXAMPLE = 'cat <<\'JSON\' | agent-browser batch --bail --json\n[\n  ["open"],\n  ["network", "route", "*", "--abort", "--resource-type", "script"],\n  ["cookies", "set", "--curl", "cookies.curl", "--domain", "localhost"],\n  ["navigate", "http://localhost:3000/target"]\n]\nJSON'


def correct_browser_guide(argv: list[str], stdout: str) -> tuple[str, str | None]:
    """Correct the known example only in successful core-guide output."""
    if argv != ["skills", "get", "core", "--full"] or _OLD_BATCH_EXAMPLE not in stdout:
        return stdout, None
    return stdout.replace(_OLD_BATCH_EXAMPLE, _CORRECT_BATCH_EXAMPLE, 1), BATCH_GUIDE_PATCH
