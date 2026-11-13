"""Print the exact task used by the Chat runner for copying or inspection."""

import argparse

from . import CASES, PHASES, render_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--page-url", default="http://127.0.0.1:19482")
    parser.add_argument("--anchor")
    args = parser.parse_args()
    try:
        print(render_task(**vars(args)))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
