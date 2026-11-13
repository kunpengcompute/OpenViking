#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Create a source checkout from a released OpenViking patch bundle."""

import sys

from patch_manager import main

if __name__ == "__main__":
    raise SystemExit(main(["materialize", *sys.argv[1:]]))
