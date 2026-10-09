#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Desktop app entrypoint for token-workshed macOS .app packaging.

Launches token-workshed desktop UI with a default model when no model argument is provided.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys


def _consume_ide_deep_link() -> bool:
    """Accept the local JetBrains launcher URL without forwarding it to argparse.

    The URL only opens the desktop app; it does not carry a command, model, or
    credential. Pairing remains a separate user approval in the native UI.
    """
    requested = False
    companion = False
    forwarded = [sys.argv[0]]
    for argument in sys.argv[1:]:
        normalized = str(argument or "").strip()
        if normalized.startswith("tokenworkshed://ide/start"):
            requested = True
            continue
        if normalized.startswith("tokenworkshed://ide/companion"):
            requested = True
            companion = True
            continue
        # macOS may append a process-serial-number argument when opening an
        # application via a URL scheme; it is not a desktop_ui option.
        if normalized.startswith("-psn_"):
            continue
        forwarded.append(argument)
    if requested:
        sys.argv = forwarded
    if companion:
        os.environ["TOKEN_WORKSHED_IDE_COMPANION"] = "1"
    return requested


def _run_python_c_payload_if_needed() -> bool:
    """Emulate `python -c ...` for multiprocessing helper subprocesses."""
    try:
        c_index = sys.argv.index("-c")
    except ValueError:
        return False

    if c_index + 1 >= len(sys.argv):
        return False

    code = sys.argv[c_index + 1]
    globals_dict: dict[str, object] = {"__name__": "__main__", "__file__": "<string>"}
    exec(code, globals_dict, globals_dict)
    return True


if __name__ == "__main__":
    mp.freeze_support()

    if _run_python_c_payload_if_needed():
        raise SystemExit(0)

    _consume_ide_deep_link()

    from vllm_mlx.desktop_ui import main

    main()
