# SPDX-License-Identifier: Apache-2.0
"""SGLang runtime metadata used by diagnostics and desktop status views."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from .sglang_runtime import SGLANG_COMMIT, SGLANG_VERSION


def get_sglang_runtime_info() -> dict[str, Any]:
    """Describe the pinned runtime without loading a model or probing hardware."""

    try:
        installed_version = version("sglang")
    except PackageNotFoundError:
        installed_version = "Not installed"
    return {
        "runtime": "SGLang",
        "version": installed_version,
        "required_version": SGLANG_VERSION,
        "commit": SGLANG_COMMIT,
        "available": installed_version == SGLANG_VERSION,
    }
