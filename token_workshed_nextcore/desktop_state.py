# SPDX-License-Identifier: Apache-2.0
"""Shared persistent state used by the desktop launcher and integrations."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

_STATE_LOCK = threading.RLock()


def canonical_model_id(model_id: str) -> str:
    """Normalize known historical LM Studio owner aliases."""
    value = str(model_id or "").strip()
    if not value:
        return ""
    lowered = value.lower()
    for prefix in ("imstudio-community/", "1mstudio-community/"):
        if lowered.startswith(prefix):
            suffix = value.split("/", 1)[1].strip() if "/" in value else ""
            return f"lmstudio-community/{suffix}" if suffix else "lmstudio-community"
    return value


def desktop_state_file() -> Path:
    """Return the app's shared desktop-state path."""
    override = str(os.environ.get("TOKEN_WORKSHED_DESKTOP_STATE_PATH", "") or "").strip()
    if override:
        path = Path(override).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "token-workshed-nextcore"
    elif os.name == "nt":
        appdata = os.environ.get("APPDATA")
        base = (
            Path(appdata) / "token-workshed-nextcore"
            if appdata
            else Path.home() / ".token-workshed-nextcore"
        )
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME")
        base = (
            Path(xdg) / "token-workshed-nextcore"
            if xdg
            else Path.home() / ".config" / "token-workshed-nextcore"
        )
    base.mkdir(parents=True, exist_ok=True)
    return base / "desktop_state.json"


def _read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    payload["updated_at"] = int(time.time())
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=str(path.parent), delete=False
    ) as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2))
        temporary = Path(handle.name)
    try:
        os.chmod(temporary, 0o600)
    except OSError:
        pass
    temporary.replace(path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def read_desktop_state() -> dict[str, Any]:
    """Read one consistent snapshot of the shared desktop state."""
    with _STATE_LOCK:
        return _read_state(desktop_state_file())


def mutate_desktop_state(mutator: Callable[[dict[str, Any]], bool]) -> bool:
    """Apply a same-process read-modify-write transaction atomically."""
    with _STATE_LOCK:
        path = desktop_state_file()
        state = _read_state(path)
        changed = bool(mutator(state))
        if changed:
            _write_state(path, state)
        return changed


def clear_legacy_agent_settings() -> bool:
    """Remove persisted Agent configuration from the retired runtimes."""

    def remove_legacy_settings(state: dict[str, Any]) -> bool:
        changed = False
        for key in ("agent_profiles", "agent_runtime", "openclaw_enabled"):
            if key in state:
                state.pop(key, None)
                changed = True
        return changed

    return mutate_desktop_state(remove_legacy_settings)
