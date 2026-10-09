# SPDX-License-Identifier: Apache-2.0
"""Persistent, non-secret Agent Profile storage.

Profiles are deliberately stored beside the desktop launcher state instead of
in the native UI preference file.  The Python chat service is the runtime
consumer, while the native UI is only a controller, so one small shared state
document prevents the two from drifting apart.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROFILE_VERSION = 1
_VALID_TOOL_CALLING_STATES = frozenset({"passed", "degraded"})
_STATE_LOCK = threading.RLock()
_SECRET_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "token",
        "secret",
        "password",
        "authorization",
        "credential",
        "credentials",
        "environment",
        "env",
    }
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b((?:api[_ -]?key|access[_ -]?token|auth(?:orization)?|password|secret)\b\s*[:=]\s*)(?:bearer\s+)?(?:['\"]?)[^\s,;\"']+"
)
_SECRET_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{6,}")
_SECRET_LITERAL_RE = re.compile(
    r"(?i)\b(?:sk-[A-Za-z0-9_-]{8,}|(?:hf|ghp|gho)_[A-Za-z0-9_-]{8,})\b"
)
_SECRET_QUERY_RE = re.compile(
    r"(?i)([?&](?:api[_ -]?key|access[_ -]?token|token|secret)=[^&#\s]+)"
)


def canonical_model_id(model_id: str) -> str:
    """Return the model key used by desktop model management.

    Keep this intentionally conservative: model revisions and aliases are
    meaningful to users, while the two historical LM Studio owner typos are
    known to be accidental and are already normalised by ``desktop_ui``.
    """
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
    """Locate the existing desktop launcher state file.

    The override is intentionally private to tests/developer diagnostics and
    avoids putting test profiles into a user's real launcher state.
    """
    override = str(
        os.environ.get("TOKEN_WORKSHED_DESKTOP_STATE_PATH", "") or ""
    ).strip()
    if override:
        path = Path(override).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "token-workshed"
    elif os.name == "nt":
        appdata = os.environ.get("APPDATA")
        base = (
            Path(appdata) / "token-workshed"
            if appdata
            else Path.home() / ".token-workshed"
        )
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME")
        base = (
            Path(xdg) / "token-workshed"
            if xdg
            else Path.home() / ".config" / "token-workshed"
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
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        delete=False,
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


def mutate_desktop_state(
    mutator: Callable[[dict[str, Any]], bool],
) -> bool:
    """Atomically apply a same-process read-modify-write transaction.

    ``mutator`` receives the current mutable state and returns whether it
    changed.  Keeping the read, mutation, and atomic replace under the same
    re-entrant lock prevents desktop preferences and Agent Profiles from
    overwriting one another when background jobs finish concurrently.
    """
    with _STATE_LOCK:
        path = desktop_state_file()
        state = _read_state(path)
        changed = bool(mutator(state))
        if changed:
            _write_state(path, state)
        return changed


def _redact_secret_text(value: str) -> str:
    """Remove token-shaped data that can arrive in diagnostics text."""
    text = _SECRET_ASSIGNMENT_RE.sub(r"\1[redacted]", value)
    text = _SECRET_BEARER_RE.sub("Bearer [redacted]", text)
    text = _SECRET_LITERAL_RE.sub("[redacted]", text)
    return _SECRET_QUERY_RE.sub(
        lambda match: match.group(1).split("=", 1)[0] + "=[redacted]", text
    )


def _json_safe(value: Any) -> Any:
    """Copy JSON-compatible data while stripping secrets and environment maps."""
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            lowered = key.lower().replace("-", "_")
            if lowered in _SECRET_KEYS or lowered.endswith(
                ("_api_key", "_token", "_secret", "_password")
            ):
                continue
            output[key] = _json_safe(item)
        return output
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, str):
        return _redact_secret_text(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)


def _valid_profile(profile: Any, model_id: str) -> dict[str, Any] | None:
    if not isinstance(profile, dict):
        return None
    canonical = canonical_model_id(model_id)
    if not canonical:
        return None
    if int(profile.get("profile_version", 0) or 0) != PROFILE_VERSION:
        return None
    if canonical_model_id(str(profile.get("model_id", "") or "")) != canonical:
        return None
    openclaw = profile.get("openclaw")
    hermes = profile.get("hermes")
    if not isinstance(openclaw, dict) or not isinstance(hermes, dict):
        return None
    if not bool(openclaw.get("ready")) or not bool(hermes.get("ready")):
        return None
    for runtime in (openclaw, hermes):
        # Profiles written before the capability field was introduced only
        # recorded ``ready``.  Keep those model profiles usable and let the
        # runtime treat the missing state as the historical "passed" value.
        tool_state = str(runtime.get("tool_calling") or "").strip().lower()
        if tool_state and tool_state not in _VALID_TOOL_CALLING_STATES:
            return None
    return _json_safe(profile)


def load_agent_profile(model_id: str) -> dict[str, Any] | None:
    """Read a valid profile for ``model_id`` from shared desktop state."""
    canonical = canonical_model_id(model_id)
    if not canonical:
        return None
    state = read_desktop_state()
    profiles = state.get("agent_profiles")
    if not isinstance(profiles, dict):
        return None
    return _valid_profile(profiles.get(canonical), canonical)


def save_agent_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Persist one profile and return the redacted stored representation."""
    model_id = canonical_model_id(str(profile.get("model_id", "") or ""))
    if not model_id:
        raise ValueError("Agent Profile requires a model_id.")
    cleaned = _json_safe(profile)
    cleaned["profile_version"] = PROFILE_VERSION
    cleaned["model_id"] = model_id
    if not _valid_profile(cleaned, model_id):
        raise ValueError("Agent Profile is incomplete; both runtimes must be ready.")

    def store(state: dict[str, Any]) -> bool:
        profiles = state.get("agent_profiles")
        if not isinstance(profiles, dict):
            profiles = {}
        profiles[model_id] = cleaned
        state["agent_profiles"] = profiles
        return True

    mutate_desktop_state(store)
    return dict(cleaned)


def delete_agent_profile(model_id: str) -> bool:
    """Delete only the model profile, never model data or launcher settings."""
    canonical = canonical_model_id(model_id)
    if not canonical:
        return False

    def remove(state: dict[str, Any]) -> bool:
        profiles = state.get("agent_profiles")
        if not isinstance(profiles, dict) or canonical not in profiles:
            return False
        profiles.pop(canonical, None)
        if profiles:
            state["agent_profiles"] = profiles
        else:
            state.pop("agent_profiles", None)
        return True

    return mutate_desktop_state(remove)
