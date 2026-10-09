# SPDX-License-Identifier: Apache-2.0
"""Concurrency coverage for the shared Python desktop state document."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from vllm_mlx import agent_profiles, desktop_ui


def _profile(model: str = "owner/model") -> dict[str, Any]:
    runtime = {
        "ready": True,
        "tool_calling": "passed",
        "tool_call_parser": "qwen",
        "reasoning_parser": "qwen3",
        "timeout_seconds": 240,
        "probe": {},
    }
    return {
        "profile_version": 1,
        "model_id": model,
        "configured_at": 1,
        "family": "qwen3",
        "context_window": 32768,
        "chat_template": "{{ messages }}",
        "generation_defaults": {},
        "tool_policy": {},
        "openclaw": dict(runtime),
        "hermes": dict(runtime),
        "warnings": [],
    }


def test_profile_and_desktop_preference_writes_share_one_transaction_lock(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "desktop_state.json"
    monkeypatch.setenv("TOKEN_WORKSHED_DESKTOP_STATE_PATH", str(state_path))

    profile_read_started = threading.Event()
    release_profile_write = threading.Event()
    preference_started = threading.Event()
    preference_finished = threading.Event()
    failures: list[BaseException] = []
    original_read = agent_profiles._read_state

    def blocking_read(path: Path) -> dict[str, Any]:
        state = original_read(path)
        if threading.current_thread().name == "profile-writer":
            profile_read_started.set()
            if not release_profile_write.wait(timeout=2):
                raise TimeoutError("profile transaction was not released")
        return state

    monkeypatch.setattr(agent_profiles, "_read_state", blocking_read)

    def write_profile() -> None:
        try:
            agent_profiles.save_agent_profile(_profile())
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    def write_preference() -> None:
        try:
            preference_started.set()
            desktop_ui._save_last_model_preference("owner/model")
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)
        finally:
            preference_finished.set()

    profile_thread = threading.Thread(target=write_profile, name="profile-writer")
    preference_thread = threading.Thread(
        target=write_preference,
        name="preference-writer",
    )
    profile_thread.start()
    try:
        assert profile_read_started.wait(timeout=2)
        preference_thread.start()
        assert preference_started.wait(timeout=2)
        assert not preference_finished.wait(timeout=0.05)
    finally:
        release_profile_write.set()

    profile_thread.join(timeout=2)
    preference_thread.join(timeout=2)
    assert not profile_thread.is_alive()
    assert not preference_thread.is_alive()
    assert failures == []

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["last_model"] == "owner/model"
    assert state["agent_profiles"]["owner/model"]["model_id"] == "owner/model"
    assert desktop_ui._desktop_state_file() == state_path
