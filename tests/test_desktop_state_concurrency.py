# SPDX-License-Identifier: Apache-2.0
"""Concurrency coverage for the shared Python desktop state document."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from token_workshed_nextcore import desktop_state, desktop_ui


def test_desktop_state_mutations_share_one_transaction_lock(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "desktop_state.json"
    monkeypatch.setenv("TOKEN_WORKSHED_DESKTOP_STATE_PATH", str(state_path))

    state_read_started = threading.Event()
    release_state_write = threading.Event()
    preference_started = threading.Event()
    preference_finished = threading.Event()
    failures: list[BaseException] = []
    original_read = desktop_state._read_state

    def blocking_read(path: Path) -> dict[str, Any]:
        state = original_read(path)
        if threading.current_thread().name == "state-writer":
            state_read_started.set()
            if not release_state_write.wait(timeout=2):
                raise TimeoutError("state transaction was not released")
        return state

    monkeypatch.setattr(desktop_state, "_read_state", blocking_read)

    def write_state() -> None:
        try:
            desktop_state.mutate_desktop_state(
                lambda state: state.update({"state_writer": "complete"}) or True
            )
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

    state_thread = threading.Thread(target=write_state, name="state-writer")
    preference_thread = threading.Thread(
        target=write_preference,
        name="preference-writer",
    )
    state_thread.start()
    try:
        assert state_read_started.wait(timeout=2)
        preference_thread.start()
        assert preference_started.wait(timeout=2)
        assert not preference_finished.wait(timeout=0.05)
    finally:
        release_state_write.set()

    state_thread.join(timeout=2)
    preference_thread.join(timeout=2)
    assert not state_thread.is_alive()
    assert not preference_thread.is_alive()
    assert failures == []

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["last_model"] == "owner/model"
    assert state["state_writer"] == "complete"
    assert desktop_ui._desktop_state_file() == state_path
