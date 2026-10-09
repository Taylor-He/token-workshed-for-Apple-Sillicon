# SPDX-License-Identifier: Apache-2.0
"""vllm-mlx command-line behavior used by the native Terminal page."""

from __future__ import annotations

import time
from pathlib import Path

from vllm_mlx import desktop_ui


def test_desktop_terminal_root_points_to_project_checkout() -> None:
    assert desktop_ui.TOKEN_WORKSHED_ROOT.name == "token-workshed-0.2.0"
    assert (desktop_ui.TOKEN_WORKSHED_ROOT / "vllm_mlx").is_dir()


def _wait_for_terminal_output(fragment: str, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(
            fragment in line
            for line in desktop_ui._DEVELOPER_TERMINAL_OUTPUT.tail(300)
        ):
            return True
        time.sleep(0.03)
    return False


def test_developer_terminal_keeps_shell_state_and_streams_output(tmp_path: Path) -> None:
    session = desktop_ui._DeveloperTerminalSession(tmp_path)
    try:
        assert session.send("printf 'terminal-ready\\n'")
        assert _wait_for_terminal_output("terminal-ready")

        assert session.send(f"cd {tmp_path}")
        assert session.send("pwd")
        assert _wait_for_terminal_output(str(tmp_path))
    finally:
        session.stop()


def test_terminal_output_is_raw_and_does_not_include_desktop_diagnostics(
    tmp_path: Path,
) -> None:
    desktop_ui._developer_log("diagnostic-only-not-terminal")
    session = desktop_ui._DeveloperTerminalSession(tmp_path)
    try:
        assert session.send("printf 'raw-terminal\\n'")
        assert _wait_for_terminal_output("raw-terminal")
        deadline = time.monotonic() + 3.0
        output = desktop_ui._DEVELOPER_TERMINAL_OUTPUT.tail(300)
        while "raw-terminal" not in output and time.monotonic() < deadline:
            time.sleep(0.03)
            output = desktop_ui._DEVELOPER_TERMINAL_OUTPUT.tail(300)
        assert not any("diagnostic-only-not-terminal" in line for line in output)
        assert any(line == "raw-terminal" for line in output)
        assert not any(line[:2].isdigit() and line[2] == ":" for line in output)
    finally:
        session.stop()


def test_developer_terminal_rejects_empty_or_oversized_commands(tmp_path: Path) -> None:
    session = desktop_ui._DeveloperTerminalSession(tmp_path)
    try:
        try:
            session.send("   ")
        except ValueError as exc:
            assert "empty" in str(exc).lower()
        else:
            raise AssertionError("empty command should be rejected")

        try:
            session.send("x" * 8_001)
        except ValueError as exc:
            assert "8000" in str(exc)
        else:
            raise AssertionError("oversized command should be rejected")
    finally:
        session.stop()
