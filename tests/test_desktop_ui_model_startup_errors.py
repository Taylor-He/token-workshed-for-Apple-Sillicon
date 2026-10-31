# SPDX-License-Identifier: Apache-2.0
"""Model server startup failures must include safe, actionable diagnostics."""

from __future__ import annotations

import io

from vllm_mlx.desktop_ui import LocalModelServerManager


class _FailedProcess:
    pid = 12345
    returncode = 1

    def __init__(self) -> None:
        self.stdout = io.BytesIO(
            b"ImportError: [metal::load_device] No Metal device available.\n"
            b"HF_TOKEN=hf_0123456789abcdef0123456789abcdef\n"
        )

    def poll(self) -> int:
        return 1


def test_model_start_failure_includes_redacted_metal_diagnostic(monkeypatch) -> None:
    monkeypatch.setenv("TOKEN_WORKSHED_ENABLE_AUTO_TOOL_CHOICE", "0")
    monkeypatch.setattr("vllm_mlx.desktop_ui._list_listening_pids", lambda _port: [])
    monkeypatch.setattr("vllm_mlx.desktop_ui._can_connect_tcp", lambda *_args: False)
    monkeypatch.setattr(
        "vllm_mlx.desktop_ui.subprocess.Popen",
        lambda *_args, **_kwargs: _FailedProcess(),
    )
    manager = LocalModelServerManager("python", "127.0.0.1", 8000)
    monkeypatch.setattr(manager, "_managed_pids_locked", lambda: set())
    monkeypatch.setattr(
        manager,
        "_build_serve_cmd",
        lambda *_args, **_kwargs: ["python", "serve", "model"],
    )

    ok, message, process, _parser = manager._start_model_process_locked(
        "workshed/test-model",
        port=8000,
        serve_model="/managed/model",
    )

    assert ok is False
    assert process is None
    assert "No Metal device available" in message
    assert "hf_0123456789abcdef0123456789abcdef" not in message
    assert "HF_TOKEN=[redacted]" in message
