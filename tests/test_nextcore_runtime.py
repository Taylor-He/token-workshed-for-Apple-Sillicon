from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException

from token_workshed_nextcore import (
    desktop_state,
    jetbrains_integration,
    sglang_runtime,
)
from token_workshed_nextcore.desktop_ui import (
    LocalModelServerManager,
)


class NativeFIMTokenizer:
    special_tokens_map = {
        "additional_special_tokens": [
            "<|fim_prefix|>",
            "<|fim_suffix|>",
            "<|fim_middle|>",
        ]
    }


def test_legacy_agent_runtime_state_is_removed_without_touching_sglang_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state_path = tmp_path / "desktop_state.json"
    state_path.write_text(
        '{"agent_profiles":{"granite":{"openclaw":{},"hermes":{}}},'
        '"agent_runtime":"hermes","openclaw_enabled":true,'
        '"last_model":"ibm-granite/granite-4.0-1b",'
        '"tool_call_parser_overrides":{"granite":{"parser":"hermes","verified":true}}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("TOKEN_WORKSHED_DESKTOP_STATE_PATH", str(state_path))

    assert desktop_state.clear_legacy_agent_settings() is True
    cleaned = desktop_state.read_desktop_state()

    assert "agent_profiles" not in cleaned
    assert "agent_runtime" not in cleaned
    assert "openclaw_enabled" not in cleaned
    assert cleaned["last_model"] == "ibm-granite/granite-4.0-1b"
    assert cleaned["tool_call_parser_overrides"]["granite"]["parser"] == "hermes"
    assert desktop_state.clear_legacy_agent_settings() is False


def test_server_command_targets_pinned_runtime_backends_and_parser_aliases() -> None:
    mlx = sglang_runtime.server_command(
        "/venv/bin/python", "model", "127.0.0.1", 8010, tool_parser="qwen3"
    )
    assert mlx[:3] == [
        "/venv/bin/python",
        "-m",
        "token_workshed_nextcore.sglang_runtime",
    ]
    assert "mlx" in mlx
    assert "--tool-call-parser" in mlx
    assert mlx[mlx.index("--tool-call-parser") + 1] == "qwen"
    assert "--enable-auto-tool-choice" not in mlx
    assert "--cuda-graph-backend-decode" in mlx

    torch = sglang_runtime.server_command(
        "/venv/bin/python", "model", "127.0.0.1", 8010, backend="torch-mps"
    )
    assert "torch-mps" in torch
    assert "--device" in torch
    assert torch[torch.index("--device") + 1] == "mps"


@pytest.mark.parametrize("raise_error", [False, True])
def test_workbench_runtime_restores_models_after_success_or_failure(
    monkeypatch: pytest.MonkeyPatch, raise_error: bool
) -> None:
    manager = LocalModelServerManager(sys.executable, "127.0.0.1", 18102)
    manager._proc = object()  # the state is supplied by the lifecycle stubs below
    manager._current_model = "chat-model"
    manager._active_model = "extra-model"
    started: list[tuple[str, str]] = []
    restored: dict[str, Any] = {}
    temporary_process = object()

    monkeypatch.setattr(manager, "_is_primary_running_locked", lambda: True)
    monkeypatch.setattr(manager, "concurrent_models", lambda: ["extra-model"])

    def stop() -> None:
        manager._proc = None
        manager._current_model = ""
        manager._active_model = ""

    monkeypatch.setattr(manager, "stop", stop)

    def start_model(model: str, **kwargs: Any):
        started.append((model, kwargs["backend"]))
        return True, "ready", temporary_process, None

    monkeypatch.setattr(manager, "_start_model_process_locked", start_model)
    monkeypatch.setattr(
        "token_workshed_nextcore.desktop_ui._terminate_process",
        lambda proc: restored.setdefault("terminated", proc),
    )

    def restore_primary(model: str):
        restored["primary"] = model
        manager._proc = object()
        manager._current_model = model
        return True, "restored"

    monkeypatch.setattr(manager, "_start_locked", restore_primary)

    def restore_extras(models: list[str]):
        restored["extras"] = models
        return True, "restored"

    monkeypatch.setattr(manager, "set_concurrent_models", restore_extras)

    def restore_active(model: str) -> None:
        restored["active"] = model
        manager._active_model = model

    monkeypatch.setattr(manager, "_select_running_model_locked", restore_active)

    if raise_error:
        with pytest.raises(ValueError, match="worker failed"):
            with manager.workbench_runtime("score-model", backend="torch-mps"):
                raise ValueError("worker failed")
    else:
        with manager.workbench_runtime("score-model", backend="torch-mps"):
            assert manager._workbench_reserved is True

    assert started == [("score-model", "torch-mps")]
    assert restored == {
        "terminated": temporary_process,
        "primary": "chat-model",
        "extras": ["extra-model"],
        "active": "extra-model",
    }
    assert manager._workbench_reserved is False
    assert manager._current_model == "chat-model"
    assert manager._active_model == "extra-model"


def test_fim_completion_translates_suffix_and_persists_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored: dict[str, Any] = {}
    monkeypatch.setattr(
        jetbrains_integration,
        "_load_fim_tokenizer",
        lambda _model, _manager: NativeFIMTokenizer(),
    )
    monkeypatch.setattr(
        jetbrains_integration, "mutate_desktop_state", lambda fn: fn(stored)
    )
    body = {
        "model": "local-model",
        "prompt": "def answer():\n    ",
        "suffix": "\nprint(answer())",
        "max_tokens": 16,
    }

    prepared = jetbrains_integration._prepare_fim_completion(body, object())

    assert prepared["prompt"] == (
        "<|fim_prefix|>def answer():\n    <|fim_suffix|>\nprint(answer())"
        "<|fim_middle|>"
    )
    assert "suffix" not in prepared
    assert body["suffix"] == "\nprint(answer())"
    record = stored[jetbrains_integration.INTEGRATION_STATE_KEY][
        "completion_diagnostics"
    ][-1]
    assert record["native_fim"] is True
    assert record["model"] == "local-model"


def test_fim_completion_rejects_non_string_prefix_or_suffix() -> None:
    with pytest.raises(HTTPException) as error:
        jetbrains_integration._prepare_fim_completion(
            {
                "model": "local-model",
                "prompt": ["not", "a", "string"],
                "suffix": "tail",
            },
            object(),
        )
    assert error.value.status_code == 400


def test_sglang_launcher_rejects_an_unpinned_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sglang_runtime, "version", lambda _name: "0.6.0")
    monkeypatch.setattr(sys, "argv", ["sglang_runtime", "--backend", "mlx"])
    with pytest.raises(SystemExit, match="requires SGLang Runtime 0.5.22"):
        sglang_runtime.main()
