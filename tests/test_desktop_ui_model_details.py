# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_mlx.desktop_ui import (
    LocalModelServerManager,
    _build_model_details,
    _CombinedManagedModelResolver,
)


def test_build_model_details_includes_runtime_metadata() -> None:
    details = _build_model_details(
        ["mlx-community/Llama-3.2-3B-Instruct-4bit", "local-scratch"],
        active_model="mlx-community/Llama-3.2-3B-Instruct-4bit",
        running_models=[
            {
                "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
                "server_url": "http://127.0.0.1:8000",
                "kind": "primary",
                "active": True,
                "reasoning_parser": "qwen3",
                "tool_call_parser": "qwen",
            }
        ],
        runtime_config={"kv_cache_quantization": True},
    )

    active = details[0]
    assert active["id"] == "mlx-community/Llama-3.2-3B-Instruct-4bit"
    assert active["status"] == "active"
    assert active["source"] == "Hugging Face"
    assert active["backend"] == "vllm-mlx"
    assert active["quantization"] == "4-bit + KV cache quantization"
    assert active["running"] is True
    assert active["server_url"] == "http://127.0.0.1:8000"
    assert active["kind"] == "primary"
    assert active["role"] == "primary"
    assert active["reasoning_parser"] == "qwen3"
    assert active["tool_call_parser"] == "qwen"

    local = next(item for item in details if item["id"] == "local-scratch")
    assert local["status"] == "local"
    assert local["source"] == "Local cache"
    assert local["quantization"] == "KV cache quantization"


def test_build_model_details_adds_running_model_missing_from_available_list() -> None:
    details = _build_model_details(
        [],
        active_model="",
        running_models=[
            {
                "model": "Qwen/Qwen3-4B-Instruct",
                "server_url": "http://127.0.0.1:8001",
                "kind": "extra",
            }
        ],
        runtime_config={},
    )

    assert [item["id"] for item in details] == ["Qwen/Qwen3-4B-Instruct"]
    assert details[0]["status"] == "running"
    assert details[0]["role"] == "extra"


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"quantization": {"bits": 3, "group_size": 64}}, "3-bit"),
        ({"quantization": {"bits": 4, "group_size": 128}}, "4-bit"),
        ({"quantization": {"bits": 6, "group_size": 32}}, "6-bit"),
        ({"quantization": {"bits": 8, "group_size": 64}}, "8-bit"),
        ({"quantization_config": {"bits": "8"}}, "8-bit"),
    ],
)
def test_managed_model_details_read_config_through_manager_resolver(
    monkeypatch, tmp_path: Path, config: dict, expected: str
) -> None:
    monkeypatch.setenv("TOKEN_WORKSHED_DESKTOP_STATE_PATH", str(tmp_path / "state.json"))
    model_path = tmp_path / "managed-model"
    model_path.mkdir()
    config_path = model_path / "config.json"
    config_text = json.dumps(config)
    config_path.write_text(config_text, encoding="utf-8")
    model_id = "workshed/ui-quant-granite350m-20261008"
    quantization = SimpleNamespace(resolve_model=lambda _model: None)
    workshed = SimpleNamespace(
        resolve_model=lambda model: str(model_path) if model == model_id else None
    )
    manager = LocalModelServerManager("python", "127.0.0.1", 8000)
    manager.set_managed_model_resolver(
        _CombinedManagedModelResolver(quantization, workshed)
    )
    runtime_config = {"kv_cache_quantization": True, "max_num_seqs": 3}
    runtime_before = deepcopy(runtime_config)
    details = _build_model_details(
        [model_id, "org/other-model"],
        active_model=model_id,
        running_models=[],
        runtime_config=runtime_config,
        managed_model_resolver=manager._resolve_managed_model,
    )

    assert details[0]["source"] == "Workshed managed"
    assert details[0]["quantization"] == f"{expected} + KV cache quantization"
    assert details[0]["active"] is True
    assert details[1]["source"] == "Hugging Face"
    assert runtime_config == runtime_before
    assert config_path.read_text(encoding="utf-8") == config_text
    assert not (tmp_path / "state.json").exists()


def test_local_model_details_prefer_config_bits_over_directory_name(tmp_path: Path) -> None:
    model_path = tmp_path / "model-8bit"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps({"quantization": {"bits": 4}}), encoding="utf-8"
    )
    details = _build_model_details(
        [str(model_path)],
        active_model="",
        running_models=[],
        runtime_config={},
    )

    assert details[0]["source"] == "Local cache"
    assert details[0]["quantization"] == "4-bit"


@pytest.mark.parametrize(
    "config_text",
    [
        None,
        "not json",
        "[]",
        "{}",
        '{"quantization": {"bits": true}}',
        '{"quantization": {"bits": 4.5}}',
        '{"quantization": {"bits": 0}}',
        '{"quantization": {"bits": 64}}',
        '{"quantization": {"bits": "²"}}',
        pytest.param(
            '{"quantization": {"bits": "' + "1" * 5000 + '"}}', id="long-bits"
        ),
        pytest.param(" " * (2 * 1024 * 1024 + 1), id="oversized-config"),
    ],
)
def test_managed_model_details_fall_back_when_config_is_unusable(
    tmp_path: Path, config_text: str | None
) -> None:
    if config_text is not None:
        (tmp_path / "config.json").write_text(config_text, encoding="utf-8")
    details = _build_model_details(
        ["workshed/model-8bit", "workshed/unknown-model"],
        active_model="",
        running_models=[],
        runtime_config={},
        managed_model_resolver=lambda _model: str(tmp_path),
    )

    assert details[0]["source"] == "Workshed managed"
    assert details[0]["quantization"] == "8-bit"
    assert details[1]["quantization"] == "Unknown"


def test_model_details_keep_name_metadata_when_resolver_fails() -> None:
    def broken_resolver(_model: str) -> str | None:
        raise OSError("registry unavailable")

    details = _build_model_details(
        ["mlx-community/model-4bit"],
        active_model="",
        running_models=[],
        runtime_config={},
        managed_model_resolver=broken_resolver,
    )

    assert details[0]["source"] == "Hugging Face"
    assert details[0]["quantization"] == "4-bit"
