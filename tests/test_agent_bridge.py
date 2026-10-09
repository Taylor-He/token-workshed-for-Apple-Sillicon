from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_mlx.agent_bridge.command_builder import build_commands
from vllm_mlx.agent_bridge.config_writer import (
    detect_api_mode,
    write_hermes_config,
    write_openclaw_config,
)
from vllm_mlx.agent_bridge.defaults import BridgePaths
from vllm_mlx.agent_bridge.family_detect import detect_model_family
from vllm_mlx.agent_bridge.launcher_probe import _model_id_matches
from vllm_mlx.agent_bridge.model_meta import read_model_metadata
from vllm_mlx.agent_bridge.orchestrator import run_bridge_programmatic
from vllm_mlx.agent_bridge.registry import load_registry, match_rule, resolve_settings
from vllm_mlx.agent_bridge.report import build_report
from vllm_mlx.agent_bridge.smoke import run_tool_call_smoke


def _make_local_model(tmp_path: Path) -> Path:
    model_dir = tmp_path / "demo-model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
                "max_position_embeddings": 65536,
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "chat_template": "{% for msg in messages %}{{ msg['content'] }}{% endfor %}",
                "eos_token": "<|im_end|>",
                "additional_special_tokens": ["<tool_call>", "</tool_call>"],
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "generation_config.json").write_text(
        json.dumps({"max_length": 4096}),
        encoding="utf-8",
    )
    return model_dir


def test_read_model_metadata_from_local_path(tmp_path: Path) -> None:
    model_dir = _make_local_model(tmp_path)
    meta = read_model_metadata(str(model_dir), allow_download=False)
    assert meta["model_dir"] == str(model_dir)
    assert meta["model_type"] == "qwen3"
    assert meta["architecture"] == ["Qwen3ForCausalLM"]
    assert meta["max_context_length"] == 65536
    assert "<tool_call>" in meta["special_tokens"]["additional_special_tokens"]
    assert isinstance(meta["chat_template"], str)


def test_read_model_metadata_prefers_nested_context_over_tokenizer_sentinel(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "nested-context-model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps({"text_config": {"max_position_embeddings": 32768}}),
        encoding="utf-8",
    )
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 10**30}),
        encoding="utf-8",
    )

    meta = read_model_metadata(str(model_dir), allow_download=False)

    assert meta["max_context_length"] == 32768


def test_family_detect_and_registry_resolution(tmp_path: Path) -> None:
    model_dir = _make_local_model(tmp_path)
    meta = read_model_metadata(str(model_dir), allow_download=False)
    family = detect_model_family("Qwen/Qwen3-Coder-32B", meta)
    assert family == "qwen3_coder"
    rules, warnings = load_registry(None)
    assert not warnings
    matched_rule, needs_manual_review = match_rule(
        model_id="Qwen/Qwen3-Coder-32B",
        family=family,
        metadata=meta,
        rules=rules,
    )
    assert matched_rule is not None
    assert needs_manual_review is False
    settings = resolve_settings(
        family=family,
        matched_rule=matched_rule,
        override_tool_call_parser=None,
        override_reasoning_parser=None,
        override_chat_template=None,
    )
    assert settings["tool_call_parser"] == "qwen3_coder"
    assert settings["reasoning_parser"] == "qwen3"


def test_granite_family_uses_dedicated_parser() -> None:
    family = detect_model_family(
        "ibm-granite/granite-4.0-1b",
        {"model_type": "granite", "architecture": ["GraniteForCausalLM"]},
    )
    assert family == "granite"
    rules, warnings = load_registry(None)
    assert not warnings
    matched_rule, needs_manual_review = match_rule(
        model_id="ibm-granite/granite-4.0-1b",
        family=family,
        metadata={},
        rules=rules,
    )
    assert matched_rule is not None
    assert needs_manual_review is False
    settings = resolve_settings(
        family=family,
        matched_rule=matched_rule,
        override_tool_call_parser=None,
        override_reasoning_parser=None,
        override_chat_template=None,
    )
    assert settings["tool_call_parser"] == "granite"


def test_metadata_warnings_do_not_block_ready_report() -> None:
    report = build_report(
        model="demo",
        family="unknown",
        matched_rule_name=None,
        needs_manual_review=False,
        run_mode="run",
        commands={},
        metadata={"warnings": ["metadata was not cached"]},
        health={"ok": True},
        smoke={"ok": True},
        openclaw_written=False,
        hermes_written=False,
        failures=[],
        fixes=[],
        warnings=["metadata was not cached"],
    )
    assert report["agent_ready"] is True
    assert report["warnings"] == ["metadata was not cached"]


def test_model_id_matching_accepts_served_aliases() -> None:
    assert _model_id_matches("owner/model", "owner/model")
    assert _model_id_matches("owner/model", "model")
    assert _model_id_matches("model", "owner/model")
    assert _model_id_matches("owner/model", "/tmp/models/model")
    assert not _model_id_matches("owner/model-a", "owner/model-b")


def test_command_builder_tracks_unsupported_overrides() -> None:
    settings = {
        "tool_call_parser": "hermes",
        "reasoning_parser": "qwen3",
        "chat_template": "foo.jinja",
        "enable_auto_tool_choice": True,
    }
    output = build_commands(
        model="NousResearch/Hermes-3-Llama-3.1-8B",
        host="127.0.0.1",
        port=8000,
        settings=settings,
        python_executable="python3.10",
        primary_module="vllm_mlx.cli",
        gpu_args=["--max-num-seqs 8"],
        context_length=65536,
        served_model_name="hermes-local",
        max_model_len=65536,
    )
    assert "--enable-auto-tool-choice" in output["mlx_primary_command"]
    assert "--tool-call-parser" in output["mlx_primary_command"]
    assert "--chat-template" in output["upstream_reference_command"]
    assert "--served-model-name" in output["upstream_reference_command"]
    assert "--max-model-len" in output["upstream_reference_command"]
    assert any(
        "mlx primary command does not expose --chat-template" in item
        for item in output["unsupported_overrides"]
    )


def test_strict_smoke_rejects_literal_markup(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyResponse:
        status_code = 200
        ok = True
        text = ""

        @staticmethod
        def json() -> dict:
            return {
                "choices": [
                    {
                        "message": {
                            "content": '<tool_call>{"name":"exec"}</tool_call>',
                            "tool_calls": [],
                        }
                    }
                ]
            }

    def fake_post(*args, **kwargs):
        return DummyResponse()

    monkeypatch.setattr("vllm_mlx.agent_bridge.smoke.requests.post", fake_post)
    result = run_tool_call_smoke(
        base_url="http://127.0.0.1:8000",
        model="demo",
        timeout_seconds=3.0,
    )
    assert result["ok"] is False
    assert "literal <tool_call>" in result["reason"]


def test_strict_smoke_retries_when_required_choice_is_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyResponse:
        def __init__(self, status_code: int, text: str, body: dict | None = None):
            self.status_code = status_code
            self.ok = 200 <= status_code < 300
            self.text = text
            self._body = body or {}

        def json(self) -> dict:
            return self._body

    requests_seen: list[dict] = []

    def fake_post(*args, **kwargs):
        requests_seen.append(kwargs["json"])
        if len(requests_seen) == 1:
            return DummyResponse(
                400,
                '{"error":"tool_choice required is unsupported"}',
            )
        return DummyResponse(
            200,
            "",
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "tw_smoke_tool",
                                        "arguments": '{"value":"ok"}',
                                    }
                                }
                            ],
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr("vllm_mlx.agent_bridge.smoke.requests.post", fake_post)
    result = run_tool_call_smoke(
        base_url="http://127.0.0.1:8000",
        model="demo",
        timeout_seconds=3.0,
    )
    assert result["ok"] is True
    assert result["recovery_attempted"] is True
    assert result["tool_choice_fallback"] == "auto"
    assert requests_seen[0]["tool_choice"] == "required"
    assert requests_seen[1]["tool_choice"] == "auto"


def test_strict_smoke_accepts_legacy_function_call_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyResponse:
        status_code = 200
        ok = True
        text = ""

        @staticmethod
        def json() -> dict:
            return {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "function_call": {
                                "name": "tw_smoke_tool",
                                "arguments": '{"value":"ok"}',
                            },
                        }
                    }
                ]
            }

    monkeypatch.setattr(
        "vllm_mlx.agent_bridge.smoke.requests.post", lambda *a, **k: DummyResponse()
    )
    result = run_tool_call_smoke(
        base_url="http://127.0.0.1:8000",
        model="legacy-function-call-model",
        timeout_seconds=3.0,
    )

    assert result["ok"] is True
    assert result["tool_calling"] == "passed"
    assert result["tool_call_format"] == "function_call"


def test_configure_smoke_accepts_healthy_text_only_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyResponse:
        status_code = 200
        ok = True
        text = ""

        @staticmethod
        def json() -> dict:
            return {
                "choices": [
                    {
                        "message": {
                            "content": "ok",
                            "tool_calls": [],
                        }
                    }
                ]
            }

    monkeypatch.setattr(
        "vllm_mlx.agent_bridge.smoke.requests.post", lambda *a, **k: DummyResponse()
    )
    result = run_tool_call_smoke(
        base_url="http://127.0.0.1:8000",
        model="plain-text-model",
        timeout_seconds=3.0,
        allow_text_fallback=True,
    )
    assert result["ok"] is True
    assert result["tool_calling"] == "degraded"
    assert result["degraded"] is True
    assert "plain text" in result["reason"]


def test_configure_smoke_recovers_literal_markup_with_plain_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyResponse:
        def __init__(self, content: str):
            self.status_code = 200
            self.ok = True
            self.text = ""
            self._content = content

        def json(self) -> dict:
            return {
                "choices": [
                    {
                        "message": {
                            "content": self._content,
                            "tool_calls": [],
                        }
                    }
                ]
            }

    responses = iter(
        [
            DummyResponse('<tool_call>{"name":"read"}</tool_call>'),
            DummyResponse("ok"),
        ]
    )
    monkeypatch.setattr(
        "vllm_mlx.agent_bridge.smoke.requests.post",
        lambda *a, **k: next(responses),
    )
    result = run_tool_call_smoke(
        base_url="http://127.0.0.1:8000",
        model="markup-model",
        timeout_seconds=3.0,
        allow_text_fallback=True,
    )
    assert result["ok"] is True
    assert result["tool_calling"] == "degraded"
    assert "literal tool-call markup" in result["reason"]


def test_configure_smoke_recovers_malformed_tool_arguments_with_plain_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyResponse:
        def __init__(self, content: str, arguments: str | None = None):
            self.status_code = 200
            self.ok = True
            self.text = ""
            self._content = content
            self._arguments = arguments

        def json(self) -> dict:
            message = {"content": self._content}
            if self._arguments is not None:
                message["tool_calls"] = [
                    {
                        "function": {
                            "name": "tw_smoke_tool",
                            "arguments": self._arguments,
                        }
                    }
                ]
            else:
                message["tool_calls"] = []
            return {"choices": [{"message": message}]}

    responses = iter(
        [
            DummyResponse("", arguments="not-json"),
            DummyResponse("ok"),
        ]
    )
    monkeypatch.setattr(
        "vllm_mlx.agent_bridge.smoke.requests.post",
        lambda *a, **k: next(responses),
    )
    result = run_tool_call_smoke(
        base_url="http://127.0.0.1:8000",
        model="malformed-args-model",
        timeout_seconds=3.0,
        allow_text_fallback=True,
    )

    assert result["ok"] is True
    assert result["tool_calling"] == "degraded"
    assert "invalid JSON tool arguments" in result["reason"]


def test_config_writer_backup_and_write(tmp_path: Path) -> None:
    openclaw_cfg = tmp_path / "openclaw.json"
    openclaw_cfg.parent.mkdir(parents=True, exist_ok=True)
    openclaw_cfg.write_text('{"old": true}\n', encoding="utf-8")

    hermes_home = tmp_path / "hermes-home"
    hermes_env = hermes_home / ".env"
    hermes_cfg = hermes_home / "config.yaml"
    hermes_home.mkdir(parents=True, exist_ok=True)
    hermes_env.write_text("OLD=1\n", encoding="utf-8")
    hermes_cfg.write_text("model: old\n", encoding="utf-8")

    paths = BridgePaths(
        root_dir=tmp_path,
        openclaw_config_path=openclaw_cfg,
        openclaw_workspace_dir=tmp_path / "workspace",
        hermes_home_dir=hermes_home,
        hermes_env_path=hermes_env,
        hermes_config_path=hermes_cfg,
    )
    openclaw_result = write_openclaw_config(
        paths=paths,
        model_id="demo-model",
        base_url_v1="http://127.0.0.1:8000/v1",
        context_window=8192,
        max_tokens=512,
        api_key="token-workshed-local",
        dry_run=False,
    )
    hermes_result = write_hermes_config(
        paths=paths,
        model_id="demo-model",
        base_url_v1="http://127.0.0.1:8000/v1",
        api_key="token-workshed-local",
        dry_run=False,
    )

    assert openclaw_result.written is True
    assert openclaw_result.backup_path is not None
    assert Path(openclaw_result.path).exists()
    assert hermes_result["env"].written is True
    assert hermes_result["config"].written is True
    assert hermes_result["env"].backup_path is not None
    assert hermes_result["config"].backup_path is not None
    assert detect_api_mode("https://api.openai.com/v1") == "codex_responses"


def test_programmatic_bridge_dry_run_returns_report(tmp_path: Path) -> None:
    model_dir = _make_local_model(tmp_path)
    report = run_bridge_programmatic(
        model=str(model_dir),
        agent="openclaw",
        run=False,
        dry_run=True,
        allow_download=False,
    )
    assert report["run_mode"] == "dry-run"
    assert report["openclaw_config_written"] is False
    assert report["tool_calling"] == "failed"
    assert report["exit_code"] == 0
