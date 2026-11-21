# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the local, Scratch-like Workshed coordinator."""

from __future__ import annotations

import io
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from token_workshed_nextcore.workshed import WorkshedService, _redact


def _wait_for_run(service: WorkshedService, run: dict[str, Any]) -> dict[str, Any]:
    for _ in range(120):
        current = service.get_run(str(run["id"])) or run
        if current["status"] in {"completed", "failed", "cancelled"}:
            return current
        time.sleep(0.025)
    return service.get_run(str(run["id"])) or run


def test_non_finite_runtime_values_are_safe_for_receipt_digests() -> None:
    assert _redact({"loss": float("nan"), "nested": [float("inf")]}) == {
        "loss": None,
        "nested": [None],
    }


def test_redaction_keeps_token_count_parameters_and_hides_credentials() -> None:
    assert _redact(
        {"max_new_tokens": 12, "tokenizer": "local/tokenizer", "hf_token": "secret"}
    ) == {
        "max_new_tokens": 12,
        "tokenizer": "local/tokenizer",
        "hf_token": "[redacted]",
    }


def _valid_order(service: WorkshedService) -> dict:
    order = service.create_work_order({"name": "tiny"})
    order["blocks"][0]["params"]["model_name"] = "tiny"
    order["blocks"].extend(
        [
            {"id": "model", "type_id": "load_model", "lane": "source", "order": 1, "params": {"model": "local"}},
            {"id": "deliver", "type_id": "register_model", "lane": "deliver", "order": 2, "params": {"test_when_finished": False}},
        ]
    )
    order["edges"] = [
        {
            "from": {"block_id": "model", "port": "model", "type": "Model"},
            "to": {"block_id": "deliver", "port": "model", "type": "Model"},
        }
    ]
    return service.update_work_order(order["id"], order, 1)


def test_root_delivery_and_typed_edges_are_enforced(tmp_path: Path) -> None:
    service = WorkshedService(root=tmp_path)
    order = service.create_work_order()
    assert order["blocks"][0]["locked"] is True
    blocked = service.preflight({"work_order": order})
    assert blocked["ok"] is False
    assert any(error["code"] == "delivery_required" for error in blocked["errors"])

    valid = _valid_order(service)
    valid["edges"] = [
        {
            "from": {"block_id": "model", "port": "model", "type": "Model"},
            "to": {"block_id": "deliver", "port": "model", "type": "Model"},
        }
    ]
    assert service.preflight({"work_order": valid})["ok"] is True
    valid["edges"][0]["to"]["type"] = "Dataset"
    invalid = service.preflight({"work_order": valid})
    assert any(error["code"] == "port_type" for error in invalid["errors"])


def test_duplicate_remaps_edge_references(tmp_path: Path) -> None:
    service = WorkshedService(root=tmp_path)
    order = _valid_order(service)
    order["edges"] = [
        {
            "from": {"block_id": "model", "port": "model", "type": "Model"},
            "to": {"block_id": "deliver", "port": "model", "type": "Model"},
        }
    ]
    service.update_work_order(order["id"], order, 2)
    duplicate = service.action_work_order(order["id"], "duplicate")
    ids = {block["id"] for block in duplicate["blocks"]}
    assert all(edge["from"]["block_id"] in ids and edge["to"]["block_id"] in ids for edge in duplicate["edges"])


def test_injected_runner_persists_managed_artifact_and_recovers(tmp_path: Path) -> None:
    def runner(step: dict, workspace: Path, cancel, emit):
        del cancel, emit
        output = workspace / "model"
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text("{}", encoding="utf-8")
        return {
            "id": "workshed/test-artifact",
            "kind": "model",
            "path": str(output),
            "sha256": "test",
            "bytes": 2,
            "source_step": step["id"],
        }

    service = WorkshedService(root=tmp_path, runner=runner)
    order = _valid_order(service)
    preflight = service.preflight({"work_order": order})
    run = service.start({
        "preflight_id": preflight["preflight_id"],
        "accepted_consent_ids": preflight["required_consents"],
    })
    for _ in range(40):
        run = service.get_run(run["id"]) or run
        if run["status"] in {"completed", "failed"}:
            break
        time.sleep(0.025)
    assert run["status"] == "completed"
    assert service.artifacts()[0]["run_id"] == run["id"]
    assert service.managed_model_ids() == ["workshed/test-artifact"]


def test_unknown_toolchain_digest_is_rejected(tmp_path: Path) -> None:
    service = WorkshedService(root=tmp_path)
    with pytest.raises(ValueError, match="locked Workshed catalog"):
        service.prepare_toolchains(["../escape"])


def test_preflight_is_invalidated_by_saved_order_change(tmp_path: Path) -> None:
    service = WorkshedService(root=tmp_path)
    order = _valid_order(service)
    preflight = service.preflight({"work_order": order})
    order["name"] = "changed after preflight"
    service.update_work_order(order["id"], order, order["revision"])
    with pytest.raises(ValueError, match="changed after preflight"):
        service.start({"preflight_id": preflight["preflight_id"], "accepted_consent_ids": []})


def test_lane_constraints_remain_typed(tmp_path: Path) -> None:
    service = WorkshedService(root=tmp_path)
    order = _valid_order(service)
    order["blocks"][1]["lane"] = "verify"
    with pytest.raises(ValueError, match="cannot be placed"):
        service.update_work_order(order["id"], order, order["revision"])


def test_catalog_defaults_are_strict_and_dynamic_options_are_declared(tmp_path: Path) -> None:
    service = WorkshedService(root=tmp_path)
    capabilities = service.capabilities()
    definitions = {
        item["type_id"]: item
        for item in [capabilities["root_block"], *capabilities["blocks"]]
    }
    for definition in definitions.values():
        schema = definition["param_schema"]
        assert schema["additionalProperties"] is False
        assert set(schema["properties"]) == set(schema["required"])
        assert set(schema["properties"]) == set(definition["default_params"])
        Draft202012Validator(schema).validate(definition["default_params"])

    expected = {
        ("load_model", "ref"): "models.hub",
        ("load_dataset", "ref"): "datasets.hub",
        ("evaluate_lm", "tasks"): "lm_eval.tasks",
    }
    for (type_id, field), provider in expected.items():
        definition = definitions[type_id]
        field_schema = definition["param_schema"]["properties"][field]
        assert field_schema["x-ui"]["options_provider"] == provider
        assert definition["ui_schema"]["options_providers"][field] == provider

    retired = {
        "train_sft", "train_lora", "train_qlora", "train_dora", "train_full",
        "train_dpo", "fuse_adapter", "quantize_mlx", "response_distill",
        "logits_distill",
    }
    assert retired.isdisjoint(definitions)

def test_v1_work_order_migrates_aliases_bindings_origins_and_keeps_backup(tmp_path: Path) -> None:
    root_id = "root_legacy"
    legacy_order = {
        "schema_version": 1,
        "id": "legacy-order",
        "revision": 7,
        "name": "legacy",
        "root_block_id": root_id,
        "blocks": [
            {
                "id": root_id,
                "type_id": "model_development",
                "lane": "source",
                "order": 0,
                "locked": True,
                "params": {"model_name": "legacy"},
            },
            {
                "id": "model",
                "type_id": "load_model",
                "lane": "source",
                "order": 1,
                "params": {"model": "org/base", "trust_remote_code": False},
            },
            {
                "id": "train",
                "type_id": "train_lora",
                "lane": "improve",
                "order": 2,
                "params": {
                    "model": "/legacy/model",
                    "dataset": "/legacy/data.jsonl",
                    "rank": "16",
                    "max_steps": "12",
                    "removed_knob": "review me",
                },
            },
        ],
        "edges": [],
    }
    raw = {"version": 1, "work_orders": {"legacy-order": legacy_order}}
    (tmp_path / "work-orders.json").write_text(json.dumps(raw), encoding="utf-8")

    service = WorkshedService(root=tmp_path)
    migrated = service.get_work_order("legacy-order")
    assert migrated is not None
    assert migrated["schema_version"] == 2
    assert migrated["migration"]["from_schema_version"] == 1
    model = next(block for block in migrated["blocks"] if block["id"] == "model")
    assert model["params"]["ref"] == "org/base"
    assert model["param_origins"]["ref"]["kind"] == "migration"
    train = next(block for block in migrated["blocks"] if block["id"] == "train")
    assert train["params"]["lora_rank"] == 16
    assert train["params"]["max_steps"] == 12
    assert train["input_bindings"]["model"]["value"] == "/legacy/model"
    assert train["input_bindings"]["dataset"]["value"] == "/legacy/data.jsonl"
    assert train["legacy_params"] == {"removed_knob": "review me"}
    assert train["needs_review"] is True
    assert (tmp_path / "work-orders.v1.backup.json").is_file()
    assert json.loads((tmp_path / "work-orders.v1.backup.json").read_text(encoding="utf-8")) == raw


def test_saved_user_model_reference_keeps_its_origin(tmp_path: Path) -> None:
    service = WorkshedService(root=tmp_path)
    order = _valid_order(service)
    model = next(block for block in order["blocks"] if block["type_id"] == "load_model")
    model["params"].pop("model", None)
    model["params"]["ref"] = "local-user-model"
    model["param_origins"].pop("ref", None)
    saved = service.update_work_order(order["id"], order, order["revision"])
    model = next(block for block in saved["blocks"] if block["type_id"] == "load_model")
    assert model["params"]["ref"] == "local-user-model"
    assert model["param_origins"]["ref"]["kind"] == "user"

def test_legacy_training_work_order_is_viewable_but_cannot_run(tmp_path: Path) -> None:
    calls: list[str] = []
    service = WorkshedService(
        root=tmp_path,
        runner=lambda step, *_args: calls.append(step["type_id"]) or {},
    )
    order = service.create_work_order({"name": "archived training order"})
    definition = service._definitions()["train_lora"]
    order["blocks"].append(
        {
            "id": "legacy-train",
            "type_id": "train_lora",
            "lane": "improve",
            "order": 1,
            "params": deepcopy(definition["default_params"]),
            "input_bindings": {},
        }
    )
    saved = service.update_work_order(order["id"], order, order["revision"])

    viewed = service.get_work_order(order["id"])
    assert viewed is not None
    assert any(block["type_id"] == "train_lora" for block in viewed["blocks"])
    preflight = service.preflight({"work_order": saved})
    assert preflight["ok"] is False
    assert any(issue["code"] == "removed_operation" for issue in preflight["errors"])
    assert service.list_runs() == []
    assert calls == []

def test_retired_quantization_is_not_a_runnable_capability(tmp_path: Path) -> None:
    service = WorkshedService(root=tmp_path)
    capabilities = service.capabilities()
    assert "quantize_mlx" not in {block["type_id"] for block in capabilities["blocks"]}
    assert "quantize_mlx" in service._definitions()

def test_preflight_rejects_retired_quantization_without_starting_runner(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    service = WorkshedService(
        root=tmp_path,
        runner=lambda step, *_args: calls.append(step["type_id"]) or {},
    )
    order = service.create_work_order({"name": "legacy quantization"})
    definition = service._definitions()["quantize_mlx"]
    order["blocks"].append(
        {
            "id": "legacy-quantize",
            "type_id": "quantize_mlx",
            "lane": "verify",
            "order": 1,
            "params": deepcopy(definition["default_params"]),
            "input_bindings": {},
        }
    )
    saved = service.update_work_order(order["id"], order, order["revision"])

    preflight = service.preflight({"work_order": saved})
    assert preflight["ok"] is False
    assert any(issue["code"] == "removed_operation" for issue in preflight["errors"])
    assert service.list_runs() == []
    assert calls == []

def test_retired_quantization_work_order_remains_available_for_review(
    tmp_path: Path,
) -> None:
    service = WorkshedService(root=tmp_path)
    order = service.create_work_order({"name": "review old quantization"})
    definition = service._definitions()["quantize_mlx"]
    order["blocks"].append(
        {
            "id": "legacy-quantize",
            "type_id": "quantize_mlx",
            "lane": "verify",
            "order": 1,
            "params": deepcopy(definition["default_params"]),
            "input_bindings": {},
        }
    )
    saved = service.update_work_order(order["id"], order, order["revision"])

    viewed = service.get_work_order(order["id"])
    assert viewed is not None
    assert any(block["type_id"] == "quantize_mlx" for block in viewed["blocks"])
    preflight = service.preflight({"work_order": saved})
    assert preflight["ok"] is False
    assert any(issue["code"] == "removed_operation" for issue in preflight["errors"])

def test_workbench_capabilities_identify_the_pinned_sglang_runner(
    tmp_path: Path,
) -> None:
    service = WorkshedService(root=tmp_path)
    capabilities = service.capabilities()
    assert capabilities["runtime"]["inference_engine"] == "SGLang"
    assert capabilities["runtime"]["sglang_version"] == "0.5.22"
    assert capabilities["runners"][0]["id"] == "manager"
    assert capabilities["runners"][0]["label"] == "SGLang Runtime 0.5.22"

def test_register_model_smoke_uses_app_serving_runtime(tmp_path: Path, monkeypatch) -> None:
    service = WorkshedService(root=tmp_path, python_executable="/app/python")
    model_path = tmp_path / "model"
    model_path.mkdir()
    captured: list[tuple[str, str, Path]] = []

    def serve_smoke(run_id: str, path: str, cwd: Path) -> dict[str, Any]:
        captured.append((run_id, path, cwd))
        return {
            "status": "passed",
            "runner": "token_workshed_nextcore.openai_chat_completions",
            "engine": "sglang",
            "served_model": "model",
            "response_chars": 5,
            "elapsed_seconds": 1.25,
            "scope": "App serving smoke test.",
            "argv": [
                "/app/python",
                "-m",
                "token_workshed_nextcore.cli",
                "serve",
                str(model_path),
            ],
            "endpoint": "http://127.0.0.1:12345",
        }

    monkeypatch.setattr(service, "_run_serving_smoke_test", serve_smoke)
    step = {
        "type_id": "register_model",
        "inputs": {"model": str(model_path)},
        "params": {"model_id": "smoke-tested", "collision_policy": "fail", "test_when_finished": True},
    }

    artifact = service._execute_step("run-id", step, tmp_path / "workspace")

    assert artifact is not None
    assert captured == [("run-id", str(model_path), tmp_path / "workspace")]
    assert artifact["smoke_test"]["runner"] == "token_workshed_nextcore.openai_chat_completions"
    assert artifact["runner_projection"]["endpoint"] == "http://127.0.0.1:12345"


def test_serving_smoke_loads_model_and_issues_chat_completion(tmp_path: Path, monkeypatch) -> None:
    service = WorkshedService(root=tmp_path, python_executable="/app/python")
    model_path = tmp_path / "model"
    model_path.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    started: dict[str, Any] = {}

    class FakeProcess:
        pid = 12345
        stdout = io.StringIO("server starting\n")

        def __init__(self) -> None:
            self.returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            self.returncode = 0
            return 0

    process = FakeProcess()

    def fake_popen(command: list[str], **kwargs: Any) -> FakeProcess:
        started["command"] = command
        started["kwargs"] = kwargs
        return process

    class FakeResponse(io.BytesIO):
        pass

    request_timeouts: list[float] = []

    def fake_urlopen(request: Any, timeout: float) -> FakeResponse:
        request_timeouts.append(timeout)
        if request.full_url.endswith("/v1/models"):
            payload = {"data": [{"id": str(model_path)}]}
        else:
            payload = {"choices": [{"message": {"content": "ready"}}]}
        return FakeResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr("token_workshed_nextcore.workshed.subprocess.Popen", fake_popen)
    monkeypatch.setattr("token_workshed_nextcore.workshed.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("token_workshed_nextcore.workshed.os.killpg", lambda *_args: None)

    result = service._run_serving_smoke_test("run-id", str(model_path), workspace)

    assert result["status"] == "passed"
    assert result["runner"] == "token_workshed_nextcore.openai_chat_completions"
    assert result["response_chars"] == len("ready")
    assert request_timeouts == [3, 120]
    assert "-m" in started["command"]
    assert "token_workshed_nextcore.sglang_runtime" in started["command"]
    assert started["kwargs"]["start_new_session"] is True


def test_start_rejects_tampered_frozen_run_plan(tmp_path: Path) -> None:
    service = WorkshedService(root=tmp_path, runner=lambda *_args: None)
    order = _valid_order(service)
    preflight = service.preflight({"work_order": order})
    service._preflights[preflight["preflight_id"]]["run_plan"]["steps"][0]["resolved_params"]["ref"] = "changed"
    with pytest.raises(ValueError, match="run plan digest|Effective parameters"):
        service.start({"preflight_id": preflight["preflight_id"], "accepted_consent_ids": []})
