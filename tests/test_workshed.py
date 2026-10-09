# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the local, Scratch-like Workshed coordinator."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from vllm_mlx.workshed import WorkshedService


def _wait_for_run(service: WorkshedService, run: dict[str, Any]) -> dict[str, Any]:
    for _ in range(120):
        current = service.get_run(str(run["id"])) or run
        if current["status"] in {"completed", "failed", "cancelled"}:
            return current
        time.sleep(0.025)
    return service.get_run(str(run["id"])) or run


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
        ("quantize_mlx", "preset_id"): "quantization.presets",
        ("evaluate_lm", "tasks"): "lm_eval.tasks",
        ("train_lora", "target_modules"): "model.target_modules",
    }
    for (type_id, field), provider in expected.items():
        definition = definitions[type_id]
        field_schema = definition["param_schema"]["properties"][field]
        assert field_schema["x-ui"]["options_provider"] == provider
        assert definition["ui_schema"]["options_providers"][field] == provider
    assert definitions["quantize_mlx"]["param_schema"]["properties"]["preset_id"]["x-ui"]["options_context"] == {"engine": "mlx_local"}


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


def test_user_override_beats_preset_even_when_client_returns_stale_origin(tmp_path: Path) -> None:
    service = WorkshedService(root=tmp_path, runner=lambda *_args: None)
    order = service.create_work_order({"name": "override"})
    order["blocks"][0]["params"]["model_name"] = "override"
    quant_definition = next(
        item for item in service.capabilities()["blocks"] if item["type_id"] == "quantize_mlx"
    )
    params = deepcopy(quant_definition["default_params"])
    params["bits"] = 6
    order["blocks"].extend(
        [
            {
                "id": "quant",
                "type_id": "quantize_mlx",
                "lane": "verify",
                "order": 1,
                "params": params,
                "param_origins": {
                    key: {"kind": "catalog_default", "definition_version": 2}
                    for key in params
                },
                "input_bindings": {"model": {"kind": "literal", "value": "org/base"}},
            },
            {"id": "deliver", "type_id": "register_model", "lane": "deliver", "order": 2, "params": {}},
        ]
    )
    order["edges"] = [
        {
            "from": {"block_id": "quant", "port": "model", "type": "Model"},
            "to": {"block_id": "deliver", "port": "model", "type": "Model"},
        }
    ]
    saved = service.update_work_order(order["id"], order, order["revision"])
    quant = next(block for block in saved["blocks"] if block["id"] == "quant")
    assert quant["param_origins"]["bits"]["kind"] == "user"
    preflight = service.preflight({"work_order": saved})
    step = next(item for item in preflight["steps"] if item["type_id"] == "quantize_mlx")
    assert step["resolved_params"]["bits"] == 6
    assert step["resolved_params"]["group_size"] == 128
    assert step["param_origins"]["bits"]["kind"] == "user"
    assert step["param_origins"]["group_size"] == {"kind": "preset", "preset_id": "mlx-balanced"}


def test_parameter_to_runner_to_artifact_receipt_golden_path(tmp_path: Path) -> None:
    executed: list[dict[str, Any]] = []

    def runner(step: dict[str, Any], workspace: Path, cancel: Any, emit: Any) -> dict[str, Any]:
        del cancel, emit
        executed.append(deepcopy(step))
        output = workspace / str(step["id"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(json.dumps({"type": step["type_id"]}), encoding="utf-8")
        kind = {
            "load_dataset": "dataset",
            "train_lora": "adapter",
        }.get(str(step["type_id"]), "model")
        return {
            "id": f"workshed/{step['id']}",
            "kind": kind,
            "path": str(output),
            "sha256": "runner-placeholder",
            "bytes": 1,
        }

    service = WorkshedService(root=tmp_path, runner=runner)
    order = service.create_work_order({"name": "golden"})
    order["blocks"][0]["params"].update({"model_name": "golden-model", "seed": 123})
    order["blocks"].extend(
        [
            {
                "id": "model",
                "type_id": "load_model",
                "lane": "source",
                "order": 1,
                "params": {
                    "source_kind": "huggingface",
                    "ref": "org/base-model",
                    "revision": "0123456789abcdef",
                    "trust_remote_code": False,
                    "dtype": "bfloat16",
                },
            },
            {
                "id": "data",
                "type_id": "load_dataset",
                "lane": "source",
                "order": 2,
                "params": {"source_kind": "huggingface", "ref": "org/instructions", "split": "train"},
            },
            {
                "id": "train",
                "type_id": "train_lora",
                "lane": "improve",
                "order": 3,
                "params": {
                    "preset_id": "custom",
                    "max_steps": 7,
                    "learning_rate": 0.0002,
                    "lora_rank": 12,
                    "lora_alpha": 24.0,
                    "target_modules": ["q_proj", "v_proj"],
                    "output_name": "golden-adapter",
                },
            },
            {"id": "fuse", "type_id": "fuse_adapter", "lane": "improve", "order": 4, "params": {"output_name": "golden-fused"}},
            {
                "id": "quant",
                "type_id": "quantize_mlx",
                "lane": "verify",
                "order": 5,
                "params": {"preset_id": "custom", "bits": 3, "group_size": 64, "mode": "affine", "output_name": "golden-3bit"},
            },
            {"id": "deliver", "type_id": "register_model", "lane": "deliver", "order": 6, "params": {"model_id": "golden-final", "collision_policy": "fail"}},
        ]
    )
    order["edges"] = [
        {"from": {"block_id": "model", "port": "model", "type": "Model"}, "to": {"block_id": "train", "port": "model", "type": "Model"}},
        {"from": {"block_id": "data", "port": "dataset", "type": "Dataset"}, "to": {"block_id": "train", "port": "dataset", "type": "Dataset"}},
        {"from": {"block_id": "model", "port": "model", "type": "Model"}, "to": {"block_id": "fuse", "port": "model", "type": "Model"}},
        {"from": {"block_id": "train", "port": "adapter", "type": "Adapter"}, "to": {"block_id": "fuse", "port": "adapter", "type": "Adapter"}},
        {"from": {"block_id": "fuse", "port": "model", "type": "Model"}, "to": {"block_id": "quant", "port": "model", "type": "Model"}},
        {"from": {"block_id": "quant", "port": "model", "type": "Model"}, "to": {"block_id": "deliver", "port": "model", "type": "Model"}},
    ]
    saved = service.update_work_order(order["id"], order, order["revision"])
    preflight = service.preflight({"work_order": saved})
    assert preflight["ok"] is True
    assert preflight["schema_version"] == 2
    assert preflight["run_plan"]["catalog_digest"] == preflight["catalog_digest"]
    assert preflight["run_plan"]["plan_digest"]
    train_plan = next(step for step in preflight["run_plan"]["steps"] if step["type_id"] == "train_lora")
    assert train_plan["resolved_params"]["max_steps"] == 7
    assert train_plan["resolved_params"]["seed"] == 123
    assert train_plan["effective_params_digest"]
    assert train_plan["execution_contract_digest"]
    assert train_plan["runner"] == {"id": "mlx_local", "version": "2"}
    assert train_plan["toolchains"][0]["python_executable"].endswith("/venv/bin/python")

    run = service.start(
        {
            "preflight_id": preflight["preflight_id"],
            "accepted_consent_ids": preflight["required_consents"],
        }
    )
    run = _wait_for_run(service, run)
    assert run["status"] == "completed", run.get("error")
    assert [step["type_id"] for step in executed] == [
        "load_model", "load_dataset", "train_lora", "fuse_adapter", "quantize_mlx", "register_model"
    ]
    train_execution = next(step for step in executed if step["type_id"] == "train_lora")
    assert train_execution["params"]["max_steps"] == 7
    assert train_execution["params"]["target_modules"] == ["q_proj", "v_proj"]
    assert train_execution["params"]["seed"] == 123
    assert set(train_execution["inputs"]) == {"model", "dataset"}
    assert all(str(value).startswith(str(tmp_path / "artifacts")) for value in train_execution["inputs"].values())
    quant_execution = next(step for step in executed if step["type_id"] == "quantize_mlx")
    assert quant_execution["params"]["bits"] == 3
    assert quant_execution["params"]["group_size"] == 64
    assert quant_execution["input_provenance"]["model"]["artifact_id"] == "workshed/step_fuse"

    final_artifact = next(item for item in service.artifacts() if item["id"] == "workshed/step_deliver")
    provenance = final_artifact["provenance"]
    assert final_artifact["registered"] is True
    assert final_artifact["parent_artifact_ids"] == ["workshed/step_quant"]
    assert provenance["effective_params"] == {
        "model_id": "golden-final", "collision_policy": "fail", "test_when_finished": True,
    }
    assert provenance["param_origins"]["model_id"]["kind"] == "user"
    assert provenance["inputs"]["model"]["artifact_id"] == "workshed/step_quant"
    assert final_artifact["receipt_digest"]

    receipt = service.run_receipt(str(run["id"]))
    assert receipt["schema_version"] == 2
    assert receipt["finalized"] is True
    assert receipt["status"] == "completed"
    assert receipt["plan_digest"] == preflight["run_plan"]["plan_digest"]
    assert receipt["receipt_digest"]
    receipt_train = next(step for step in receipt["steps"] if step["type_id"] == "train_lora")
    assert receipt_train["effective_params"]["max_steps"] == 7
    assert receipt_train["input_provenance"]["model"]["artifact_id"] == "workshed/step_model"
    assert service.run_receipt(str(run["id"])) == receipt


def test_builtin_quantize_uses_frozen_toolchain_python_and_explicit_argv(tmp_path: Path) -> None:
    service = WorkshedService(root=tmp_path, python_executable="/app/python-must-not-run")
    order = service.create_work_order({"name": "argv"})
    order["blocks"][0]["params"]["model_name"] = "argv"
    order["blocks"].extend(
        [
            {
                "id": "quant",
                "type_id": "quantize_mlx",
                "lane": "verify",
                "order": 1,
                "params": {"preset_id": "custom", "bits": 6, "group_size": 32, "mode": "affine", "output_name": "explicit"},
                "input_bindings": {"model": {"kind": "literal", "value": "org/base"}},
            },
            {"id": "deliver", "type_id": "register_model", "lane": "deliver", "order": 2, "params": {}},
        ]
    )
    order["edges"] = [
        {"from": {"block_id": "quant", "port": "model", "type": "Model"}, "to": {"block_id": "deliver", "port": "model", "type": "Model"}}
    ]
    saved = service.update_work_order(order["id"], order, order["revision"])
    preflight = service.preflight({"work_order": saved})
    step = deepcopy(next(item for item in preflight["run_plan"]["steps"] if item["type_id"] == "quantize_mlx"))
    interpreter = Path(step["toolchains"][0]["python_executable"])
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    interpreter.write_text("synthetic", encoding="utf-8")
    (interpreter.parents[2] / ".ready").write_text("ready", encoding="utf-8")
    captured: list[str] = []

    def fake_command(_run_id: str, command: list[str], _cwd: Path) -> None:
        captured.extend(command)
        output = Path(command[command.index("--mlx-path") + 1])
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text("{}", encoding="utf-8")

    service._run_command = fake_command  # type: ignore[method-assign]
    step["params"] = deepcopy(step["resolved_params"])
    step["inputs"] = {"model": "org/base"}
    artifact = service._execute_step("run", step, tmp_path / "workspace")
    assert artifact is not None
    assert captured[0] == str(interpreter)
    assert captured[0] != service.python_executable
    assert captured[captured.index("--q-bits") + 1] == "6"
    assert captured[captured.index("--q-group-size") + 1] == "32"
    assert captured[captured.index("--q-mode") + 1] == "affine"


def test_start_rejects_tampered_frozen_run_plan(tmp_path: Path) -> None:
    service = WorkshedService(root=tmp_path, runner=lambda *_args: None)
    order = _valid_order(service)
    preflight = service.preflight({"work_order": order})
    service._preflights[preflight["preflight_id"]]["run_plan"]["steps"][0]["resolved_params"]["ref"] = "changed"
    with pytest.raises(ValueError, match="run plan digest|Effective parameters"):
        service.start({"preflight_id": preflight["preflight_id"], "accepted_consent_ids": []})
