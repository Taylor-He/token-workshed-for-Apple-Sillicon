# SPDX-License-Identifier: Apache-2.0
"""Dependency-light coverage for Workshed dataset and quality result flows."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from token_workshed_nextcore.workshed import WorkshedQualityGateFailure, WorkshedService
from token_workshed_nextcore.workshed_execution import (
    _json_safe,
    _render_messages,
    prepare_dataset,
    write_json,
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_file_path_worker_does_not_shadow_stdlib_platform() -> None:
    runner = Path(__file__).resolve().parents[1] / "token_workshed_nextcore" / "workshed_execution.py"
    code = (
        "import importlib.util, sys\n"
        "sys.modules.pop('platform', None)\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "spec = importlib.util.spec_from_file_location('workshed_worker_probe', sys.argv[2])\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "import platform\n"
        "assert callable(platform.system)\n"
        "assert not platform.__file__.startswith(sys.argv[1])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(runner.parent), str(runner)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_chat_format_falls_back_when_tokenizer_has_no_chat_template() -> None:
    class TokenizerWithoutTemplate:
        chat_template = None

        @staticmethod
        def apply_chat_template(*_args: Any, **_kwargs: Any) -> str:
            raise AssertionError("an unconfigured chat template must not be called")

    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    assert _render_messages(TokenizerWithoutTemplate(), messages) == (
        "user: question\nassistant: answer"
    )


def test_json_writer_normalizes_non_finite_metrics(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    write_json(path, {"loss": float("nan"), "perplexity": float("inf")})
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "loss": None,
        "perplexity": None,
    }
    assert _json_safe({"loss": float("-inf")}) == {"loss": None}


def test_dataset_prepare_maps_completion_fields_and_creates_deterministic_split(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [{"question": f"q{i}", "answer": f"a{i}"} for i in range(4)])
    output = tmp_path / "normalized"

    result = prepare_dataset(
        {
            "params": {
                "source_kind": "local",
                "ref": str(source),
                "format": "auto",
                "record_format": "completions",
                "prompt_field": "question",
                "response_field": "answer",
                "validation_fraction": 0.25,
                "shuffle": False,
                "seed": 7,
            },
            "output_dir": str(output),
        }
    )

    manifest = result["manifest"]
    assert manifest["counts"] == {"source": 4, "train": 3, "valid": 1}
    train = [
        json.loads(line)
        for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    valid = json.loads((output / "valid.jsonl").read_text(encoding="utf-8").strip())
    assert train[0]["prompt"] == "q1"
    assert train[0]["completion"] == "a1"
    assert valid["messages"][-1] == {"role": "assistant", "content": "a0"}


def test_dataset_prepare_normalizes_chat_and_preference_records(tmp_path: Path) -> None:
    chat_path = tmp_path / "chat.jsonl"
    _write_jsonl(
        chat_path,
        [
            {
                "turns": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            }
        ],
    )
    chat = prepare_dataset(
        {
            "params": {
                "source_kind": "local",
                "ref": str(chat_path),
                "record_format": "chat",
                "messages_field": "turns",
                "validation_fraction": 0,
                "shuffle": False,
            },
            "output_dir": str(tmp_path / "chat-out"),
        }
    )
    chat_row = json.loads(
        (Path(chat["path"]) / "train.jsonl").read_text(encoding="utf-8").strip()
    )
    assert chat_row["prompt"] == "user: hi"
    assert chat_row["completion"] == "hello"

    preference_path = tmp_path / "preferences.jsonl"
    _write_jsonl(
        preference_path, [{"instruction": "choose", "good": "yes", "bad": "no"}]
    )
    preference = prepare_dataset(
        {
            "params": {
                "source_kind": "local",
                "ref": str(preference_path),
                "record_format": "preference",
                "prompt_field": "instruction",
                "chosen_field": "good",
                "rejected_field": "bad",
                "validation_fraction": 0,
                "shuffle": False,
            },
            "output_dir": str(tmp_path / "preference-out"),
        }
    )
    preference_row = json.loads(
        (Path(preference["path"]) / "train.jsonl").read_text(encoding="utf-8").strip()
    )
    assert preference_row == {"prompt": "choose", "chosen": "yes", "rejected": "no"}


def test_quality_gate_records_pass_missing_skip_and_failure(tmp_path: Path) -> None:
    service = WorkshedService(root=tmp_path)
    step = {
        "id": "gate",
        "type_id": "quality_gate",
        "params": {
            "metric_name": "eval.loss",
            "operator": "lte",
            "threshold": 0.8,
            "missing": "fail",
        },
        "inputs": {"metric": {"eval": {"loss": 0.5}}},
        "output_ports": [{"name": "passed", "type": "Boolean"}],
    }
    passed = service._execute_step("run", deepcopy(step), tmp_path / "workspace")
    assert passed is not None
    assert passed["step_result"]["status"] == "passed"
    assert passed["outputs"]["passed"] is True

    missing = deepcopy(step)
    missing["inputs"]["metric"] = {"accuracy": 1.0}
    missing["params"]["missing"] = "skip"
    skipped = service._execute_step("run", missing, tmp_path / "workspace")
    assert skipped is not None
    assert skipped["step_result"]["status"] == "skipped"
    assert skipped["outputs"]["passed"] is None

    failing = deepcopy(step)
    failing["inputs"]["metric"] = {"eval": {"loss": 1.2}}
    with pytest.raises(
        WorkshedQualityGateFailure, match="Quality gate failed"
    ) as error:
        service._execute_step("run", failing, tmp_path / "workspace")
    assert error.value.result["value"] == 1.2


def test_quality_gate_failure_marks_remaining_steps_skipped_in_receipt(
    tmp_path: Path,
) -> None:
    service: WorkshedService

    def runner(step: dict[str, Any], workspace: Path, cancel, emit):
        del workspace, cancel, emit
        if step["type_id"] == "quality_gate":
            raise WorkshedQualityGateFailure(
                "gate rejected metric",
                {"status": "failed", "metric_name": "loss", "value": 2.0},
            )
        return None

    service = WorkshedService(root=tmp_path, runner=runner)
    steps = [
        {
            "id": "metric",
            "type_id": "evaluate_loss",
            "status": "pending",
            "output_ports": [],
        },
        {
            "id": "gate",
            "type_id": "quality_gate",
            "status": "pending",
            "output_ports": [],
        },
        {
            "id": "deliver",
            "type_id": "register_model",
            "status": "pending",
            "output_ports": [],
        },
    ]
    plan = {"steps": deepcopy(steps), "toolchains": []}
    run_id = "quality-fail"
    service._runs[run_id] = {
        "id": run_id,
        "status": "queued",
        "steps": deepcopy(steps),
        "events": [],
        "event_cursor": 0,
        "artifacts": [],
        "plan": plan,
    }
    service._active_run_id = run_id
    service._validate_run_plan = lambda _plan: None  # type: ignore[method-assign]
    service._resolve_step_inputs = lambda step, _outputs: (step, {}, [])  # type: ignore[method-assign]

    service._run(run_id, plan)
    run = service.get_run(run_id)
    assert run is not None
    assert run["status"] == "failed"
    assert [step["status"] for step in run["steps"]] == [
        "succeeded",
        "failed",
        "skipped",
    ]
    assert (
        run["steps"][2]["skip_reason"] == "Skipped because quality gate 'gate' failed."
    )
    receipt = service.run_receipt(run_id)
    assert receipt["finalized"] is True
    assert receipt["steps"][1]["result"]["value"] == 2.0
    assert receipt["steps"][2]["status"] == "skipped"


def test_response_distillation_maps_teacher_column_into_chat_messages(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text(
        json.dumps({"format": "completions", "counts": {"train": 1}}),
        encoding="utf-8",
    )
    _write_jsonl(
        dataset / "train.jsonl",
        [{"prompt": "Say hello", "teacher_text": "Hello there."}],
    )

    prepared = WorkshedService(root=tmp_path / "state")._prepare_response_dataset(
        str(dataset), "teacher_text", tmp_path / "workspace"
    )
    row = json.loads((prepared / "train.jsonl").read_text(encoding="utf-8").strip())
    assert row["completion"] == "Hello there."
    assert row["messages"] == [
        {"role": "user", "content": "Say hello"},
        {"role": "assistant", "content": "Hello there."},
    ]
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["response_distillation"] == {
        "response_field": "teacher_text",
        "records": 1,
    }


def test_dataset_mapping_fields_remain_and_training_blocks_are_retired(
    tmp_path: Path,
) -> None:
    service = WorkshedService(root=tmp_path)
    capabilities = service.capabilities()
    blocks = {block["type_id"]: block for block in capabilities["blocks"]}
    dataset = blocks["load_dataset"]
    assert {
        "record_format",
        "messages_field",
        "text_field",
        "chosen_field",
        "rejected_field",
    } <= set(dataset["param_schema"]["properties"])
    assert dataset["runner"]["id"] == "manager"
    assert dataset["runner"]["toolchain_profiles"] == ["dataset-prepare"]
    assert "train_dpo" not in blocks
    assert "train_dpo" in service._definitions()


def test_legacy_training_work_order_fails_preflight_before_execution(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def runner(step: dict[str, Any], *_args: Any) -> dict[str, Any]:
        calls.append(str(step.get("type_id", "")))
        return {}

    service = WorkshedService(root=tmp_path, runner=runner)
    order = service.create_work_order({"name": "legacy training"})
    order["blocks"][0]["params"]["model_name"] = "legacy-training"
    definition = service._definitions()["train_sft"]
    order["blocks"].append(
        {
            "id": "legacy-train",
            "type_id": "train_sft",
            "lane": definition["allowed_lanes"][0],
            "order": 1,
            "params": definition["default_params"],
            "param_origins": {},
            "input_bindings": {},
        }
    )

    result = service.preflight({"work_order": order})
    assert result["ok"] is False
    assert any(error["code"] == "removed_operation" for error in result["errors"])
    assert service.list_runs() == []
    assert calls == []

    with pytest.raises(RuntimeError, match="removed in SGLang-only NextCore"):
        service._execute_step("guard-check", {"type_id": "train_sft"}, tmp_path)
    assert calls == []
