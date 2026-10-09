# SPDX-License-Identifier: Apache-2.0
"""Local, block-oriented model development work orders.

The Workshed service intentionally sits beside the existing quantization
service.  Quantization was the first specialised job type; Workshed is the
versioned compiler and coordinator for a complete local model-development
plan.  The manager is the only process the native UI talks to.

The first implementation keeps the wire contract useful even when optional
MLX/HF training dependencies are absent: capability probing and preflight are
dependency-light, while execution reports an actionable missing-tool error
instead of pretending that a block succeeded.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")
_TERMINAL_RUNS = {"completed", "failed", "cancelled", "interrupted"}
_ACTIVE_RUNS = {"preparing", "queued", "running", "pausing", "paused", "cancelling"}
_BLOCK_LIMIT = 250
_MAX_NESTING = 8
_REPEAT_LIMIT = 100
_RETRY_LIMIT = 5
_WORK_ORDER_SCHEMA_VERSION = 2
_BLOCK_DEFINITION_VERSION = 2
_SCHEMA_URI = "https://json-schema.org/draft/2020-12/schema"
_BINDING_KINDS = {"literal", "auto", "inherited"}
_PARAM_ORIGINS = {
    "user",
    "catalog_default",
    "preset",
    "inherited",
    "edge",
    "source_resolution",
    "derived",
    "migration",
    "secret_ref",
}
_RUNNER_VERSIONS = {"mlx_local": "2", "hf_mps": "2", "manager": "2"}
_RESOURCE_PORT_TYPES = {"Model", "Adapter", "Dataset", "Report", "Artifact", "Metric"}
_FINAL_RECEIPT_STATUSES = {"completed", "cancelled"}
_SMOKE_TEST_TIMEOUT_SECONDS = 120
_SMOKE_TEST_RESULT_PREFIX = "WORKSHED_SMOKE_RESULT="
# This program is application-owned. Work-order values are passed separately
# through argv and cannot change Python source or package names.
_SMOKE_TEST_PROGRAM = """
import json, sys, time
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
started = time.monotonic()
mx.random.seed(42)
model, tokenizer = load(sys.argv[1], tokenizer_config={"trust_remote_code": False})
prompt = "Introduce yourself in one short sentence."
if getattr(tokenizer, "chat_template", None):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
answer = generate(model, tokenizer, prompt=prompt, max_tokens=24,
                  sampler=make_sampler(temp=0.0), verbose=False)
if not answer.strip():
    raise RuntimeError("The model loaded but did not produce a text response.")
print("WORKSHED_SMOKE_RESULT=" + json.dumps({
    "status": "passed", "engine": "mlx-lm", "seed": 42,
    "generation_limit": 24, "generated_text": answer[:1000],
    "elapsed_seconds": round(time.monotonic() - started, 3),
    "scope": "Model loading and short generation only; not a quality evaluation."
}))
"""

# A small, reviewable lock manifest. User-entered block parameters never flow
# into these package specs. Deployments may update this table as a release,
# which changes the manifest digest and causes a fresh consent prompt.
_TOOLCHAIN_LOCK: dict[str, tuple[str, ...]] = {
    "mlx-train": ("mlx-lm==0.31.1", "datasets==4.8.3"),
    "hf-posttrain": (
        "transformers==5.3.0",
        "datasets==4.8.3",
        "accelerate==1.12.0",
        "trl==0.26.1",
        "peft==0.18.1",
    ),
    "evaluation": ("lm-eval==0.4.9", "evaluate==0.4.6"),
}


def _iso(value: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(value or time.time()))


def _slug(value: str, fallback: str = "new-model") -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip()).strip(".-_")
    return (result[:120] or fallback).lower()


def _redact(value: Any) -> Any:
    """Keep secrets out of work-order state, events, and exception text."""

    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(part in lowered for part in ("token", "secret", "password", "authorization")):
                result[str(key)] = "[redacted]"
            else:
                result[str(key)] = _redact(item)
        return result
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return re.sub(r"(?i)bearer\s+[A-Za-z0-9._~+/-]+", "Bearer [redacted]", value)[:4000]
    return value


@dataclass(frozen=True)
class WorkOrderError:
    code: str
    message: str
    block_id: str = ""

    def as_dict(self) -> dict[str, str]:
        value = {"code": self.code, "message": self.message}
        if self.block_id:
            value["block_id"] = self.block_id
        return value


class WorkshedService:
    """Persistent local work-order compiler and job coordinator."""

    def __init__(
        self,
        *,
        root: str | Path | None = None,
        python_executable: str | None = None,
        runner: Callable[[dict[str, Any], Path, threading.Event, Callable[..., None]], None] | None = None,
    ) -> None:
        configured = str(os.environ.get("TOKEN_WORKSHED_WORKSHED_ROOT", "") or "").strip()
        default = (
            Path(configured).expanduser()
            if configured
            else Path.home() / "Library" / "Application Support" / "token-workshed" / "workshed"
        )
        self.root = Path(root or default).expanduser().resolve()
        self.work_orders_path = self.root / "work-orders.json"
        self.runs_path = self.root / "runs.json"
        self.artifacts_path = self.root / "artifacts.json"
        self.toolchains_root = self.root / "toolchains"
        self.staging_root = self.root / "staging"
        self.python_executable = str(python_executable or sys.executable)
        self._runner = runner
        self._lock = threading.RLock()
        self._work_orders: dict[str, dict[str, Any]] = {}
        self._runs: dict[str, dict[str, Any]] = {}
        self._artifacts: dict[str, dict[str, Any]] = {}
        self._preflights: dict[str, dict[str, Any]] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._active_run_id = ""
        self._queued_runs: deque[str] = deque()
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _read_json(self, path: Path, default: Any) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return default

    def _write_json_atomic(self, path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(_redact(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temporary, path)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary)

    def _load(self) -> None:
        with self._lock:
            raw_orders = self._read_json(self.work_orders_path, {})
            raw_runs = self._read_json(self.runs_path, {})
            raw_artifacts = self._read_json(self.artifacts_path, {})
            migrated_orders = False
            if isinstance(raw_orders, dict):
                source_orders = raw_orders.get("work_orders", raw_orders)
                if isinstance(source_orders, dict):
                    for key, value in source_orders.items():
                        if not isinstance(value, dict):
                            continue
                        upgraded, migrated = self._migrate_work_order(value)
                        self._work_orders[str(key)] = upgraded
                        migrated_orders = migrated_orders or migrated
            if migrated_orders and self.work_orders_path.is_file():
                backup = self.work_orders_path.with_name("work-orders.v1.backup.json")
                if not backup.exists():
                    self._write_json_atomic(backup, raw_orders)
            if isinstance(raw_runs, dict):
                self._runs = {
                    str(key): deepcopy(value)
                    for key, value in raw_runs.get("runs", raw_runs).items()
                    if isinstance(value, dict)
                }
            if isinstance(raw_artifacts, dict):
                self._artifacts = {
                    str(key): deepcopy(value)
                    for key, value in raw_artifacts.get("artifacts", raw_artifacts).items()
                    if isinstance(value, dict)
                }
            for run in self._runs.values():
                if run.get("status") in _ACTIVE_RUNS:
                    run["status"] = "interrupted"
                    run["stage"] = "recovered"
                    run["message"] = "Manager restarted before the local step completed."
                    run["updated_at"] = _iso()
            self._persist_locked()

    def _persist_locked(self) -> None:
        self._write_json_atomic(self.work_orders_path, {"version": 2, "work_orders": self._work_orders})
        self._write_json_atomic(self.runs_path, {"version": 2, "runs": self._runs})
        self._write_json_atomic(self.artifacts_path, {"version": 2, "artifacts": self._artifacts})

    # ------------------------------------------------------------------
    # Capability catalog
    # ------------------------------------------------------------------
    def capabilities(self) -> dict[str, Any]:
        """Return the versioned compiler catalog consumed by every client.

        Defaults, presets, validation and inspector metadata deliberately live
        in the same BlockDefinitionV2 documents.  Native clients may render a
        friendlier editor, but may not invent defaults that the compiler does
        not know about.
        """

        blocks = self._block_definitions()
        return {
            "ok": True,
            "schema_version": _WORK_ORDER_SCHEMA_VERSION,
            "catalog_digest": self._catalog_digest(),
            "runtime": {
                "platform": sys.platform,
                "python": sys.version.split()[0],
                "local_only": True,
                "concurrency": 1,
            },
            "runners": [
                {
                    "id": "mlx_local",
                    "version": _RUNNER_VERSIONS["mlx_local"],
                    "label": "MLX-LM · This Mac",
                    "kind": "local",
                    "status": "ready" if sys.platform == "darwin" else "limited",
                    "blocks": [item["type_id"] for item in blocks if "mlx_local" in item.get("runners", [])],
                },
                {
                    "id": "hf_mps",
                    "version": _RUNNER_VERSIONS["hf_mps"],
                    "label": "Transformers · MPS",
                    "kind": "local",
                    "status": "available",
                    "blocks": [item["type_id"] for item in blocks if "hf_mps" in item.get("runners", [])],
                    "notes": ["MPS fallback to CPU is reported during preflight and never selected silently."],
                },
            ],
            "toolchains": self._toolchain_catalog(),
            "root_block": self._root_definition(),
            "blocks": blocks,
        }

    @staticmethod
    def _object_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
        return {
            "$schema": _SCHEMA_URI,
            "type": "object",
            "additionalProperties": False,
            "properties": deepcopy(properties),
            "required": list(required if required is not None else properties),
        }

    @staticmethod
    def _ui_schema(defaults: dict[str, Any], *, advanced: list[str] | None = None) -> dict[str, Any]:
        advanced_fields = list(advanced or [])
        basic = [name for name in defaults if name not in advanced_fields]
        return {
            "order": list(defaults),
            "groups": [
                {"id": "basic", "label": "Setup", "fields": basic},
                {"id": "advanced", "label": "Advanced", "fields": advanced_fields, "collapsed": True},
            ],
            "advanced": advanced_fields,
            "widgets": {},
        }

    @staticmethod
    def _literal_schema(port_type: str) -> dict[str, Any]:
        if port_type in {"Model", "Dataset", "Adapter", "Metric", "Report", "Artifact"}:
            return {"type": "string", "minLength": 1, "maxLength": 2048}
        if port_type == "CandidateSet" or port_type == "List":
            return {"type": "array", "minItems": 1, "maxItems": 100, "items": {}}
        if port_type in {"Number", "Integer"}:
            return {"type": "number" if port_type == "Number" else "integer"}
        if port_type == "Boolean":
            return {"type": "boolean"}
        return {}

    def _port(self, name: str, port_type: str, *, required: bool = True) -> dict[str, Any]:
        return {
            "name": name,
            "type": port_type,
            "required": required,
            "binding_modes": ["edge", "literal", "auto", "inherited"],
            "literal_schema": self._literal_schema(port_type),
        }

    def _root_definition(self) -> dict[str, Any]:
        defaults = {"model_name": "", "description": "", "seed": 42, "tags": []}
        properties = {
            "model_name": {"type": "string", "maxLength": 120},
            "description": {"type": "string", "maxLength": 2000},
            "seed": {"type": "integer", "minimum": 0, "maximum": 4294967295},
            "tags": {"type": "array", "maxItems": 32, "uniqueItems": True, "items": {"type": "string", "minLength": 1, "maxLength": 40}},
        }
        return {
            "schema_version": 2,
            "definition_version": _BLOCK_DEFINITION_VERSION,
            "type_id": "model_development",
            "category": "project",
            "label": "model development",
            "shape": "root",
            "inputs": [],
            "outputs": [],
            "runners": ["manager"],
            "allowed_lanes": ["source"],
            "param_schema": self._object_schema(properties),
            "ui_schema": self._ui_schema(defaults, advanced=["description", "seed", "tags"]),
            "default_params": defaults,
            "presets": [],
            "runner": {"id": "manager", "version": _RUNNER_VERSIONS["manager"], "toolchain_profiles": []},
        }

    @staticmethod
    def _toolchain_catalog() -> list[dict[str, Any]]:
        labels = {"mlx-train": "MLX training", "hf-posttrain": "HF post-training", "evaluation": "Evaluation"}
        return [
            {
                "id": profile,
                "label": labels[profile],
                "packages": [package.split("==", 1)[0] for package in packages],
                "locked_packages": list(packages),
            }
            for profile, packages in _TOOLCHAIN_LOCK.items()
        ]

    def _catalog_digest(self) -> str:
        blocks = deepcopy(self._block_definitions())
        for block in blocks:
            block.pop("available", None)
            block.pop("unavailable_reason", None)
        payload = {
            "schema_version": _WORK_ORDER_SCHEMA_VERSION,
            "definition_version": _BLOCK_DEFINITION_VERSION,
            "root": self._root_definition(),
            "blocks": blocks,
            "toolchains": {key: list(value) for key, value in _TOOLCHAIN_LOCK.items()},
            "runner_versions": _RUNNER_VERSIONS,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _builtin_supported(self, type_id: str) -> bool:
        return type_id in {
            "load_model", "load_dataset", "train_sft", "train_lora", "train_qlora", "train_dora",
            "train_full", "fuse_adapter", "quantize_mlx", "register_model",
        }

    def _block_definitions(self) -> list[dict[str, Any]]:
        string_or_null = {"type": ["string", "null"], "maxLength": 120}
        output_name = {"type": ["string", "null"], "pattern": r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$"}
        seed = {"type": ["integer", "null"], "minimum": 0, "maximum": 4294967295}
        common_train_defaults = {
            "preset_id": "balanced", "batch_size": 1, "gradient_accumulation_steps": 4,
            "max_sequence_length": 2048, "max_steps": 1000, "learning_rate": 1e-5,
            "optimizer": "adamw", "weight_decay": 0.01, "checkpoint_steps": 100,
            "eval_steps": 100, "num_layers": 16, "seed": None, "output_name": None,
        }
        common_train_properties = {
            "preset_id": {"type": "string", "enum": ["quick", "balanced", "quality", "custom"]},
            "batch_size": {"type": "integer", "minimum": 1, "maximum": 128},
            "gradient_accumulation_steps": {"type": "integer", "minimum": 1, "maximum": 1024},
            "max_sequence_length": {"type": "integer", "minimum": 128, "maximum": 131072},
            "max_steps": {"type": "integer", "minimum": 1, "maximum": 10000000},
            "learning_rate": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
            "optimizer": {"type": "string", "enum": ["adamw", "adam", "sgd", "adafactor"]},
            "weight_decay": {"type": "number", "minimum": 0, "maximum": 1},
            "checkpoint_steps": {"type": "integer", "minimum": 1, "maximum": 1000000},
            "eval_steps": {"type": "integer", "minimum": 1, "maximum": 1000000},
            "num_layers": {"type": "integer", "minimum": -1, "maximum": 1024},
            "seed": seed,
            "output_name": output_name,
        }
        train_presets = [
            {"id": "quick", "label": "Quick", "description": "Small smoke run", "params": {"max_steps": 100, "checkpoint_steps": 50, "eval_steps": 50}},
            {"id": "balanced", "label": "Balanced", "description": "Recommended starting point", "params": {"max_steps": 1000, "checkpoint_steps": 100, "eval_steps": 100}},
            {"id": "quality", "label": "Quality", "description": "Longer, lower-rate run", "params": {"max_steps": 3000, "learning_rate": 5e-6, "checkpoint_steps": 250, "eval_steps": 250}},
            {"id": "custom", "label": "Custom", "description": "Use every advanced override", "params": {}},
        ]

        def action(
            type_id: str,
            category: str,
            label: str,
            inputs: list[tuple[str, str] | tuple[str, str, bool]],
            outputs: list[tuple[str, str]],
            defaults: dict[str, Any],
            properties: dict[str, Any],
            *,
            runners: list[str] | None = None,
            lanes: list[str] | None = None,
            presets: list[dict[str, Any]] | None = None,
            advanced: list[str] | None = None,
            shape: str = "action",
        ) -> dict[str, Any]:
            input_ports = [
                self._port(value[0], value[1], required=value[2] if len(value) == 3 else True)
                for value in inputs
            ]
            runner_ids = list(runners or ["mlx_local"])
            profiles = self._toolchains_for_block(type_id)
            available = self._runner is not None or self._builtin_supported(type_id)
            ui = self._ui_schema(defaults, advanced=advanced)
            if "preset_id" in defaults:
                ui["widgets"]["preset_id"] = "preset-cards"
            option_providers: dict[str, str] = {}
            option_context: dict[str, Any] = {}
            for field_name, field_schema in properties.items():
                if not isinstance(field_schema, dict):
                    continue
                x_ui = field_schema.get("x-ui") if isinstance(field_schema.get("x-ui"), dict) else {}
                provider = str(x_ui.get("options_provider", "")).strip()
                if provider:
                    option_providers[field_name] = provider
                if isinstance(x_ui.get("options_context"), dict):
                    option_context[field_name] = deepcopy(x_ui["options_context"])
            if option_providers:
                # Kept in both locations intentionally: param_schema is the
                # field-level source of truth, while ui_schema lets a client
                # discover providers without walking the whole JSON Schema.
                ui["options_providers"] = option_providers
            if option_context:
                ui["options_context"] = option_context
            return {
                "schema_version": 2,
                "definition_version": _BLOCK_DEFINITION_VERSION,
                "type_id": type_id,
                "category": category,
                "label": label,
                "shape": shape,
                "inputs": input_ports,
                "outputs": [{"name": name, "type": port_type} for name, port_type in outputs],
                "runners": runner_ids,
                "allowed_lanes": list(lanes or ["source", "improve", "verify", "deliver"]),
                "param_schema": self._object_schema(properties),
                "ui_schema": ui,
                "default_params": deepcopy(defaults),
                "presets": deepcopy(presets or []),
                "runner": {"id": runner_ids[0], "version": _RUNNER_VERSIONS[runner_ids[0]], "toolchain_profiles": profiles},
                "available": available,
                "unavailable_reason": "This runner adapter is not implemented in the installed manager." if not available else "",
            }

        lora_defaults = {
            "lora_rank": 8,
            "lora_alpha": 16.0,
            "lora_dropout": 0.0,
            "target_modules": [],
        }
        lora_properties = {
            "lora_rank": {"type": "integer", "minimum": 1, "maximum": 1024},
            "lora_alpha": {"type": "number", "exclusiveMinimum": 0, "maximum": 4096},
            "lora_dropout": {"type": "number", "minimum": 0, "exclusiveMaximum": 1},
            "target_modules": {
                "type": "array",
                "maxItems": 256,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1, "maxLength": 200},
                "x-ui": {
                    "options_provider": "model.target_modules",
                    "options_context": {"model_input": "model"},
                },
            },
        }
        train_advanced = [
            "batch_size", "gradient_accumulation_steps", "max_sequence_length", "max_steps",
            "learning_rate", "optimizer", "weight_decay", "checkpoint_steps", "eval_steps",
            "num_layers", "seed", "output_name",
        ]
        adapter_train_advanced = [
            *train_advanced,
            "lora_rank", "lora_alpha", "lora_dropout", "target_modules",
        ]

        definitions = [
            action(
                "load_model", "models", "load base model", [], [("model", "Model")],
                {"source_kind": "huggingface", "ref": "", "revision": "", "trust_remote_code": False, "dtype": "auto"},
                {
                    "source_kind": {"type": "string", "enum": ["huggingface", "local", "managed"]},
                    "ref": {
                        "type": "string",
                        "maxLength": 2048,
                        "x-ui": {
                            "options_provider": "models.hub",
                            "options_provider_by": {
                                "field": "source_kind",
                                "huggingface": "models.hub",
                                "local": "models.local",
                                "managed": "models.managed",
                            },
                        },
                    },
                    "revision": {"type": "string", "maxLength": 200},
                    "trust_remote_code": {"type": "boolean"},
                    "dtype": {"type": "string", "enum": ["auto", "float16", "bfloat16", "float32"]},
                }, lanes=["source"], advanced=["revision", "trust_remote_code", "dtype"],
            ),
            action(
                "load_dataset", "data", "prepare dataset", [], [("dataset", "Dataset")],
                {
                    "source_kind": "huggingface", "ref": "", "revision": "", "subset": "", "split": "train",
                    "format": "auto", "prompt_field": "prompt", "response_field": "response",
                    "validation_fraction": 0.05, "shuffle": True, "seed": None,
                },
                {
                    "source_kind": {"type": "string", "enum": ["huggingface", "local", "managed"]},
                    "ref": {
                        "type": "string",
                        "maxLength": 2048,
                        "x-ui": {
                            "options_provider": "datasets.hub",
                            "options_provider_by": {"field": "source_kind", "huggingface": "datasets.hub"},
                        },
                    },
                    "revision": {"type": "string", "maxLength": 200},
                    "subset": {"type": "string", "maxLength": 200}, "split": {"type": "string", "minLength": 1, "maxLength": 200},
                    "format": {"type": "string", "enum": ["auto", "json", "jsonl", "parquet", "arrow", "text"]},
                    "prompt_field": {"type": "string", "minLength": 1, "maxLength": 120},
                    "response_field": {"type": "string", "minLength": 1, "maxLength": 120},
                    "validation_fraction": {"type": "number", "minimum": 0, "maximum": 0.5},
                    "shuffle": {"type": "boolean"}, "seed": seed,
                }, lanes=["source"], advanced=["revision", "subset", "split", "format", "prompt_field", "response_field", "validation_fraction", "shuffle", "seed"],
            ),
            action(
                "generate_teacher_responses", "distill", "generate teacher responses",
                [("teacher", "Model"), ("dataset", "Dataset")], [("dataset", "Dataset")],
                {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 512, "batch_size": 1, "response_field": "teacher_response", "seed": None, "on_error": "fail"},
                {
                    "temperature": {"type": "number", "minimum": 0, "maximum": 2}, "top_p": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
                    "max_new_tokens": {"type": "integer", "minimum": 1, "maximum": 32768}, "batch_size": {"type": "integer", "minimum": 1, "maximum": 256},
                    "response_field": {"type": "string", "minLength": 1, "maxLength": 120}, "seed": seed,
                    "on_error": {"type": "string", "enum": ["fail", "skip"]},
                }, lanes=["improve"], advanced=["temperature", "top_p", "max_new_tokens", "batch_size", "response_field", "seed", "on_error"],
            ),
            action(
                "response_distill", "distill", "distill responses", [("student", "Model"), ("dataset", "Dataset")], [("model", "Model")],
                {**common_train_defaults, **lora_defaults, "response_field": "teacher_response"},
                {**common_train_properties, **lora_properties, "response_field": {"type": "string", "minLength": 1, "maxLength": 120}},
                lanes=["improve"], presets=train_presets, advanced=[*adapter_train_advanced, "response_field"],
            ),
            action(
                "logits_distill", "distill", "distill compatible logits", [("teacher", "Model"), ("student", "Model"), ("dataset", "Dataset")], [("model", "Model")],
                {**common_train_defaults, "temperature": 2.0, "distillation_weight": 0.5, "task_weight": 0.5},
                {**common_train_properties, "temperature": {"type": "number", "exclusiveMinimum": 0, "maximum": 20}, "distillation_weight": {"type": "number", "minimum": 0, "maximum": 1}, "task_weight": {"type": "number", "minimum": 0, "maximum": 1}},
                runners=["hf_mps"], lanes=["improve"], presets=train_presets, advanced=[*train_advanced, "temperature", "distillation_weight", "task_weight"],
            ),
            action(
                "train_sft", "train", "train with SFT", [("model", "Model"), ("dataset", "Dataset")], [("model", "Model")],
                {**common_train_defaults, "gradient_checkpointing": False},
                {**common_train_properties, "gradient_checkpointing": {"type": "boolean"}},
                lanes=["improve"], presets=train_presets, advanced=[*train_advanced, "gradient_checkpointing"],
            ),
            action(
                "train_lora", "train", "train with LoRA", [("model", "Model"), ("dataset", "Dataset")], [("adapter", "Adapter")],
                {**common_train_defaults, **lora_defaults}, {**common_train_properties, **lora_properties},
                lanes=["improve"], presets=train_presets, advanced=adapter_train_advanced,
            ),
            action(
                "train_qlora", "train", "train with QLoRA", [("model", "Model"), ("dataset", "Dataset")], [("adapter", "Adapter")],
                {**common_train_defaults, **lora_defaults, "base_bits": 4, "quantization_mode": "affine"},
                {**common_train_properties, **lora_properties, "base_bits": {"type": "integer", "enum": [2, 3, 4, 6, 8]}, "quantization_mode": {"type": "string", "enum": ["affine", "mxfp4"]}},
                lanes=["improve"], presets=train_presets, advanced=[*adapter_train_advanced, "base_bits", "quantization_mode"],
            ),
            action(
                "train_dora", "train", "train with DoRA", [("model", "Model"), ("dataset", "Dataset")], [("adapter", "Adapter")],
                {**common_train_defaults, **lora_defaults}, {**common_train_properties, **lora_properties},
                lanes=["improve"], presets=train_presets, advanced=adapter_train_advanced,
            ),
            action(
                "train_full", "train", "full fine-tune", [("model", "Model"), ("dataset", "Dataset")], [("model", "Model")],
                {**common_train_defaults, "max_steps": 100, "gradient_accumulation_steps": 8, "num_layers": -1, "gradient_checkpointing": True},
                {**common_train_properties, "gradient_checkpointing": {"type": "boolean"}},
                lanes=["improve"], presets=train_presets, advanced=[*train_advanced, "gradient_checkpointing"],
            ),
            action(
                "train_dpo", "train", "train with DPO", [("model", "Model"), ("dataset", "Dataset")], [("model", "Model")],
                {**common_train_defaults, "learning_rate": 5e-7, "max_steps": 500, "gradient_accumulation_steps": 8, "beta": 0.1, "loss_type": "sigmoid", "max_prompt_length": 1024},
                {**common_train_properties, "beta": {"type": "number", "exclusiveMinimum": 0, "maximum": 10}, "loss_type": {"type": "string", "enum": ["sigmoid", "hinge", "ipo"]}, "max_prompt_length": {"type": "integer", "minimum": 64, "maximum": 65536}},
                runners=["hf_mps"], lanes=["improve"], presets=train_presets, advanced=[*train_advanced, "beta", "loss_type", "max_prompt_length"],
            ),
            action(
                "fuse_adapter", "train", "fuse adapter", [("model", "Model"), ("adapter", "Adapter")], [("model", "Model")],
                {"output_name": None, "dequantize": False}, {"output_name": output_name, "dequantize": {"type": "boolean"}}, lanes=["improve"], advanced=["dequantize"],
            ),
            action(
                "quantize_mlx", "quantize", "quantize with MLX", [("model", "Model")], [("model", "Model")],
                {"preset_id": "mlx-balanced", "bits": None, "group_size": None, "mode": None, "output_name": None},
                {
                    "preset_id": {
                        "type": "string",
                        "enum": ["mlx-compressed", "mlx-balanced", "mlx-quality", "custom"],
                        "x-ui": {
                            "options_provider": "quantization.presets",
                            "options_context": {"engine": "mlx_local"},
                        },
                    },
                    "bits": {"type": ["integer", "null"], "enum": [2, 3, 4, 6, 8, None]},
                    "group_size": {"type": ["integer", "null"], "enum": [32, 64, 128, 256, None]},
                    "mode": {"type": ["string", "null"], "enum": ["affine", "mxfp4", None]}, "output_name": output_name,
                }, lanes=["verify"],
                presets=[
                    {"id": "mlx-compressed", "label": "Compressed", "params": {"bits": 3, "group_size": 64, "mode": "affine"}},
                    {"id": "mlx-balanced", "label": "Balanced", "params": {"bits": 4, "group_size": 128, "mode": "affine"}},
                    {"id": "mlx-quality", "label": "Quality", "params": {"bits": 8, "group_size": 128, "mode": "affine"}},
                    {"id": "custom", "label": "Custom", "params": {}},
                ], advanced=["bits", "group_size", "mode", "output_name"],
            ),
            action(
                "evaluate_loss", "test_compare", "evaluate loss", [("model", "Model"), ("dataset", "Dataset")], [("metric", "Metric")],
                {"split": "validation", "max_batches": 25, "batch_size": 1},
                {"split": {"type": "string", "minLength": 1, "maxLength": 120}, "max_batches": {"type": "integer", "minimum": 1, "maximum": 100000}, "batch_size": {"type": "integer", "minimum": 1, "maximum": 256}}, lanes=["verify"], advanced=["max_batches", "batch_size"],
            ),
            action(
                "evaluate_lm", "test_compare", "run reproducible lm-eval", [("model", "Model")], [("report", "Report")],
                {"tasks": ["gsm8k"], "fewshot": 0, "limit": 100, "batch_size": "auto", "seed": None},
                {"tasks": {"type": "array", "minItems": 1, "maxItems": 100, "uniqueItems": True, "items": {"type": "string", "minLength": 1, "maxLength": 120}, "x-ui": {"options_provider": "lm_eval.tasks"}}, "fewshot": {"type": "integer", "minimum": 0, "maximum": 100}, "limit": {"type": ["integer", "null"], "minimum": 1}, "batch_size": {"oneOf": [{"const": "auto"}, {"type": "integer", "minimum": 1, "maximum": 512}]}, "seed": seed}, runners=["hf_mps"], lanes=["verify"], advanced=["fewshot", "limit", "batch_size", "seed"],
            ),
            action(
                "benchmark_model", "test_compare", "benchmark model", [("model", "Model")], [("metric", "Metric")],
                {"warmup_runs": 1, "runs": 5, "max_new_tokens": 128, "concurrency": 1},
                {"warmup_runs": {"type": "integer", "minimum": 0, "maximum": 100}, "runs": {"type": "integer", "minimum": 1, "maximum": 1000}, "max_new_tokens": {"type": "integer", "minimum": 1, "maximum": 32768}, "concurrency": {"type": "integer", "minimum": 1, "maximum": 1024}}, lanes=["verify"], advanced=["warmup_runs", "runs", "max_new_tokens", "concurrency"],
            ),
            action(
                "compare_models", "test_compare", "compare candidates", [("candidates", "CandidateSet")], [("report", "Report")],
                {"metrics": ["quality", "latency", "size"], "quality_weight": 0.6, "latency_weight": 0.25, "size_weight": 0.15},
                {"metrics": {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"type": "string", "enum": ["quality", "latency", "size"]}}, "quality_weight": {"type": "number", "minimum": 0, "maximum": 1}, "latency_weight": {"type": "number", "minimum": 0, "maximum": 1}, "size_weight": {"type": "number", "minimum": 0, "maximum": 1}}, runners=["hf_mps"], lanes=["verify"], advanced=["quality_weight", "latency_weight", "size_weight"],
            ),
            action(
                "quality_gate", "test_compare", "quality gate", [("metric", "Metric")], [("passed", "Boolean")],
                {"metric_name": "loss", "operator": "lte", "threshold": 1.0, "missing": "fail"},
                {"metric_name": {"type": "string", "minLength": 1, "maxLength": 120}, "operator": {"type": "string", "enum": ["lt", "lte", "gt", "gte", "eq"]}, "threshold": {"type": "number"}, "missing": {"type": "string", "enum": ["fail", "skip"]}}, lanes=["verify"], advanced=["missing"],
            ),
            action(
                "register_model", "deliver", "register new model", [("model", "Model")], [("artifact", "Artifact")],
                {"model_id": None, "collision_policy": "fail", "test_when_finished": True},
                {"model_id": string_or_null, "collision_policy": {"type": "string", "enum": ["fail", "replace", "version"]}, "test_when_finished": {"type": "boolean", "description": "Load the saved model and generate a short response before registering it. This is not a quality evaluation."}},
                lanes=["deliver"], advanced=["collision_policy"],
            ),
            action("reveal_artifact", "deliver", "show in Finder", [("artifact", "Artifact")], [("artifact", "Artifact")], {}, {}, lanes=["deliver"]),
            action(
                "export_report", "deliver", "export report", [("report", "Report")], [("artifact", "Artifact")],
                {"format": "json", "filename": None, "collision_policy": "fail"},
                {"format": {"type": "string", "enum": ["json", "markdown"]}, "filename": output_name, "collision_policy": {"type": "string", "enum": ["fail", "replace", "version"]}}, lanes=["deliver"], advanced=["filename", "collision_policy"],
            ),
            action("repeat", "control", "repeat N times", [], [], {"count": 2}, {"count": {"type": "integer", "minimum": 1, "maximum": _REPEAT_LIMIT}}, lanes=["improve", "verify"], shape="c"),
            action("for_each", "control", "for each item", [("items", "List")], [("candidates", "CandidateSet")], {"max_items": 20}, {"max_items": {"type": "integer", "minimum": 1, "maximum": 100}}, lanes=["improve", "verify"], shape="c"),
            action("if_else", "logic", "if / else", [("condition", "Boolean")], [], {"on_false": "skip"}, {"on_false": {"type": "string", "enum": ["skip", "stop"]}}, lanes=["improve", "verify"], shape="c"),
            action("retry", "control", "retry N times", [], [], {"count": 2, "backoff_seconds": 5, "retry_on": "transient"}, {"count": {"type": "integer", "minimum": 0, "maximum": _RETRY_LIMIT}, "backoff_seconds": {"type": "integer", "minimum": 0, "maximum": 3600}, "retry_on": {"type": "string", "enum": ["transient", "any"]}}, lanes=["improve", "verify"], shape="c"),
            action("stop", "logic", "stop work order", [], [], {"reason": ""}, {"reason": {"type": "string", "maxLength": 500}}, lanes=["improve", "verify"], shape="terminal"),
        ]
        return definitions

    def _definitions(self) -> dict[str, dict[str, Any]]:
        return {
            item["type_id"]: item
            for item in [self._root_definition(), *self._block_definitions()]
        }

    @staticmethod
    def _origin(kind: str, **details: Any) -> dict[str, Any]:
        return {"kind": kind, **details}

    @staticmethod
    def _coerce_legacy_value(value: Any, schema: dict[str, Any]) -> Any:
        """Perform the only permissive coercion allowed by the v1 migrator."""

        if not isinstance(value, str):
            return value
        text = value.strip()
        declared = schema.get("type")
        types = set(declared if isinstance(declared, list) else [declared])
        try:
            if "integer" in types and re.fullmatch(r"[-+]?\d+", text):
                return int(text)
            if "number" in types and re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text):
                number = float(text)
                return number if math.isfinite(number) else value
        except (TypeError, ValueError, OverflowError):
            return value
        return value

    def _migrate_work_order(self, source: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Upgrade a v1 document and canonicalize defaults without losing it.

        Known aliases are accepted for one compatibility cycle. Unknown v1
        values are quarantined in ``legacy_params`` and make preflight block
        until a user reviews them; v2 documents remain strict and retain the
        unknown value so validation can point at the exact key.
        """

        order = deepcopy(source)
        try:
            original_version = int(order.get("schema_version", 1))
        except (TypeError, ValueError):
            original_version = 1
        legacy = original_version < _WORK_ORDER_SCHEMA_VERSION
        changed = legacy
        definitions = self._definitions()
        edges = order.get("edges") if isinstance(order.get("edges"), list) else []
        edge_targets = {
            (str((edge.get("to") or {}).get("block_id", "")), str((edge.get("to") or {}).get("port", "")))
            for edge in edges
            if isinstance(edge, dict) and isinstance(edge.get("to"), dict)
        }
        aliases: dict[str, dict[str, str]] = {
            "load_model": {"model": "ref", "source": "ref", "path": "ref"},
            "load_dataset": {"dataset": "ref", "data": "ref", "source": "ref", "path": "ref"},
            "quantize_mlx": {"preset": "preset_id"},
            "train_sft": {"rank": "lora_rank"},
            "train_lora": {"rank": "lora_rank"},
            "train_qlora": {"rank": "lora_rank"},
            "train_dora": {"rank": "lora_rank"},
            "quality_gate": {"metric": "metric_name"},
            "register_model": {"output_name": "model_id"},
        }
        blocks = order.get("blocks") if isinstance(order.get("blocks"), list) else []
        for raw in blocks:
            if not isinstance(raw, dict):
                continue
            block_id = str(raw.get("id", ""))
            type_id = str(raw.get("type_id", ""))
            definition = definitions.get(type_id)
            if definition is None:
                if legacy:
                    raw["disabled"] = True
                    raw["needs_review"] = True
                continue
            params = deepcopy(raw.get("params") if isinstance(raw.get("params"), dict) else {})
            origins = deepcopy(raw.get("param_origins") if isinstance(raw.get("param_origins"), dict) else {})
            alias_map = aliases.get(type_id, {})
            for old_name, new_name in alias_map.items():
                if old_name not in params:
                    continue
                if new_name not in params or params.get(new_name) is None or params.get(new_name) == "":
                    params[new_name] = params[old_name]
                    origins[new_name] = self._origin("migration", alias=old_name)
                params.pop(old_name, None)
                origins.pop(old_name, None)
                changed = True
            bindings = deepcopy(raw.get("input_bindings") if isinstance(raw.get("input_bindings"), dict) else {})
            for port in definition.get("inputs", []):
                port_name = str(port.get("name", ""))
                if port_name not in params:
                    continue
                literal = params.pop(port_name)
                origins.pop(port_name, None)
                if (block_id, port_name) not in edge_targets and port_name not in bindings:
                    bindings[port_name] = {"kind": "literal", "value": literal, "origin": "migration"}
                changed = True
            schema = definition.get("param_schema", {})
            properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
            if legacy:
                unknown = {key: params.pop(key) for key in list(params) if key not in properties}
                if unknown:
                    raw["legacy_params"] = {**deepcopy(raw.get("legacy_params") or {}), **unknown}
                    raw["needs_review"] = True
                    changed = True
            for key, value in list(params.items()):
                if key in properties and legacy:
                    coerced = self._coerce_legacy_value(value, properties[key])
                    if coerced != value or type(coerced) is not type(value):
                        params[key] = coerced
                        origins[key] = self._origin("migration", coercion="numeric_string")
                        changed = True
            defaults = definition.get("default_params", {})
            supplied = set(params)
            for key, value in defaults.items():
                if key not in params:
                    params[key] = deepcopy(value)
                    origins[key] = self._origin("catalog_default", definition_version=_BLOCK_DEFINITION_VERSION)
                    changed = True
                elif key not in origins:
                    origins[key] = self._origin("migration" if legacy else "user")
            for key in supplied:
                if key in params and key not in origins:
                    origins[key] = self._origin("migration" if legacy else "user")
                elif (
                    key in params
                    and key in defaults
                    and isinstance(origins.get(key), dict)
                    and origins[key].get("kind") == "catalog_default"
                    and params[key] != defaults[key]
                ):
                    # A schema-driven client may send the block's original
                    # origin map back unchanged after editing a value.  The
                    # manager can prove that a non-default value is a user
                    # override, preventing preset expansion from silently
                    # replacing it during preflight.
                    origins[key] = self._origin("user")
            raw["params"] = params
            raw["param_origins"] = origins
            raw["input_bindings"] = bindings
            raw["definition_version"] = _BLOCK_DEFINITION_VERSION
        order["blocks"] = blocks
        order["edges"] = edges
        order["schema_version"] = _WORK_ORDER_SCHEMA_VERSION
        if legacy:
            order["migration"] = {
                "from_schema_version": original_version,
                "to_schema_version": _WORK_ORDER_SCHEMA_VERSION,
                "migrated_at": _iso(),
            }
        return order, changed

    # ------------------------------------------------------------------
    # Work-order CRUD
    # ------------------------------------------------------------------
    def _new_work_order(self, name: str = "") -> dict[str, Any]:
        order_id = uuid.uuid4().hex
        root_id = f"root_{order_id[:8]}"
        title = str(name or "").strip() or "New model"
        return {
            "schema_version": _WORK_ORDER_SCHEMA_VERSION,
            "id": order_id,
            "revision": 1,
            "name": title,
            "root_block_id": root_id,
            "blocks": [{
                "id": root_id,
                "type_id": "model_development",
                "lane": "source",
                "order": 0,
                "definition_version": _BLOCK_DEFINITION_VERSION,
                "params": deepcopy(self._root_definition()["default_params"]),
                "param_origins": {
                    key: self._origin("catalog_default", definition_version=_BLOCK_DEFINITION_VERSION)
                    for key in self._root_definition()["default_params"]
                },
                "input_bindings": {},
                "locked": True,
            }],
            "edges": [],
            "created_at": _iso(),
            "updated_at": _iso(),
        }

    def list_work_orders(self) -> list[dict[str, Any]]:
        with self._lock:
            return [deepcopy(item) for item in sorted(self._work_orders.values(), key=lambda x: str(x.get("updated_at", "")), reverse=True)]

    def get_work_order(self, work_order_id: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._work_orders.get(str(work_order_id or ""))
            return deepcopy(value) if value else None

    def create_work_order(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload if isinstance(payload, dict) else {}
        order = deepcopy(payload) if payload.get("blocks") else self._new_work_order(str(payload.get("name", "")))
        order, _ = self._migrate_work_order(order)
        errors = self._validate_shape(order, allow_missing_root=False)
        if errors:
            raise ValueError("; ".join(error.message for error in errors))
        order["id"] = str(order.get("id") or uuid.uuid4().hex)
        order["revision"] = 1
        order["created_at"] = str(order.get("created_at") or _iso())
        order["updated_at"] = _iso()
        with self._lock:
            self._work_orders[order["id"]] = order
            self._persist_locked()
            return deepcopy(order)

    def update_work_order(self, work_order_id: str, payload: dict[str, Any], expected_revision: int) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Work order must be an object.")
        with self._lock:
            current = self._work_orders.get(str(work_order_id or ""))
            if not current:
                raise KeyError("Work order not found.")
            if int(current.get("revision", 0)) != int(expected_revision):
                raise ValueError("Work order revision is stale; reload before saving.")
            next_order = deepcopy(payload)
            next_order["id"] = current["id"]
            next_order["revision"] = int(current.get("revision", 0)) + 1
            next_order["created_at"] = current.get("created_at", _iso())
            next_order["updated_at"] = _iso()
            next_order, _ = self._migrate_work_order(next_order)
            errors = self._validate_shape(next_order, allow_missing_root=False)
            if errors:
                raise ValueError("; ".join(error.message for error in errors))
            self._work_orders[next_order["id"]] = next_order
            self._persist_locked()
            return deepcopy(next_order)

    def action_work_order(self, work_order_id: str, action: str) -> dict[str, Any]:
        action = str(action or "").strip().lower()
        with self._lock:
            current = self._work_orders.get(str(work_order_id or ""))
            if not current:
                raise KeyError("Work order not found.")
            if action == "duplicate":
                duplicate = deepcopy(current)
                duplicate["id"] = uuid.uuid4().hex
                duplicate["name"] = f"{current.get('name', 'Work order')} copy"
                duplicate["revision"] = 1
                duplicate["created_at"] = _iso()
                duplicate["updated_at"] = _iso()
                id_map: dict[str, str] = {}
                for block in duplicate.get("blocks", []):
                    old_id = str(block.get("id", ""))
                    block["id"] = f"{block.get('type_id', 'block')}_{uuid.uuid4().hex[:8]}"
                    id_map[old_id] = block["id"]
                for edge in duplicate.get("edges", []):
                    if not isinstance(edge, dict):
                        continue
                    for endpoint in (edge.get("from"), edge.get("to")):
                        if isinstance(endpoint, dict) and str(endpoint.get("block_id", "")) in id_map:
                            endpoint["block_id"] = id_map[str(endpoint["block_id"])]
                duplicate["root_block_id"] = next((item["id"] for item in duplicate["blocks"] if item.get("locked")), duplicate["root_block_id"])
                self._work_orders[duplicate["id"]] = duplicate
                self._persist_locked()
                return deepcopy(duplicate)
            if action in {"archive", "restore"}:
                current["archived"] = action == "archive"
                current["updated_at"] = _iso()
                self._persist_locked()
                return deepcopy(current)
        raise ValueError("Unsupported work order action.")

    # ------------------------------------------------------------------
    # Validation and compilation
    # ------------------------------------------------------------------
    @staticmethod
    def _contains_non_finite(value: Any) -> bool:
        if isinstance(value, float):
            return not math.isfinite(value)
        if isinstance(value, dict):
            return any(WorkshedService._contains_non_finite(item) for item in value.values())
        if isinstance(value, list):
            return any(WorkshedService._contains_non_finite(item) for item in value)
        return False

    def _validate_shape(self, order: dict[str, Any], *, allow_missing_root: bool) -> list[WorkOrderError]:
        errors: list[WorkOrderError] = []
        if int(order.get("schema_version", 0) or 0) != _WORK_ORDER_SCHEMA_VERSION:
            errors.append(WorkOrderError("schema_version", f"Work orders must use schema version {_WORK_ORDER_SCHEMA_VERSION}."))
        blocks = order.get("blocks")
        if not isinstance(blocks, list):
            raise ValueError("Work order blocks must be a list.")
        if len(blocks) > _BLOCK_LIMIT:
            errors.append(WorkOrderError("block_limit", f"A work order can contain at most {_BLOCK_LIMIT} blocks."))
        definitions = self._definitions()
        block_ids: set[str] = set()
        root_id = str(order.get("root_block_id", ""))
        locked_roots = 0
        blocks_by_id: dict[str, dict[str, Any]] = {}
        for raw in blocks:
            if not isinstance(raw, dict):
                errors.append(WorkOrderError("block_object", "Every block must be an object."))
                continue
            block_id = str(raw.get("id", "")).strip()
            type_id = str(raw.get("type_id", "")).strip()
            if not _SAFE_ID.fullmatch(block_id):
                errors.append(WorkOrderError("block_id", "Block ids must be stable safe identifiers.", block_id))
            if block_id in block_ids:
                errors.append(WorkOrderError("duplicate_block", "Block ids must be unique.", block_id))
            block_ids.add(block_id)
            blocks_by_id[block_id] = raw
            if type_id not in definitions:
                errors.append(WorkOrderError("unknown_block", f"Unknown block type: {type_id}.", block_id))
            if type_id in definitions:
                lane = str(raw.get("lane", "")).strip().lower()
                allowed_lanes = {
                    str(value).strip().lower()
                    for value in definitions[type_id].get("allowed_lanes", [])
                }
                if lane not in allowed_lanes:
                    errors.append(
                        WorkOrderError(
                            "lane",
                            f"Block '{type_id}' cannot be placed in the {lane or 'unknown'} lane.",
                            block_id,
                        )
                    )
            params = raw.get("params")
            if not isinstance(params, dict):
                errors.append(WorkOrderError("params", "Block params must be an object.", block_id))
                params = {}
            for key in params:
                lowered = str(key).strip().lower()
                if lowered in {"command", "shell", "python", "code", "script", "pip", "package_url", "git_url", "index_url"}:
                    errors.append(WorkOrderError("unsafe_param", f"Parameter '{key}' is not allowed in a Workshed block.", block_id))
            if self._contains_non_finite(params):
                errors.append(WorkOrderError("param_finite", "Numeric parameters must be finite.", block_id))
            definition = definitions.get(type_id)
            if definition is not None:
                validator = Draft202012Validator(definition["param_schema"])
                for issue in sorted(validator.iter_errors(params), key=lambda item: list(item.absolute_path)):
                    location = ".".join(str(item) for item in issue.absolute_path)
                    label = f"Parameter '{location}'" if location else "Parameters"
                    errors.append(WorkOrderError("param_schema", f"{label}: {issue.message}", block_id))
            origins = raw.get("param_origins")
            if not isinstance(origins, dict):
                errors.append(WorkOrderError("param_origins", "Block param_origins must be an object.", block_id))
                origins = {}
            for key, origin in origins.items():
                if key not in params:
                    errors.append(WorkOrderError("param_origin", f"Origin for unknown parameter '{key}' is not allowed.", block_id))
                    continue
                kind = str(origin.get("kind", "")) if isinstance(origin, dict) else str(origin)
                if kind not in _PARAM_ORIGINS:
                    errors.append(WorkOrderError("param_origin", f"Parameter '{key}' has an invalid origin.", block_id))
            declared_inputs = {
                str(item.get("name", "")): item
                for item in (definition or {}).get("inputs", [])
            }
            bindings = raw.get("input_bindings")
            if not isinstance(bindings, dict):
                errors.append(WorkOrderError("input_bindings", "Block input_bindings must be an object.", block_id))
                bindings = {}
            for port_name, binding in bindings.items():
                if port_name not in declared_inputs:
                    errors.append(WorkOrderError("binding_port", f"Unknown input binding port '{port_name}'.", block_id))
                    continue
                if not isinstance(binding, dict):
                    errors.append(WorkOrderError("binding", f"Binding '{port_name}' must be an object.", block_id))
                    continue
                kind = str(binding.get("kind", ""))
                if kind not in _BINDING_KINDS:
                    errors.append(WorkOrderError("binding_kind", f"Binding '{port_name}' must be literal, auto, or inherited.", block_id))
                    continue
                extra_binding_keys = set(binding) - {"kind", "value", "selector", "name", "origin"}
                if extra_binding_keys:
                    errors.append(WorkOrderError("binding", f"Binding '{port_name}' has unsupported fields: {', '.join(sorted(extra_binding_keys))}.", block_id))
                if kind == "literal":
                    if "value" not in binding:
                        errors.append(WorkOrderError("binding_literal", f"Literal binding '{port_name}' needs a value.", block_id))
                    else:
                        literal_validator = Draft202012Validator(declared_inputs[port_name].get("literal_schema", {}))
                        for issue in literal_validator.iter_errors(binding.get("value")):
                            errors.append(WorkOrderError("binding_literal", f"Literal binding '{port_name}': {issue.message}", block_id))
                elif kind == "inherited" and not str(binding.get("name", "")).strip():
                    errors.append(WorkOrderError("binding_inherited", f"Inherited binding '{port_name}' needs a name.", block_id))
            output_name = str(params.get("output_name", "")).strip()
            if output_name and not _SAFE_NAME.fullmatch(output_name):
                errors.append(WorkOrderError("output_name", "Output names must be safe local identifiers.", block_id))
            if raw.get("needs_review"):
                errors.append(WorkOrderError("migration_review", "Review or remove legacy parameters before running this block.", block_id))
            if raw.get("locked"):
                locked_roots += 1
                if block_id != root_id or type_id != "model_development":
                    errors.append(WorkOrderError("root_block", "The model development root block cannot be replaced.", block_id))
            if type_id == "compare_models" and all(isinstance(params.get(key), (int, float)) for key in ("quality_weight", "latency_weight", "size_weight")):
                total_weight = sum(float(params[key]) for key in ("quality_weight", "latency_weight", "size_weight"))
                if not math.isclose(total_weight, 1.0, abs_tol=1e-9):
                    errors.append(WorkOrderError("weight_sum", "Comparison weights must add up to 1.0.", block_id))
            if (
                type_id == "logits_distill"
                and all(isinstance(params.get(key), (int, float)) for key in ("distillation_weight", "task_weight"))
                and not math.isclose(
                    float(params["distillation_weight"]) + float(params["task_weight"]),
                    1.0,
                    abs_tol=1e-9,
                )
            ):
                errors.append(WorkOrderError("weight_sum", "Distillation and task weights must add up to 1.0.", block_id))
        if not allow_missing_root and (locked_roots != 1 or root_id not in block_ids):
            errors.append(WorkOrderError("root_block", "A work order must have one locked model development root block."))
        edges = order.get("edges", [])
        if not isinstance(edges, list):
            errors.append(WorkOrderError("edges", "Work order edges must be a list."))
            edges = []
        adjacency: dict[str, list[str]] = {block_id: [] for block_id in block_ids}
        ports_by_block = {
            str(raw.get("id", "")): definitions.get(str(raw.get("type_id", "")), {})
            for raw in blocks
            if isinstance(raw, dict)
        }
        target_bindings: set[tuple[str, str]] = set()
        for edge in edges:
            if not isinstance(edge, dict):
                errors.append(WorkOrderError("edge_object", "Every edge must be an object."))
                continue
            source = edge.get("from") if isinstance(edge.get("from"), dict) else {}
            target = edge.get("to") if isinstance(edge.get("to"), dict) else {}
            source_id = str(source.get("block_id", ""))
            target_id = str(target.get("block_id", ""))
            if source_id not in block_ids or target_id not in block_ids:
                errors.append(WorkOrderError("edge_reference", "Edges must reference existing blocks."))
                continue
            source_port = str(source.get("port", ""))
            target_port = str(target.get("port", ""))
            target_key = (target_id, target_port)
            if target_key in target_bindings:
                errors.append(WorkOrderError("duplicate_binding", f"Input '{target_port}' may have only one edge.", target_id))
            target_bindings.add(target_key)
            target_block = blocks_by_id.get(target_id, {})
            if target_port in (target_block.get("input_bindings") if isinstance(target_block.get("input_bindings"), dict) else {}):
                errors.append(WorkOrderError("duplicate_binding", f"Input '{target_port}' cannot have both an edge and a block binding.", target_id))
            source_def = ports_by_block.get(source_id, {})
            target_def = ports_by_block.get(target_id, {})
            source_type = next((str(item.get("type", "")) for item in source_def.get("outputs", []) if str(item.get("name", "")) == source_port), "")
            target_type = next((str(item.get("type", "")) for item in target_def.get("inputs", []) if str(item.get("name", "")) == target_port), "")
            if not source_type or not target_type:
                errors.append(WorkOrderError("port_reference", "Edges must reference declared source and target ports."))
            elif source_type != target_type and not ({source_type, target_type} <= {"Number", "Integer"}):
                errors.append(WorkOrderError("port_type", f"Incompatible edge types: {source_type} cannot connect to {target_type}."))
            elif str(source.get("type", source_type)) != source_type or str(target.get("type", target_type)) != target_type:
                errors.append(WorkOrderError("port_type", "Edge-declared types must match the capability port types."))
            adjacency[source_id].append(target_id)
        block_map = {str(raw.get("id", "")): raw for raw in blocks if isinstance(raw, dict)}
        for raw in blocks:
            if not isinstance(raw, dict) or not raw.get("parent_control_id"):
                continue
            depth = 0
            parent = str(raw.get("parent_control_id", ""))
            seen_parents: set[str] = set()
            while parent:
                if parent in seen_parents:
                    errors.append(WorkOrderError("control_cycle", "Control containers cannot contain themselves.", str(raw.get("id", ""))))
                    break
                seen_parents.add(parent)
                depth += 1
                parent_raw = block_map.get(parent)
                if parent_raw is None:
                    errors.append(
                        WorkOrderError(
                            "control_parent",
                            "A control block parent must reference an existing block.",
                            str(raw.get("id", "")),
                        )
                    )
                    break
                parent = str(parent_raw.get("parent_control_id", "")) if isinstance(parent_raw, dict) else ""
            if depth > _MAX_NESTING:
                errors.append(WorkOrderError("nesting_limit", f"Control containers can be nested at most {_MAX_NESTING} levels.", str(raw.get("id", ""))))
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> None:
            if node in visiting:
                errors.append(WorkOrderError("cycle", "Work order data edges must be acyclic."))
                return
            if node in visited:
                return
            visiting.add(node)
            for child in adjacency.get(node, []):
                visit(child)
            visiting.remove(node)
            visited.add(node)

        for node in adjacency:
            visit(node)
        return errors

    @staticmethod
    def _digest_value(value: Any) -> str:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
        ).hexdigest()

    def _resolve_params(
        self,
        definition: dict[str, Any],
        requested: dict[str, Any],
        requested_origins: dict[str, Any],
        root_params: dict[str, Any],
        type_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        resolved = deepcopy(requested)
        origins = deepcopy(requested_origins)
        preset_id = str(resolved.get("preset_id", ""))
        preset = next((item for item in definition.get("presets", []) if item.get("id") == preset_id), None)
        if preset is not None and preset_id != "custom":
            for key, value in (preset.get("params") or {}).items():
                origin = origins.get(key) if isinstance(origins.get(key), dict) else {}
                if resolved.get(key) is None or origin.get("kind") == "catalog_default":
                    resolved[key] = deepcopy(value)
                    origins[key] = self._origin("preset", preset_id=preset_id)
        for key, value in list(resolved.items()):
            if key == "seed" and value is None:
                resolved[key] = int(root_params.get("seed", 42))
                origins[key] = self._origin("inherited", source="root.seed")
            if key in {"output_name", "filename", "model_id"} and value is None:
                root_name = _slug(str(root_params.get("model_name", "new-model")))
                if type_id == "register_model" and key == "model_id":
                    resolved[key] = root_name
                    origins[key] = self._origin("derived", source="root.model_name")
                    continue
                suffix = {
                    "quantize_mlx": "mlx", "fuse_adapter": "fused", "export_report": "report",
                }.get(type_id, type_id.replace("train_", ""))
                resolved[key] = f"{root_name}-{suffix}"
                origins[key] = self._origin("derived", source="root.model_name")
        return resolved, origins

    @staticmethod
    def _step_sort_key(step: dict[str, Any]) -> tuple[int, int, int, str]:
        lane_order = {"source": 0, "improve": 1, "verify": 2, "deliver": 3}
        try:
            order = int(step.get("order", 0))
        except (TypeError, ValueError):
            order = 0
        return (lane_order.get(str(step.get("lane", "")), 9), order, int(step.get("source_index", 0)), str(step.get("id", "")))

    def _normalize_order(self, order: dict[str, Any]) -> tuple[dict[str, Any], list[WorkOrderError], list[dict[str, Any]]]:
        normalized, _ = self._migrate_work_order(order)
        errors = self._validate_shape(normalized, allow_missing_root=False)
        definitions = self._definitions()
        steps: list[dict[str, Any]] = []
        root = next(
            (
                block
                for block in normalized.get("blocks", [])
                if isinstance(block, dict)
                and block.get("locked") is True
                and block.get("type_id") == "model_development"
            ),
            None,
        )
        root_params = root.get("params") if isinstance(root, dict) and isinstance(root.get("params"), dict) else {}
        if not str(root_params.get("model_name", "")).strip():
            errors.append(WorkOrderError("model_name", "Enter a name in the model development root block."))
        for source_index, block in enumerate(normalized.get("blocks", [])):
            if not isinstance(block, dict):
                continue
            block_id = str(block.get("id", ""))
            type_id = str(block.get("type_id", ""))
            if type_id == "model_development":
                continue
            definition = definitions.get(type_id, {})
            runners = list(definition.get("runners", ["mlx_local"]))
            requested_params = deepcopy(block.get("params") if isinstance(block.get("params"), dict) else {})
            requested_origins = deepcopy(block.get("param_origins") if isinstance(block.get("param_origins"), dict) else {})
            resolved_params, resolved_origins = self._resolve_params(
                definition, requested_params, requested_origins, root_params, type_id
            )
            step = {
                "id": f"step_{block_id}",
                "block_id": block_id,
                "type_id": type_id,
                "lane": str(block.get("lane", "improve")),
                "order": block.get("order", source_index),
                "source_index": source_index,
                "runner_id": runners[0] if runners else "mlx_local",
                "runner_version": _RUNNER_VERSIONS.get(runners[0] if runners else "mlx_local", "2"),
                "depends_on": [],
                "input_bindings": {},
                "requested_params": requested_params,
                "resolved_params": resolved_params,
                "params": deepcopy(resolved_params),
                "param_origins": resolved_origins,
                "requested_param_origins": requested_origins,
                "effective_params_digest": self._digest_value(_redact(resolved_params)),
                "toolchain_profiles": self._toolchains_for_block(type_id, resolved_params),
                "catalog_digest": self._catalog_digest(),
                "status": "pending",
            }
            steps.append(step)
        by_block = {str(step["block_id"]): step for step in steps}
        blocks_by_id = {
            str(block.get("id", "")): block
            for block in normalized.get("blocks", [])
            if isinstance(block, dict)
        }
        edge_by_target: dict[tuple[str, str], dict[str, Any]] = {}
        for edge in normalized.get("edges", []):
            if not isinstance(edge, dict):
                continue
            source = edge.get("from") if isinstance(edge.get("from"), dict) else {}
            target = edge.get("to") if isinstance(edge.get("to"), dict) else {}
            key = (str(target.get("block_id", "")), str(target.get("port", "")))
            if key not in edge_by_target:
                edge_by_target[key] = {"source": source, "target": target}
        ordered_candidates = sorted(steps, key=self._step_sort_key)
        for step in steps:
            type_id = str(step.get("type_id", ""))
            definition = definitions.get(type_id, {})
            params = step.get("resolved_params") if isinstance(step.get("resolved_params"), dict) else {}
            block = blocks_by_id.get(str(step.get("block_id", "")), {})
            requested_bindings = block.get("input_bindings") if isinstance(block.get("input_bindings"), dict) else {}
            if type_id == "load_model" and not str(params.get("ref", "")).strip():
                errors.append(WorkOrderError("model_source", "Load model needs a Hub id or local model path.", str(step.get("block_id", ""))))
            if type_id == "load_dataset" and not str(params.get("ref", "")).strip():
                errors.append(WorkOrderError("dataset_source", "Prepare dataset needs a Hub id or local JSONL path.", str(step.get("block_id", ""))))
            if type_id in {"load_model", "load_dataset"} and params.get("source_kind") == "local":
                path = Path(str(params.get("ref", ""))).expanduser()
                if not path.is_absolute() or not path.exists():
                    errors.append(WorkOrderError("local_source", "Choose an existing local path or select a downloaded model from the local catalog.", str(step.get("block_id", ""))))
                elif type_id == "load_model" and not (path.is_dir() and (path / "config.json").is_file()):
                    errors.append(WorkOrderError("local_source", "The local model folder must contain config.json.", str(step.get("block_id", ""))))
            if type_id == "quantize_mlx" and any(params.get(key) is None for key in ("bits", "group_size", "mode")):
                errors.append(WorkOrderError("quantize_params", "Custom quantization needs bits, group size, and mode.", str(step.get("block_id", ""))))
            if self._runner is None and not self._builtin_supported(type_id):
                errors.append(WorkOrderError("runner_unavailable", f"Runner for '{type_id}' is not implemented in this manager build.", str(step.get("block_id", ""))))
            for input_port in definition.get("inputs", []):
                port_name = str(input_port.get("name", ""))
                target_key = (str(step.get("block_id", "")), port_name)
                explicit = edge_by_target.get(target_key)
                requested = requested_bindings.get(port_name) if isinstance(requested_bindings.get(port_name), dict) else None
                resolved_binding: dict[str, Any] | None = None
                if explicit is not None:
                    source = explicit["source"]
                    source_step = by_block.get(str(source.get("block_id", "")))
                    if source_step is not None:
                        resolved_binding = {
                            "kind": "edge", "step_id": source_step["id"], "port": str(source.get("port", "")),
                            "type": str(input_port.get("type", "")), "origin": "edge", "requested_kind": "edge",
                        }
                elif requested is not None and requested.get("kind") == "literal":
                    resolved_binding = {
                        "kind": "literal", "value": deepcopy(requested.get("value")),
                        "type": str(input_port.get("type", "")), "origin": "literal", "requested_kind": "literal",
                    }
                elif requested is not None and requested.get("kind") in {"auto", "inherited"}:
                    requested_kind = str(requested.get("kind"))
                    selector = str(requested.get("selector", requested.get("name", ""))).strip()
                    current_key = self._step_sort_key(step)
                    candidates: list[tuple[dict[str, Any], str]] = []
                    for candidate in ordered_candidates:
                        if candidate["id"] == step["id"] or self._step_sort_key(candidate) >= current_key:
                            continue
                        candidate_def = definitions.get(str(candidate.get("type_id", "")), {})
                        for output in candidate_def.get("outputs", []):
                            if str(output.get("type", "")) != str(input_port.get("type", "")):
                                continue
                            if selector and selector not in {str(candidate.get("block_id", "")), str(candidate.get("type_id", "")), str(output.get("name", ""))}:
                                continue
                            candidates.append((candidate, str(output.get("name", ""))))
                    if candidates:
                        candidate, output_port = candidates[-1]
                        resolved_binding = {
                            "kind": "edge", "step_id": candidate["id"], "port": output_port,
                            "type": str(input_port.get("type", "")), "origin": requested_kind,
                            "requested_kind": requested_kind,
                        }
                    elif requested_kind == "inherited" and selector in root_params:
                        literal_validator = Draft202012Validator(input_port.get("literal_schema", {}))
                        inherited_value = root_params.get(selector)
                        if literal_validator.is_valid(inherited_value):
                            resolved_binding = {
                                "kind": "literal", "value": deepcopy(inherited_value),
                                "type": str(input_port.get("type", "")), "origin": "inherited",
                                "requested_kind": "inherited", "source": f"root.{selector}",
                            }
                if resolved_binding is not None:
                    step["input_bindings"][port_name] = resolved_binding
                    if resolved_binding.get("kind") == "edge":
                        step["depends_on"].append(str(resolved_binding.get("step_id", "")))
                elif input_port.get("required", True):
                    errors.append(WorkOrderError("missing_input", f"Connect or configure the '{port_name}' input for '{type_id}'.", str(step.get("block_id", ""))))
            step["depends_on"] = list(dict.fromkeys(step["depends_on"]))
        if not any(step["type_id"] in {"register_model", "reveal_artifact", "export_report"} for step in steps):
            errors.append(WorkOrderError("delivery_required", "Add a Deliver block before running the work order."))

        # Freeze a deterministic topological execution order.  Layout/lane
        # order is only the tie-breaker between otherwise independent steps.
        step_by_id = {str(step["id"]): step for step in steps}
        indegree = {step_id: 0 for step_id in step_by_id}
        children: dict[str, list[str]] = {step_id: [] for step_id in step_by_id}
        for step in steps:
            for dependency in step.get("depends_on", []):
                if dependency in step_by_id:
                    indegree[step["id"]] += 1
                    children[dependency].append(step["id"])
        ready = sorted((step_by_id[key] for key, value in indegree.items() if value == 0), key=self._step_sort_key)
        sorted_steps: list[dict[str, Any]] = []
        while ready:
            current = ready.pop(0)
            sorted_steps.append(current)
            for child in children.get(str(current["id"]), []):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(step_by_id[child])
                    ready.sort(key=self._step_sort_key)
        if len(sorted_steps) != len(steps):
            errors.append(WorkOrderError("cycle", "Resolved input bindings must be acyclic."))
            sorted_steps = sorted(steps, key=self._step_sort_key)
        for step in sorted_steps:
            step.pop("source_index", None)
        return normalized, errors, sorted_steps

    def _toolchains_for_block(self, type_id: str, params: dict[str, Any] | None = None) -> list[str]:
        if type_id == "register_model" and params and params.get("test_when_finished") is True:
            return ["mlx-train"]
        if type_id in {"train_dpo", "logits_distill", "evaluate_lm", "compare_models"}:
            return ["hf-posttrain", "evaluation"] if type_id in {"evaluate_lm", "compare_models"} else ["hf-posttrain"]
        if type_id in {"evaluate_loss", "benchmark_model", "quality_gate"}:
            return ["evaluation"]
        if type_id in {"train_sft", "train_lora", "train_qlora", "train_dora", "train_full", "fuse_adapter", "quantize_mlx", "generate_teacher_responses", "response_distill"}:
            return ["mlx-train"]
        return []

    def preflight(self, payload: dict[str, Any]) -> dict[str, Any]:
        order = payload.get("work_order") if isinstance(payload, dict) else None
        if order is None and isinstance(payload, dict):
            order_id = str(payload.get("work_order_id", ""))
            order = self.get_work_order(order_id)
        if not isinstance(order, dict):
            return {"ok": False, "errors": [{"code": "work_order", "message": "Work order is required."}], "warnings": []}
        normalized, errors, steps = self._normalize_order(order)
        warnings: list[dict[str, str]] = []
        profiles = sorted({profile for step in steps for profile in step.get("toolchain_profiles", [])})
        if any(step["type_id"] == "train_full" for step in steps):
            warnings.append({"code": "memory", "message": "Full fine-tuning can exceed Apple unified memory; review the estimate before running."})
        if any(step["type_id"] == "train_dpo" for step in steps):
            warnings.append({"code": "mps", "message": "DPO uses the Transformers/MPS runner and may fall back to CPU for unsupported operations."})
        if any(step["type_id"] == "logits_distill" for step in steps):
            warnings.append({"code": "compatibility", "message": "Logits distillation requires matching tokenizer vocabulary and compatible causal-LM architectures."})
        estimate = self._estimate(order, steps)
        trust_remote_code = any(
            bool((block.get("params") or {}).get("trust_remote_code"))
            for block in order.get("blocks", [])
            if isinstance(block, dict) and isinstance(block.get("params"), dict)
        )
        work_order_hash = self._work_order_hash(normalized, steps, profiles)
        preflight_id = uuid.uuid4().hex
        toolchain_plan = self._toolchain_plan(profiles)
        capability_digest = self._capability_digest()
        downloads = self._download_plan(normalized)
        run_plan = self._make_run_plan(
            normalized=normalized,
            steps=steps,
            toolchains=toolchain_plan,
            downloads=downloads,
            estimates=estimate,
            work_order_hash=work_order_hash,
            capability_digest=capability_digest,
        )
        required_consents = [
            "toolchain_install"
            for item in toolchain_plan
            if item.get("status") != "ready"
        ]
        if trust_remote_code:
            required_consents.append("trust_remote_code")
        result = {
            "ok": not errors,
            "preflight_id": preflight_id,
            "schema_version": 2,
            "work_order_hash": work_order_hash,
            "capability_digest": capability_digest,
            "catalog_digest": self._catalog_digest(),
            "expires_at": time.time() + 900,
            "errors": [error.as_dict() for error in errors],
            "warnings": warnings,
            "normalized_order": normalized,
            "steps": steps,
            "normalized_steps": steps,
            "toolchains": toolchain_plan,
            "downloads": downloads,
            "transfers": [],
            "estimates": estimate,
            "required_consents": list(dict.fromkeys(required_consents)),
            "run_plan": run_plan,
        }
        with self._lock:
            self._preflights[preflight_id] = deepcopy(result)
        return result

    def _capability_digest(self) -> str:
        return hashlib.sha256(json.dumps(self.capabilities(), sort_keys=True).encode()).hexdigest()[:24]

    @staticmethod
    def _work_order_hash(
        normalized: dict[str, Any],
        steps: list[dict[str, Any]],
        profiles: list[str],
    ) -> str:
        digest_payload = {"order": normalized, "steps": steps, "profiles": profiles}
        return hashlib.sha256(json.dumps(digest_payload, sort_keys=True).encode()).hexdigest()

    def _estimate(self, order: dict[str, Any], steps: list[dict[str, Any]]) -> dict[str, Any]:
        repeat_multiplier = 1
        for block in order.get("blocks", []):
            if isinstance(block, dict) and block.get("type_id") == "repeat":
                with suppress(TypeError, ValueError):
                    repeat_multiplier *= max(1, min(_REPEAT_LIMIT, int(block.get("params", {}).get("count", 1))))
        return {
            "step_count": len(steps) * repeat_multiplier,
            "estimated_seconds": max(15, len(steps) * repeat_multiplier * 30),
            "estimated_output_bytes": max(256 * 1024 * 1024, len(steps) * 64 * 1024 * 1024),
            "estimated_memory_bytes": 8 * 1024 * 1024 * 1024,
            "free_bytes": shutil.disk_usage(self.root.parent).free if self.root.parent.exists() else 0,
            "loop_multiplier": repeat_multiplier,
        }

    def _toolchain_plan(self, profiles: list[str]) -> list[dict[str, Any]]:
        catalog = {item["id"]: item for item in self.capabilities()["toolchains"]}
        result = []
        for profile in profiles:
            item = deepcopy(catalog.get(profile, {"id": profile, "packages": []}))
            packages = list(_TOOLCHAIN_LOCK.get(profile, tuple(item.get("packages", []))))
            item["packages"] = [package.split("==", 1)[0] for package in packages]
            item["locked_packages"] = packages
            manifest = {"profile": profile, "packages": packages, "python": sys.version.split()[0]}
            item["manifest_digest"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
            item["status"] = "ready" if self._toolchain_ready(item["manifest_digest"]) else "missing"
            install_path = (self.toolchains_root / item["manifest_digest"]).resolve()
            item["install_path"] = str(install_path)
            item["python_executable"] = str(install_path / "venv" / "bin" / "python")
            result.append(item)
        return result

    def _make_run_plan(
        self,
        *,
        normalized: dict[str, Any],
        steps: list[dict[str, Any]],
        toolchains: list[dict[str, Any]],
        downloads: list[dict[str, Any]],
        estimates: dict[str, Any],
        work_order_hash: str,
        capability_digest: str,
    ) -> dict[str, Any]:
        """Freeze every execution-affecting catalog choice from preflight."""

        toolchains_by_id = {str(item.get("id", "")): item for item in toolchains}
        definitions = self._definitions()
        frozen_steps: list[dict[str, Any]] = []
        for source in steps:
            step = deepcopy(source)
            type_id = str(step.get("type_id", ""))
            definition = definitions.get(type_id, {})
            runner_id = str(step.get("runner_id", ""))
            runner_version = str(step.get("runner_version", ""))
            step["runner"] = {"id": runner_id, "version": runner_version}
            step["definition_version"] = int(definition.get("definition_version", 0) or 0)
            step["catalog_digest"] = self._catalog_digest()
            step["input_ports"] = deepcopy(definition.get("inputs", []))
            step["output_ports"] = deepcopy(definition.get("outputs", []))
            step["toolchains"] = [
                {
                    "id": profile,
                    "manifest_digest": str(toolchains_by_id[profile].get("manifest_digest", "")),
                    "locked_packages": deepcopy(toolchains_by_id[profile].get("locked_packages", [])),
                    "install_path": str(toolchains_by_id[profile].get("install_path", "")),
                    "python_executable": str(toolchains_by_id[profile].get("python_executable", "")),
                }
                for profile in step.get("toolchain_profiles", [])
                if profile in toolchains_by_id
            ]
            resolved = step.get("resolved_params") if isinstance(step.get("resolved_params"), dict) else {}
            step["effective_params_digest"] = self._digest_value(_redact(resolved))
            step["execution_contract_digest"] = self._digest_value(
                {
                    "type_id": type_id,
                    "runner": step["runner"],
                    "definition_version": step["definition_version"],
                    "catalog_digest": step["catalog_digest"],
                    "params": _redact(resolved),
                    "param_origins": _redact(step.get("param_origins", {})),
                    "input_bindings": _redact(step.get("input_bindings", {})),
                    "toolchains": step["toolchains"],
                }
            )
            frozen_steps.append(step)
        plan = {
            "schema_version": 2,
            "work_order_id": str(normalized.get("id", "")),
            "work_order_revision": int(normalized.get("revision", 0) or 0),
            "work_order_name": str(normalized.get("name", "")),
            "work_order_hash": work_order_hash,
            "capability_digest": capability_digest,
            "catalog_digest": self._catalog_digest(),
            "steps": frozen_steps,
            "toolchains": deepcopy(toolchains),
            "downloads": deepcopy(downloads),
            "estimates": deepcopy(estimates),
        }
        plan["plan_digest"] = self._digest_value(plan)
        return plan

    def _validate_run_plan(self, plan: dict[str, Any]) -> None:
        supplied_digest = str(plan.get("plan_digest", ""))
        digest_payload = deepcopy(plan)
        digest_payload.pop("plan_digest", None)
        if not supplied_digest or supplied_digest != self._digest_value(digest_payload):
            raise ValueError("Frozen run plan digest is invalid; run preflight again.")
        if str(plan.get("catalog_digest", "")) != self._catalog_digest():
            raise ValueError("Workshed catalog changed after preflight; run preflight again.")
        if str(plan.get("capability_digest", "")) != self._capability_digest():
            raise ValueError("Local capabilities changed after preflight; run preflight again.")
        definitions = self._definitions()
        expected_toolchains = {
            str(item.get("id", "")): item
            for item in self._toolchain_plan(list(_TOOLCHAIN_LOCK))
        }
        for step in plan.get("steps", []):
            if not isinstance(step, dict):
                raise ValueError("Frozen run plan contains an invalid step.")
            type_id = str(step.get("type_id", ""))
            definition = definitions.get(type_id)
            if definition is None:
                raise ValueError(f"Frozen run plan references unknown block '{type_id}'.")
            runner = step.get("runner") if isinstance(step.get("runner"), dict) else {}
            runner_id = str(runner.get("id", ""))
            if runner_id not in definition.get("runners", []):
                raise ValueError(f"Runner '{runner_id}' is not valid for '{type_id}'.")
            if str(runner.get("version", "")) != _RUNNER_VERSIONS.get(runner_id, ""):
                raise ValueError(f"Runner version changed for '{type_id}'; run preflight again.")
            resolved = step.get("resolved_params") if isinstance(step.get("resolved_params"), dict) else {}
            if self._digest_value(_redact(resolved)) != str(step.get("effective_params_digest", "")):
                raise ValueError(f"Effective parameters changed for '{type_id}'; run preflight again.")
            for toolchain in step.get("toolchains", []):
                if not isinstance(toolchain, dict):
                    raise ValueError("Frozen run plan contains an invalid toolchain.")
                profile = str(toolchain.get("id", ""))
                expected = expected_toolchains.get(profile)
                if expected is None or str(toolchain.get("manifest_digest", "")) != str(expected.get("manifest_digest", "")):
                    raise ValueError(f"Toolchain manifest changed for '{type_id}'; run preflight again.")
                if Path(os.path.abspath(str(toolchain.get("python_executable", "")))) != Path(os.path.abspath(str(expected.get("python_executable", "")))):
                    raise ValueError(f"Toolchain interpreter changed for '{type_id}'; run preflight again.")

    def _toolchain_ready(self, digest: str) -> bool:
        return (self.toolchains_root / digest / ".ready").is_file()

    def _download_plan(self, order: dict[str, Any]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for block in order.get("blocks", []):
            if not isinstance(block, dict):
                continue
            params = block.get("params") if isinstance(block.get("params"), dict) else {}
            if block.get("type_id") not in {"load_model", "load_dataset"}:
                continue
            value = str(params.get("ref", "")).strip()
            if (
                str(params.get("source_kind", "")) == "huggingface"
                and value
                and value not in seen
            ):
                seen.add(value)
                result.append(
                    {
                        "ref": value,
                        "kind": "huggingface",
                        "revision": str(params.get("revision", "")),
                        "artifact_type": "model" if block.get("type_id") == "load_model" else "dataset",
                    }
                )
        return result

    # ------------------------------------------------------------------
    # Run coordinator
    # ------------------------------------------------------------------
    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [deepcopy(item) for item in sorted(self._runs.values(), key=lambda x: str(x.get("created_at", "")), reverse=True)]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._runs.get(str(run_id or ""))
            return deepcopy(value) if value else None

    def _build_run_receipt_locked(self, run: dict[str, Any]) -> dict[str, Any]:
        status = str(run.get("status", ""))
        finalized = status in _FINAL_RECEIPT_STATUSES
        artifact_ids = list(dict.fromkeys(str(item) for item in run.get("artifacts", []) if str(item)))
        artifact_receipts = [
            deepcopy(self._artifacts[artifact_id])
            for artifact_id in artifact_ids
            if artifact_id in self._artifacts
        ]
        steps = []
        for step in run.get("steps", []):
            if not isinstance(step, dict):
                continue
            artifact = step.get("artifact") if isinstance(step.get("artifact"), dict) else {}
            steps.append(
                {
                    "id": str(step.get("id", "")),
                    "block_id": str(step.get("block_id", "")),
                    "type_id": str(step.get("type_id", "")),
                    "status": str(step.get("status", "pending")),
                    "runner": deepcopy(step.get("runner", {})),
                    "toolchains": deepcopy(step.get("toolchains", [])),
                    "catalog_digest": str(step.get("catalog_digest", "")),
                    "execution_contract_digest": str(step.get("execution_contract_digest", "")),
                    "effective_params_digest": str(step.get("effective_params_digest", "")),
                    "requested_params": _redact(deepcopy(step.get("requested_params", {}))),
                    "effective_params": _redact(deepcopy(step.get("resolved_params", {}))),
                    "param_origins": _redact(deepcopy(step.get("param_origins", {}))),
                    "runner_projection": _redact(deepcopy(step.get("runner_projection", {}))),
                    "input_provenance": _redact(deepcopy(step.get("input_provenance", {}))),
                    "artifact_id": str(artifact.get("id", "")),
                    "artifact_receipt_digest": str(artifact.get("receipt_digest", "")),
                    "smoke_test": deepcopy(step.get("smoke_test", artifact.get("smoke_test"))),
                    "error": str(_redact(step.get("error", ""))),
                }
            )
        receipt = {
            "schema_version": 2,
            "run_id": str(run.get("id", "")),
            "status": status,
            "finalized": finalized,
            "work_order_id": str(run.get("work_order_id", "")),
            "work_order_revision": int(run.get("work_order_revision", 0) or 0),
            "work_order_hash": str(run.get("work_order_hash", "")),
            "plan_digest": str(run.get("plan_digest", "")),
            "catalog_digest": str(run.get("catalog_digest", "")),
            "capability_digest": str(run.get("capability_digest", "")),
            "created_at": str(run.get("created_at", "")),
            "finalized_at": str(run.get("finished_at", "")) if finalized else "",
            "steps": steps,
            "artifacts": _redact(artifact_receipts),
        }
        if finalized:
            receipt["receipt_digest"] = self._digest_value(receipt)
        return receipt

    def run_receipt(self, run_id: str) -> dict[str, Any]:
        """Return a provenance receipt for an active or completed local run."""

        with self._lock:
            run = self._runs.get(str(run_id or ""))
            if run is None:
                raise KeyError("Workshed run not found.")
            stored = run.get("run_receipt")
            if isinstance(stored, dict) and stored.get("finalized") is True:
                return deepcopy(stored)
            receipt = self._build_run_receipt_locked(run)
            if receipt.get("finalized") is True:
                run["run_receipt"] = deepcopy(receipt)
                self._persist_locked()
            return receipt

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        preflight_id = str(payload.get("preflight_id", "")) if isinstance(payload, dict) else ""
        with self._lock:
            plan = deepcopy(self._preflights.get(preflight_id))
            if not plan:
                raise ValueError("Preflight plan is missing or expired.")
            if float(plan.get("expires_at", 0)) < time.time():
                raise ValueError("Preflight plan expired; run preflight again.")
            if not bool(plan.get("ok")):
                raise ValueError("Preflight has blocking errors; resolve them before running.")
            # A preflight is a capability- and revision-bound snapshot.  If a
            # saved order changed after preflight, reject it instead of
            # running a stale plan.  A just-created local draft may race the
            # 500 ms autosave, so its embedded normalized order remains the
            # source of truth until it is persisted.
            normalized_order = plan.get("normalized_order") if isinstance(plan.get("normalized_order"), dict) else {}
            current_order = self._work_orders.get(str(normalized_order.get("id", "")))
            if current_order is not None:
                current_normalized, current_errors, current_steps = self._normalize_order(current_order)
                current_profiles = sorted({profile for step in current_steps for profile in step.get("toolchain_profiles", [])})
                if current_errors or self._work_order_hash(current_normalized, current_steps, current_profiles) != str(plan.get("work_order_hash", "")):
                    raise ValueError("Work order changed after preflight; run preflight again.")
            if str(plan.get("capability_digest", "")) != self._capability_digest():
                raise ValueError("Local capabilities changed after preflight; run preflight again.")
            run_plan = deepcopy(plan.get("run_plan"))
            if not isinstance(run_plan, dict):
                raise ValueError("Preflight did not produce a frozen run plan; run preflight again.")
            self._validate_run_plan(run_plan)
            accepted = set(str(item) for item in (payload.get("accepted_consent_ids", []) if isinstance(payload, dict) else []))
            missing_consents = set(plan.get("required_consents", [])) - accepted
            toolchains = [item for item in run_plan.get("toolchains", []) if item.get("status") != "ready"]
            if missing_consents:
                raise PermissionError(f"Accept required Workshed consent before running: {', '.join(sorted(missing_consents))}.")
            active = bool(self._active_run_id and self._runs.get(self._active_run_id, {}).get("status") in _ACTIVE_RUNS)
            run_id = uuid.uuid4().hex
            run = {
                "id": run_id,
                "work_order_id": plan.get("normalized_order", {}).get("id", ""),
                "work_order_revision": plan.get("normalized_order", {}).get("revision", 0),
                "work_order_hash": plan.get("work_order_hash", ""),
                "plan_digest": run_plan.get("plan_digest", ""),
                "catalog_digest": run_plan.get("catalog_digest", ""),
                "capability_digest": run_plan.get("capability_digest", ""),
                "status": "preparing" if toolchains else "queued",
                "stage": "preparing" if toolchains else "queued",
                "progress": 0.0,
                "eta_seconds": plan.get("estimates", {}).get("estimated_seconds"),
                "message": "Preparing local toolchains." if toolchains else "Queued for local execution.",
                "error": "",
                "steps": deepcopy(run_plan.get("steps", [])),
                "events": [],
                "event_cursor": 0,
                "artifacts": [],
                "allowed_actions": ["cancel"],
                "plan": run_plan,
                "accepted_consent_ids": sorted(accepted),
                "created_at": _iso(),
                "updated_at": _iso(),
            }
            self._runs[run_id] = run
            self._cancel[run_id] = threading.Event()
            if active:
                self._queued_runs.append(run_id)
            else:
                self._active_run_id = run_id
            self._persist_locked()
        if not active:
            threading.Thread(target=self._run, args=(run_id, run_plan), daemon=True, name=f"workshed-{run_id[:8]}").start()
        return self.get_run(run_id) or run

    def _event(self, run_id: str, event_type: str, **payload: Any) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return
            cursor = int(run.get("event_cursor", 0)) + 1
            run["event_cursor"] = cursor
            events = list(run.get("events", []))
            events.append({"id": cursor, "type": event_type, "at": _iso(), **_redact(payload)})
            run["events"] = events[-2000:]
            run["updated_at"] = _iso()
            self._persist_locked()

    def _update_run(self, run_id: str, **changes: Any) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return
            run.update(_redact(changes))
            run["updated_at"] = _iso()
            if str(run.get("status", "")) in _FINAL_RECEIPT_STATUSES:
                run["finished_at"] = str(run.get("finished_at") or _iso())
                run["run_receipt"] = self._build_run_receipt_locked(run)
            elif str(run.get("status", "")) in _ACTIVE_RUNS:
                run.pop("run_receipt", None)
                run.pop("finished_at", None)
            self._persist_locked()

    def _resolve_step_inputs(
        self,
        step: dict[str, Any],
        outputs: dict[tuple[str, str], dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
        effective_inputs: dict[str, Any] = {}
        provenance: dict[str, Any] = {}
        parent_artifact_ids: list[str] = []
        bindings = step.get("input_bindings") if isinstance(step.get("input_bindings"), dict) else {}
        for port in step.get("input_ports", []):
            if not isinstance(port, dict):
                continue
            port_name = str(port.get("name", ""))
            binding = bindings.get(port_name) if isinstance(bindings.get(port_name), dict) else None
            if binding is None:
                if port.get("required", True):
                    raise RuntimeError(f"Frozen step is missing required input '{port_name}'.")
                continue
            kind = str(binding.get("kind", ""))
            if kind == "literal":
                value = deepcopy(binding.get("value"))
                effective_inputs[port_name] = value
                provenance[port_name] = {
                    "kind": "literal",
                    "type": str(port.get("type", "")),
                    "value": _redact(deepcopy(value)),
                    "value_digest": self._digest_value(_redact(value)),
                    "origin": str(binding.get("origin", "literal")),
                }
                continue
            if kind != "edge":
                raise RuntimeError(f"Frozen input '{port_name}' has unresolved binding kind '{kind}'.")
            source_key = (str(binding.get("step_id", "")), str(binding.get("port", "")))
            source = outputs.get(source_key)
            if source is None:
                raise RuntimeError(
                    f"Input '{port_name}' could not resolve output {source_key[0]}.{source_key[1]}."
                )
            value = deepcopy(source.get("value"))
            effective_inputs[port_name] = value
            artifact_id = str(source.get("artifact_id", ""))
            if artifact_id:
                parent_artifact_ids.append(artifact_id)
            provenance[port_name] = {
                "kind": "edge",
                "type": str(port.get("type", "")),
                "source_step_id": source_key[0],
                "source_port": source_key[1],
                "artifact_id": artifact_id,
                "artifact_sha256": str(source.get("sha256", "")),
                "value": _redact(deepcopy(value)),
                "value_digest": self._digest_value(_redact(value)),
                "origin": str(binding.get("origin", "edge")),
            }
        execution_step = deepcopy(step)
        execution_step["params"] = deepcopy(step.get("resolved_params", {}))
        execution_step["inputs"] = effective_inputs
        execution_step["effective_inputs"] = deepcopy(effective_inputs)
        execution_step["input_provenance"] = provenance
        execution_step["parent_artifact_ids"] = list(dict.fromkeys(parent_artifact_ids))
        return execution_step, provenance, execution_step["parent_artifact_ids"]

    @staticmethod
    def _record_step_outputs(
        step: dict[str, Any],
        artifact: dict[str, Any] | None,
        outputs: dict[tuple[str, str], dict[str, Any]],
    ) -> None:
        if not isinstance(artifact, dict):
            return
        explicit = artifact.get("outputs") if isinstance(artifact.get("outputs"), dict) else {}
        for port in step.get("output_ports", []):
            if not isinstance(port, dict):
                continue
            port_name = str(port.get("name", ""))
            port_type = str(port.get("type", ""))
            if port_name in explicit:
                value = deepcopy(explicit[port_name])
            elif port_name in artifact:
                value = deepcopy(artifact[port_name])
            elif port_type in _RESOURCE_PORT_TYPES:
                value = deepcopy(
                    artifact.get("runtime_value")
                    or artifact.get("path")
                    or artifact.get("external_ref")
                    or artifact.get("id")
                )
            else:
                value = deepcopy(artifact.get("value"))
            if value is None:
                continue
            outputs[(str(step.get("id", "")), port_name)] = {
                "value": value,
                "type": port_type,
                "artifact_id": str(artifact.get("id", "")),
                "sha256": str(artifact.get("sha256", "")),
            }

    def _run(self, run_id: str, plan: dict[str, Any]) -> None:
        workspace = self.staging_root / run_id
        try:
            self._validate_run_plan(plan)
            workspace.mkdir(parents=True, exist_ok=True)
            toolchains = [item for item in plan.get("toolchains", []) if item.get("status") != "ready"]
            if toolchains:
                self._prepare_toolchains(run_id, toolchains)
            self._update_run(run_id, status="running", stage="running", message="Executing local model-development blocks.")
            steps = list(plan.get("steps", []))
            outputs: dict[tuple[str, str], dict[str, Any]] = {}
            total = max(1, len(steps))
            for index, step in enumerate(steps):
                if self._cancel.get(run_id, threading.Event()).is_set():
                    self._update_run(run_id, status="cancelled", stage="cancelled", progress=0.0, message="Workshed run cancelled.")
                    return
                step_id = str(step.get("id", ""))
                self._update_step(run_id, step_id, status="running", stage="running")
                self._event(run_id, "step_started", step_id=step_id, block_id=step.get("block_id"), type_id=step.get("type_id"))
                try:
                    execution_step, input_provenance, parent_artifact_ids = self._resolve_step_inputs(step, outputs)
                    self._update_step(run_id, step_id, input_provenance=input_provenance)
                    artifact = self._execute_step(run_id, execution_step, workspace)
                    if artifact:
                        artifact = self._register_artifact(
                            run_id,
                            artifact,
                            step=execution_step,
                            input_provenance=input_provenance,
                            parent_artifact_ids=parent_artifact_ids,
                            registered=str(step.get("type_id", "")) == "register_model",
                        )
                    self._record_step_outputs(execution_step, artifact, outputs)
                    self._update_step(
                        run_id,
                        step_id,
                        status="succeeded",
                        stage="completed",
                        artifact=artifact,
                        runner_projection=execution_step.get("runner_projection", {}),
                    )
                except Exception as exc:
                    detail = str(_redact(exc))
                    cancelled = self._cancel.get(run_id, threading.Event()).is_set()
                    self._update_step(run_id, step_id, status="cancelled" if cancelled else "failed", stage="cancelled" if cancelled else "failed", error=detail)
                    self._update_run(run_id, status="cancelled" if cancelled else "failed", stage="cancelled" if cancelled else "failed", error=detail, message="Workshed run cancelled." if cancelled else "A Workshed block failed.")
                    self._event(run_id, "step_cancelled" if cancelled else "step_failed", step_id=step_id, error=detail)
                    return
                self._update_run(run_id, progress=(index + 1) / total, current_step_id=step_id)
                self._event(run_id, "step_succeeded", step_id=step_id, progress=(index + 1) / total)
            self._update_run(run_id, status="completed", stage="completed", progress=1.0, message="Workshed run completed.")
        except Exception as exc:
            self._update_run(run_id, status="failed", stage="failed", error=str(_redact(exc)), message="Workshed run failed.")
        finally:
            shutil.rmtree(workspace, ignore_errors=True)
            next_run: tuple[str, dict[str, Any]] | None = None
            with self._lock:
                if self._active_run_id == run_id:
                    self._active_run_id = ""
                    while self._queued_runs:
                        candidate = self._queued_runs.popleft()
                        queued = self._runs.get(candidate)
                        if queued and queued.get("status") in _ACTIVE_RUNS:
                            self._active_run_id = candidate
                            queued["status"] = (
                                "preparing"
                                if any(
                                    item.get("status") != "ready"
                                    for item in queued.get("plan", {}).get("toolchains", [])
                                    if isinstance(item, dict)
                                )
                                else "queued"
                            )
                            queued["stage"] = queued["status"]
                            queued["message"] = "Queued behind another local Workshed run."
                            queued["updated_at"] = _iso()
                            next_run = (candidate, deepcopy(queued.get("plan", {})))
                            break
                self._persist_locked()
            if next_run:
                threading.Thread(target=self._run, args=next_run, daemon=True, name=f"workshed-{next_run[0][:8]}").start()

    def _prepare_toolchains(self, run_id: str, toolchains: list[dict[str, Any]]) -> None:
        for item in toolchains:
            digest = str(item.get("manifest_digest", "")).strip()
            if not digest:
                continue
            self._event(run_id, "toolchain_prepare", profile=item.get("id"), digest=digest)
            target = self.toolchains_root / digest
            target.mkdir(parents=True, exist_ok=True)
            # The manifest is intentionally allowlisted. Package installation
            # is handled by the dedicated toolchain preparation command. A
            # test runner is a deliberately explicit injection point and may
            # mark its synthetic environment ready without touching Python's
            # global site-packages.
            profile = str(item.get("id", ""))
            packages = list(_TOOLCHAIN_LOCK.get(profile, tuple(str(value) for value in item.get("packages", []))))
            manifest = {"profile": profile, "packages": packages, "digest": digest}
            self._write_json_atomic(target / "manifest.json", manifest)
            if not self._toolchain_ready(digest):
                if self._runner is not None:
                    marker = target / ".ready.tmp"
                    marker.write_text("synthetic runner\n", encoding="utf-8")
                    os.chmod(marker, 0o600)
                    os.replace(marker, target / ".ready")
                    continue
                self._install_toolchain(target, profile, packages)

    def _install_toolchain(self, target: Path, profile: str, packages: list[str]) -> None:
        """Install one locked profile in its own venv and probe it.

        This path is only reached after the UI's explicit toolchain consent.
        It never mutates the app venv or system site-packages and it accepts
        package names solely from ``_TOOLCHAIN_LOCK``.
        """

        venv = target / "venv"
        interpreter = venv / "bin" / "python"
        if not interpreter.is_file():
            completed = subprocess.run(
                [self.python_executable, "-m", "venv", str(venv)],
                cwd=str(target),
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "venv creation failed").strip()[-1000:]
                raise RuntimeError(f"Could not create isolated {profile} toolchain: {detail}")
        if not packages or any(not re.fullmatch(r"[A-Za-z0-9_.-]+==[A-Za-z0-9_.-]+", package) for package in packages):
            raise RuntimeError(f"Toolchain {profile} has an invalid locked package manifest.")
        install = subprocess.run(
            [str(interpreter), "-m", "pip", "install", "--disable-pip-version-check", "--no-input", *packages],
            cwd=str(target),
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
        if install.returncode != 0:
            detail = (install.stderr or install.stdout or "package installation failed").strip()[-1200:]
            raise RuntimeError(f"Could not prepare {profile} toolchain: {detail}")
        probes = {
            "mlx-train": "import mlx_lm",
            "hf-posttrain": "import transformers, datasets, accelerate, trl, peft",
            "evaluation": "import lm_eval, evaluate",
        }
        probe = subprocess.run(
            [str(interpreter), "-c", probes.get(profile, "import sys")],
            cwd=str(target),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if probe.returncode != 0:
            detail = (probe.stderr or probe.stdout or "import probe failed").strip()[-1200:]
            raise RuntimeError(f"{profile} toolchain probe failed: {detail}")
        marker = target / ".ready.tmp"
        marker.write_text("ready\n", encoding="utf-8")
        os.chmod(marker, 0o600)
        os.replace(marker, target / ".ready")

    def prepare_toolchains(self, digests: list[str]) -> list[dict[str, Any]]:
        """Prepare requested manifest digests for an explicit UI action."""

        requested = {str(digest).strip() for digest in digests if str(digest).strip()}
        plans = []
        for profile in _TOOLCHAIN_LOCK:
            item = self._toolchain_plan([profile])[0]
            if item.get("manifest_digest") in requested:
                target = self.toolchains_root / str(item["manifest_digest"])
                target.mkdir(parents=True, exist_ok=True)
                try:
                    self._install_toolchain(target, profile, list(_TOOLCHAIN_LOCK[profile]))
                    item["status"] = "ready"
                except Exception as exc:
                    item["status"] = "failed"
                    item["error"] = str(_redact(exc))
                plans.append(item)
        if requested and not plans:
            raise ValueError("No requested toolchain digest is part of the locked Workshed catalog.")
        return plans

    def artifact_action(self, artifact_id: str, action: str) -> dict[str, Any]:
        action = str(action or "").strip().lower()
        if action not in {"register", "reveal", "export", "trash"}:
            raise ValueError("Unsupported artifact action.")
        with self._lock:
            artifact = self._artifacts.get(str(artifact_id or ""))
            if not artifact:
                raise KeyError("Workshed artifact not found.")
            if action == "trash":
                return self._trash_artifact_locked(str(artifact_id), artifact)
            if artifact.get("deleted") is True:
                raise ValueError("This artifact has been removed; its history is retained as a tombstone.")
            path = Path(str(artifact.get("path", ""))).resolve()
            managed_root = (self.root / "artifacts").resolve()
            if managed_root not in path.parents and path != managed_root:
                raise PermissionError("Artifact path is outside the managed Workshed store.")
            result = deepcopy(artifact)
            result["last_action"] = action
            result["action_at"] = _iso()
            if action == "register":
                result["registered"] = True
            self._artifacts[str(artifact_id)] = result
            self._persist_locked()
            return result

    def unregister_model(self, model_id: str) -> dict[str, Any]:
        """Tombstone one named model alias; never remove its shared directory."""
        identifier = str(model_id or "").strip()
        with self._lock:
            artifact = self._artifacts.get(identifier)
            if not artifact or artifact.get("kind") != "model" or artifact.get("registered") is not True:
                raise KeyError("Registered Workshed model not found.")
            path = Path(str(artifact.get("path", "")))
            managed_root = self.root / "artifacts"
            if not path.is_absolute() or managed_root.resolve() not in path.resolve().parents:
                raise PermissionError("Registered model is outside the managed Workshed store.")
            result = {
                **deepcopy(artifact),
                "registered": False,
                "deleted": True,
                "deleted_at": _iso(),
                "deletion_kind": "alias",
                "last_action": "unregister",
            }
            self._artifacts[identifier] = result
            try:
                self._persist_locked()
            except Exception:
                self._artifacts[identifier] = artifact
                raise
            return deepcopy(result)

    @staticmethod
    def _trash_directory() -> Path:
        if sys.platform != "darwin":
            raise ValueError("Recoverable artifact removal requires macOS Trash.")
        return Path.home() / ".Trash"

    @staticmethod
    def _references_artifact(value: Any, artifact_id: str, path: str) -> bool:
        if isinstance(value, dict):
            return any(WorkshedService._references_artifact(item, artifact_id, path) for item in value.values())
        if isinstance(value, list):
            return any(WorkshedService._references_artifact(item, artifact_id, path) for item in value)
        return isinstance(value, str) and value in {artifact_id, path}

    def _trash_artifact_locked(self, artifact_id: str, artifact: dict[str, Any]) -> dict[str, Any]:
        """Move exactly one owned content directory to Trash after references are gone."""
        if artifact.get("deletion_kind") == "trash":
            return deepcopy(artifact)
        if artifact.get("deleted") is True:
            raise ValueError("Choose the original content artifact, not an unregistered model alias.")
        if artifact.get("registered") is True:
            raise ValueError("Unregister this model before moving its content to Trash.")
        if artifact.get("source_kind") or artifact.get("external_ref"):
            raise PermissionError("Source references are read-only and cannot be removed.")
        raw_path = str(artifact.get("path", ""))
        path = Path(raw_path)
        managed_root = self.root / "artifacts"
        if (
            not path.is_absolute()
            or path.parent != managed_root
            or path.is_symlink()
            or managed_root.is_symlink()
            or not path.is_dir()
            or path.resolve().parent != managed_root.resolve()
            or not re.fullmatch(r"[0-9a-f]{64}", path.name)
            or str(artifact.get("sha256", "")) != path.name
        ):
            raise PermissionError("Only an owned content-addressed Workshed artifact directory can be moved to Trash.")
        if any(item.is_symlink() for item in path.rglob("*")):
            raise PermissionError("Artifacts containing symlinks cannot be removed automatically.")
        for other_id, other in self._artifacts.items():
            if other_id == artifact_id or other.get("deleted") is True:
                continue
            other_path = str(other.get("path", ""))
            if (
                (other_path and Path(other_path).resolve() == path.resolve())
                or artifact_id in other.get("parent_artifact_ids", [])
                or str(other.get("source_artifact_id", "")) == artifact_id
            ):
                raise ValueError(f"Artifact is still referenced by '{other_id}'; shared content is retained.")
        for order in self._work_orders.values():
            if not order.get("archived") and self._references_artifact(order.get("blocks", []), artifact_id, str(path)):
                raise ValueError("Artifact is referenced by a saved work order; archive or edit that order first.")
        for run in self._runs.values():
            if run.get("status") in _ACTIVE_RUNS and self._references_artifact(run, artifact_id, str(path)):
                raise ValueError("Artifact is referenced by an active Workshed run.")
        trash_root = self._trash_directory()
        if trash_root.is_symlink():
            raise PermissionError("The macOS Trash directory must not be a symlink.")
        trash_root.mkdir(mode=0o700, parents=False, exist_ok=True)
        destination = trash_root / f"token-workshed-{path.name[:16]}-{uuid.uuid4().hex[:12]}"
        # Both directories are on the user's home volume. No recursive removal
        # or copy/delete fallback is permitted if the atomic rename fails.
        os.rename(path, destination)
        result = {
            **deepcopy(artifact),
            "registered": False,
            "deleted": True,
            "deleted_at": _iso(),
            "deletion_kind": "trash",
            "trash_path": str(destination),
            "last_action": "trash",
        }
        self._artifacts[artifact_id] = result
        lifecycle_before = {
            str(run_id): deepcopy(run.get("artifact_lifecycle"))
            for run_id, run in self._runs.items()
            if artifact_id in run.get("artifacts", [])
        }
        for run in self._runs.values():
            if artifact_id in run.get("artifacts", []):
                run.setdefault("artifact_lifecycle", {})[artifact_id] = {
                    "deleted": True, "deletion_kind": "trash", "deleted_at": result["deleted_at"],
                }
        try:
            self._persist_locked()
        except Exception:
            self._artifacts[artifact_id] = artifact
            for run_id, previous in lifecycle_before.items():
                if previous is None:
                    self._runs[run_id].pop("artifact_lifecycle", None)
                else:
                    self._runs[run_id]["artifact_lifecycle"] = previous
            os.rename(destination, path)
            raise
        return deepcopy(result)

    def _update_step(self, run_id: str, step_id: str, **changes: Any) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return
            for step in run.get("steps", []):
                if isinstance(step, dict) and step.get("id") == step_id:
                    step.update(_redact(changes))
                    break
            run["updated_at"] = _iso()
            self._persist_locked()

    @staticmethod
    def _required_step_values(step: dict[str, Any], container: str, names: list[str]) -> dict[str, Any]:
        values = step.get(container) if isinstance(step.get(container), dict) else {}
        missing = [name for name in names if name not in values or values[name] is None]
        if missing:
            raise RuntimeError(
                f"Frozen step '{step.get('type_id', '')}' is missing {container}: {', '.join(missing)}."
            )
        return values

    def _toolchain_interpreter(self, step: dict[str, Any], profile: str) -> str:
        manifest = next(
            (
                item
                for item in step.get("toolchains", [])
                if isinstance(item, dict) and str(item.get("id", "")) == profile
            ),
            None,
        )
        if manifest is None:
            raise RuntimeError(f"Frozen step is missing the '{profile}' toolchain manifest.")
        digest = str(manifest.get("manifest_digest", ""))
        toolchains_root = self.toolchains_root.resolve()
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise RuntimeError("Frozen toolchain manifest digest is invalid.")
        toolchain_root = toolchains_root / digest
        expected = toolchain_root / "venv" / "bin" / "python"
        interpreter = Path(os.path.abspath(str(manifest.get("python_executable", ""))))
        if interpreter != Path(os.path.abspath(expected)) or toolchains_root not in expected.parents:
            raise RuntimeError("Frozen toolchain interpreter is outside the managed Workshed toolchain store.")
        managed_venv = (toolchain_root / "venv").resolve()
        if toolchains_root not in managed_venv.parents:
            raise RuntimeError("Frozen toolchain virtual environment resolves outside the managed Workshed store.")
        if not self._toolchain_ready(digest) or not expected.is_file():
            raise RuntimeError(f"The '{profile}' toolchain is not ready; prepare it and run preflight again.")
        # Keep this lexical venv path: on macOS, venv/bin/python commonly links
        # to the base interpreter outside the venv, which is safe and expected.
        return str(expected)

    def _source_artifact(self, step: dict[str, Any], kind: str) -> dict[str, Any]:
        params = self._required_step_values(
            step,
            "params",
            ["source_kind", "ref", "revision"],
        )
        source_kind = str(params["source_kind"])
        reference = str(params["ref"]).strip()
        revision = str(params["revision"])
        if not reference:
            raise ValueError(f"Load {kind} needs a configured reference.")
        runtime_value = reference
        source_artifact_id = ""
        if source_kind == "local":
            path = Path(reference).expanduser()
            if not path.is_absolute() or not path.exists():
                raise ValueError(f"Local {kind} references must be existing absolute paths.")
            runtime_value = str(path.resolve())
        elif source_kind == "managed":
            with self._lock:
                managed = deepcopy(self._artifacts.get(reference))
            if not isinstance(managed, dict) or managed.get("deleted") is True or str(managed.get("kind", "")) != kind:
                raise ValueError(f"Managed {kind} '{reference}' is not available in Workshed.")
            managed_path = Path(str(managed.get("path", ""))).resolve()
            if not managed_path.exists():
                raise ValueError(f"Managed {kind} '{reference}' has no readable artifact path.")
            runtime_value = str(managed_path)
            source_artifact_id = reference
        elif source_kind == "huggingface":
            if "://" in reference or reference.startswith((".", "/")):
                raise ValueError("Hugging Face sources must be repository ids, not URLs or local paths.")
        else:
            raise ValueError(f"Unsupported {kind} source kind '{source_kind}'.")
        descriptor = {
            "kind": kind,
            "source_kind": source_kind,
            "ref": reference,
            "revision": revision,
            "runtime_value": runtime_value,
        }
        digest = self._digest_value(descriptor)
        output_port = "model" if kind == "model" else "dataset"
        return {
            "id": f"workshed/source-{digest[:24]}",
            "kind": kind,
            "external_ref": reference,
            "runtime_value": runtime_value,
            "revision": revision,
            "source_kind": source_kind,
            "sha256": digest,
            "bytes": 0,
            "source_artifact_id": source_artifact_id,
            "outputs": {output_port: runtime_value},
        }

    @staticmethod
    def _append_cli_pairs(command: list[str], params: dict[str, Any], mapping: list[tuple[str, str]]) -> None:
        for param_name, flag in mapping:
            value = params[param_name]
            if isinstance(value, bool):
                command.extend([flag, "true" if value else "false"])
            elif isinstance(value, list):
                command.extend([flag, ",".join(str(item) for item in value)])
            else:
                command.extend([flag, str(value)])

    def _execute_step(self, run_id: str, step: dict[str, Any], workspace: Path) -> dict[str, Any] | None:
        if self._runner is not None:
            return self._runner(step, workspace, self._cancel.setdefault(run_id, threading.Event()), lambda **event: self._event(run_id, "runner", **event))
        type_id = str(step.get("type_id", ""))
        params = step.get("params") if isinstance(step.get("params"), dict) else {}
        inputs = step.get("inputs") if isinstance(step.get("inputs"), dict) else {}
        if type_id == "load_model":
            return self._source_artifact(step, "model")
        if type_id == "load_dataset":
            return self._source_artifact(step, "dataset")
        if type_id in {"evaluate_loss", "benchmark_model", "compare_models", "quality_gate", "generate_teacher_responses", "response_distill", "logits_distill", "train_dpo", "evaluate_lm", "reveal_artifact", "export_report"}:
            raise RuntimeError(f"Runner for '{type_id}' is not available in this installation; run preflight after installing its toolchain.")
        if type_id in {"train_sft", "train_lora", "train_qlora", "train_dora", "train_full"}:
            adapter_training = type_id in {"train_lora", "train_qlora", "train_dora"}
            inputs = self._required_step_values(step, "inputs", ["model", "dataset"])
            required_params = [
                "batch_size", "gradient_accumulation_steps", "max_sequence_length", "max_steps",
                "learning_rate", "optimizer", "weight_decay", "checkpoint_steps", "eval_steps",
                "num_layers", "seed", "output_name",
            ]
            if adapter_training:
                required_params.extend(["lora_rank", "lora_alpha", "lora_dropout", "target_modules"])
            if type_id == "train_qlora":
                required_params.extend(["base_bits", "quantization_mode"])
            if not adapter_training:
                required_params.append("gradient_checkpointing")
            params = self._required_step_values(step, "params", required_params)
            model = str(inputs["model"]).strip()
            data = str(inputs["dataset"]).strip()
            if not model or not data:
                raise ValueError("Training blocks require resolved model and dataset input ports.")
            output = workspace / f"{_slug(str(params['output_name']))}-output"
            interpreter = self._toolchain_interpreter(step, "mlx-train")
            preparation_commands: list[list[str]] = []
            if type_id == "train_qlora":
                quantized_base = workspace / f"{_slug(str(params['output_name']))}-qlora-base"
                quantize_command = [
                    interpreter,
                    "-m", "mlx_lm.convert",
                    "--hf-path", model,
                    "--mlx-path", str(quantized_base),
                    "-q",
                    "--q-bits", str(params["base_bits"]),
                    "--q-mode", str(params["quantization_mode"]),
                ]
                self._run_command(run_id, quantize_command, workspace)
                preparation_commands.append(quantize_command)
                model = str(quantized_base)
            optimizer_config: dict[str, Any] = {
                "adam": {},
                "adamw": {},
                "sgd": {},
                "adafactor": {},
            }
            if str(params["optimizer"]) == "adamw":
                optimizer_config["adamw"] = {"weight_decay": params["weight_decay"]}
            elif float(params["weight_decay"]) != 0.0:
                raise ValueError("Non-zero weight decay is supported only with the AdamW optimizer.")
            nested_config: dict[str, Any] = {"optimizer_config": optimizer_config}
            if adapter_training:
                lora_parameters: dict[str, Any] = {
                    "rank": params["lora_rank"],
                    "dropout": params["lora_dropout"],
                    "scale": float(params["lora_alpha"]) / float(params["lora_rank"]),
                }
                if params["target_modules"]:
                    lora_parameters["keys"] = list(params["target_modules"])
                nested_config["lora_parameters"] = lora_parameters
            else:
                nested_config["grad_checkpoint"] = bool(params["gradient_checkpointing"])
            config_path = workspace / f"{_slug(str(params['output_name']))}-runner-config.json"
            self._write_json_atomic(config_path, nested_config)
            command = [
                interpreter,
                "-m", "mlx_lm.lora",
                "--config", str(config_path),
                "--model", model,
                "--data", data,
                "--train",
                "--adapter-path", str(output),
            ]
            fine_tune_type = {"train_sft": "full", "train_lora": "lora", "train_qlora": "lora", "train_dora": "dora", "train_full": "full"}[type_id]
            command.extend(["--fine-tune-type", fine_tune_type])
            self._append_cli_pairs(
                command,
                params,
                [
                    ("batch_size", "--batch-size"),
                    ("gradient_accumulation_steps", "--grad-accumulation-steps"),
                    ("max_sequence_length", "--max-seq-length"),
                    ("max_steps", "--iters"),
                    ("learning_rate", "--learning-rate"),
                    ("optimizer", "--optimizer"),
                    ("checkpoint_steps", "--save-every"),
                    ("eval_steps", "--steps-per-eval"),
                    ("seed", "--seed"),
                ],
            )
            self._append_cli_pairs(command, params, [("num_layers", "--num-layers")])
            if not adapter_training and params["gradient_checkpointing"]:
                command.append("--grad-checkpoint")
            step["runner_projection"] = {
                "preparation_argv": preparation_commands,
                "argv": command,
                "config": nested_config,
                "config_digest": self._digest_value(nested_config),
            }
            self._run_command(run_id, command, workspace)
            if fine_tune_type != "full":
                return self._manifest_artifact(output, "adapter", step)
            fused_output = workspace / f"{_slug(str(params['output_name']))}-model"
            fuse_command = [
                interpreter,
                "-m", "mlx_lm.fuse",
                "--model", model,
                "--adapter-path", str(output),
                "--save-path", str(fused_output),
            ]
            step["runner_projection"]["finalize_argv"] = fuse_command
            self._run_command(run_id, fuse_command, workspace)
            return self._manifest_artifact(fused_output, "model", step)
        if type_id == "fuse_adapter":
            inputs = self._required_step_values(step, "inputs", ["model", "adapter"])
            params = self._required_step_values(step, "params", ["output_name", "dequantize"])
            model = str(inputs["model"]).strip()
            adapter = str(inputs["adapter"]).strip()
            if not model or not adapter:
                raise ValueError("Fuse adapter requires resolved model and adapter input ports.")
            output = workspace / f"{_slug(str(params['output_name']))}-fused"
            command = [
                self._toolchain_interpreter(step, "mlx-train"),
                "-m",
                "mlx_lm.fuse",
                "--model",
                model,
                "--adapter-path",
                adapter,
                "--save-path",
                str(output),
            ]
            if bool(params["dequantize"]):
                command.append("--dequantize")
            step["runner_projection"] = {"argv": command}
            self._run_command(run_id, command, workspace)
            return self._manifest_artifact(output, "model", step)
        if type_id == "quantize_mlx":
            inputs = self._required_step_values(step, "inputs", ["model"])
            params = self._required_step_values(step, "params", ["bits", "group_size", "mode", "output_name"])
            model = str(inputs["model"]).strip()
            if not model:
                raise ValueError("MLX quantization requires a resolved model input port.")
            output = workspace / f"{_slug(str(params['output_name']))}-mlx"
            command = [
                self._toolchain_interpreter(step, "mlx-train"),
                "-m", "mlx_lm.convert",
                "--hf-path", model,
                "--mlx-path", str(output),
                "-q",
                "--q-bits", str(params["bits"]),
                "--q-group-size", str(params["group_size"]),
                "--q-mode", str(params["mode"]),
            ]
            step["runner_projection"] = {"argv": command}
            self._run_command(run_id, command, workspace)
            return self._manifest_artifact(output, "model", step)
        if type_id == "register_model":
            inputs = self._required_step_values(step, "inputs", ["model"])
            params = self._required_step_values(step, "params", ["model_id", "collision_policy"])
            model_path = str(inputs["model"]).strip()
            requested_id = str(params["model_id"]).strip()
            artifact_id = requested_id if requested_id.startswith("workshed/") else f"workshed/{_slug(requested_id)}"
            if not _SAFE_NAME.fullmatch(artifact_id.removeprefix("workshed/")):
                raise ValueError("Registered model id must be a safe local identifier.")
            collision_policy = str(params["collision_policy"])
            with self._lock:
                collision = artifact_id in self._artifacts
                if collision and collision_policy == "fail":
                    raise ValueError(f"Managed model '{artifact_id}' already exists.")
                if collision and collision_policy == "version":
                    base_id = artifact_id
                    index = 2
                    while artifact_id in self._artifacts:
                        artifact_id = f"{base_id}-{index}"
                        index += 1
            smoke_test: dict[str, Any] | None = None
            runner_projection: dict[str, Any] = {}
            if bool(params.get("test_when_finished", False)):
                if not model_path:
                    raise ValueError("The saved model path is empty; the smoke test cannot load it.")
                command = [
                    self._toolchain_interpreter(step, "mlx-train"),
                    "-m", "mlx_lm.generate",
                    "--model", model_path,
                    "--prompt", "Reply with the word ready.",
                    "--max-tokens", "16",
                ]
                runner_projection = {"argv": command}
                self._run_command(run_id, command, workspace)
                smoke_test = {
                    "status": "passed",
                    "prompt": "Reply with the word ready.",
                    "max_tokens": 16,
                    "runner": "mlx_lm.generate",
                }
            return {
                "id": artifact_id,
                "kind": "model",
                "path": model_path,
                "runtime_value": model_path,
                "outputs": {"artifact": model_path},
                **({"smoke_test": smoke_test} if smoke_test else {}),
                **({"runner_projection": runner_projection} if runner_projection else {}),
            }
        if type_id in {"repeat", "for_each", "if_else", "retry", "stop"}:
            raise RuntimeError(f"Structured control runner for '{type_id}' is not available in this manager build.")
        raise RuntimeError(f"Unsupported Workshed block: {type_id}")

    def _run_command(self, run_id: str, command: list[str], cwd: Path) -> None:
        self._event(run_id, "command_started", executable=command[0], args=command[1:])
        process = subprocess.Popen(command, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        with self._lock:
            self._processes[run_id] = process
        try:
            while True:
                if self._cancel.get(run_id, threading.Event()).is_set():
                    with suppress(ProcessLookupError, OSError):
                        os.killpg(process.pid, signal.SIGTERM)
                    raise RuntimeError("Cancellation requested.")
                line = process.stdout.readline() if process.stdout is not None else ""
                if line:
                    self._event(run_id, "log", message=line.rstrip())
                elif process.poll() is not None:
                    break
                else:
                    time.sleep(0.08)
            code = process.wait(timeout=15)
            if code != 0:
                raise RuntimeError(f"Runner exited with code {code}.")
        finally:
            with self._lock:
                self._processes.pop(run_id, None)

    def _manifest_artifact(self, path: Path, kind: str, step: dict[str, Any]) -> dict[str, Any]:
        if not path.exists():
            raise RuntimeError("Runner completed without producing an artifact.")
        digest = hashlib.sha256()
        total = 0
        for item in path.rglob("*") if path.is_dir() else [path]:
            if item.is_file():
                total += item.stat().st_size
                with item.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
        digest_hex = digest.hexdigest()
        managed_root = self.root / "artifacts" / digest_hex
        managed_root.parent.mkdir(parents=True, exist_ok=True)
        if not managed_root.exists():
            if path.is_dir():
                shutil.copytree(path, managed_root)
            else:
                managed_root.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, managed_root / path.name)
        return {"id": f"workshed/{digest_hex[:24]}", "kind": kind, "path": str(managed_root.resolve()), "bytes": total, "sha256": digest_hex, "source_step": step.get("id", "")}

    def _register_artifact(
        self,
        run_id: str,
        artifact: dict[str, Any],
        *,
        step: dict[str, Any],
        input_provenance: dict[str, Any],
        parent_artifact_ids: list[str],
        registered: bool = False,
    ) -> dict[str, Any]:
        artifact_id = str(artifact.get("id", "")).strip()
        if not artifact_id:
            raise RuntimeError("Runner returned an artifact without an id.")
        # Runner adapters may return a staging path.  Promote it into the
        # content-addressed Workshed store before the run workspace is
        # removed; never leave a managed registry pointing at a disposable
        # or user-owned directory.
        normalized = deepcopy(artifact)
        raw_path_text = str(normalized.get("path", "")).strip()
        raw_path = Path(raw_path_text).expanduser() if raw_path_text else None
        if raw_path is not None and raw_path.exists():
            raw_path = raw_path.resolve()
            managed_root = (self.root / "artifacts").resolve()
            if managed_root not in raw_path.parents and raw_path != managed_root:
                digest = hashlib.sha256()
                total = 0
                files = [item for item in raw_path.rglob("*") if item.is_file()] if raw_path.is_dir() else [raw_path]
                for item in sorted(files, key=lambda value: str(value)):
                    total += item.stat().st_size
                    with item.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(chunk)
                digest_hex = digest.hexdigest()
                destination = managed_root / digest_hex
                managed_root.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    if raw_path.is_dir():
                        shutil.copytree(raw_path, destination)
                    else:
                        destination.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(raw_path, destination / raw_path.name)
                normalized["path"] = str(destination)
                normalized["sha256"] = digest_hex
                normalized["bytes"] = total
                if not normalized.get("runtime_value"):
                    normalized["runtime_value"] = str(destination)
        parent_ids = list(
            dict.fromkeys(
                [
                    *[str(value) for value in parent_artifact_ids if str(value)],
                    str(normalized.get("source_artifact_id", "")),
                ]
            )
        )
        parent_ids = [value for value in parent_ids if value and value != artifact_id]
        with self._lock:
            run = self._runs.get(run_id) or {}
            provenance = {
                "schema_version": 2,
                "work_order_id": str(run.get("work_order_id", "")),
                "work_order_revision": int(run.get("work_order_revision", 0) or 0),
                "work_order_hash": str(run.get("work_order_hash", "")),
                "run_id": run_id,
                "plan_digest": str(run.get("plan_digest", "")),
                "step_id": str(step.get("id", "")),
                "block_id": str(step.get("block_id", "")),
                "type_id": str(step.get("type_id", "")),
                "runner": deepcopy(step.get("runner", {})),
                "toolchains": deepcopy(step.get("toolchains", [])),
                "catalog_digest": str(step.get("catalog_digest", "")),
                "definition_version": int(step.get("definition_version", 0) or 0),
                "execution_contract_digest": str(step.get("execution_contract_digest", "")),
                "requested_params": _redact(deepcopy(step.get("requested_params", {}))),
                "effective_params": _redact(deepcopy(step.get("resolved_params", {}))),
                "param_origins": _redact(deepcopy(step.get("param_origins", {}))),
                "effective_params_digest": str(step.get("effective_params_digest", "")),
                "runner_projection": _redact(deepcopy(step.get("runner_projection", {}))),
                "inputs": _redact(deepcopy(input_provenance)),
                "parent_artifact_ids": parent_ids,
            }
            receipt_payload = {
                "id": artifact_id,
                "kind": str(normalized.get("kind", "")),
                "sha256": str(normalized.get("sha256", "")),
                "bytes": int(normalized.get("bytes", 0) or 0),
                "provenance": provenance,
            }
            stored = {
                **normalized,
                "registered": bool(registered or normalized.get("registered") is True),
                "run_id": run_id,
                "work_order_id": str(run.get("work_order_id", "")),
                "work_order_revision": int(run.get("work_order_revision", 0) or 0),
                "source_step": str(step.get("id", "")),
                "parent_artifact_ids": parent_ids,
                "effective_params_digest": str(step.get("effective_params_digest", "")),
                "provenance": provenance,
                "receipt_digest": self._digest_value(receipt_payload),
                "created_at": _iso(),
            }
            self._artifacts[artifact_id] = stored
            if run:
                run.setdefault("artifacts", []).append(artifact_id)
            self._persist_locked()
            return deepcopy(stored)

    def action(self, run_id: str, action: str) -> dict[str, Any]:
        action = str(action or "").strip().lower()
        if action not in {"cancel", "pause", "resume", "retry"}:
            raise ValueError("Unsupported Workshed run action.")
        with self._lock:
            run = self._runs.get(str(run_id or ""))
            if not run:
                raise KeyError("Workshed run not found.")
            status = str(run.get("status", ""))
            if action == "cancel" and status not in _TERMINAL_RUNS:
                self._cancel.setdefault(run_id, threading.Event()).set()
                run["status"] = "cancelling"
                run["stage"] = "cancelling"
                run["message"] = "Cancellation requested."
            elif action == "retry" and status in {"failed", "interrupted"}:
                run["status"] = "queued"
                run["stage"] = "queued"
                run["error"] = ""
                run["message"] = "Queued for retry from the failed step."
                self._cancel[run_id] = threading.Event()
                retry_plan = deepcopy(run.get("plan") or {"steps": run.get("steps", []), "toolchains": [], "estimates": {}})
                threading.Thread(target=self._run, args=(run_id, retry_plan), daemon=True, name=f"workshed-retry-{run_id[:8]}").start()
            elif action in {"pause", "resume"}:
                raise ValueError("This local runner does not support safe pause; cancel and retry instead.")
            else:
                return deepcopy(run)
            self._persist_locked()
            return deepcopy(run)

    def artifacts(self) -> list[dict[str, Any]]:
        with self._lock:
            return [deepcopy(item) for item in self._artifacts.values()]

    def managed_model_ids(self) -> list[str]:
        with self._lock:
            return sorted(
                str(item.get("id", ""))
                for item in self._artifacts.values()
                if item.get("kind") == "model"
                and str(item.get("id", "")).startswith("workshed/")
                and item.get("registered") is True
                and item.get("deleted") is not True
                and Path(str(item.get("path", ""))).is_dir()
            )

    def resolve_model(self, model_id: str) -> str | None:
        identifier = str(model_id or "").strip()
        if not identifier.startswith("workshed/"):
            return None
        with self._lock:
            artifact = self._artifacts.get(identifier)
            if not artifact or artifact.get("deleted") is True or artifact.get("kind") != "model" or artifact.get("registered") is not True:
                return None
            path = Path(str(artifact.get("path", ""))).resolve()
            managed_root = (self.root / "artifacts").resolve()
            if managed_root not in path.parents or not path.exists():
                return None
            return str(path)

    def shutdown(self) -> None:
        with self._lock:
            for event in self._cancel.values():
                event.set()
            processes = list(self._processes.values())
        for process in processes:
            with suppress(OSError):
                process.terminate()


__all__ = ["WorkshedService"]
