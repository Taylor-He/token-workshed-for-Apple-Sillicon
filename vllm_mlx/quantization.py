# SPDX-License-Identifier: Apache-2.0
"""Quantization jobs for the Token Workshed manager.

The desktop UI deliberately talks to this module through the local manager
only.  Local MLX jobs run in a private temporary workspace and are promoted
to the managed model directory with an atomic rename.  CUDA jobs are proxied
to a separately deployed worker; the worker credential is never written to
the desktop state file.

This module is intentionally dependency-light.  It uses ``requests`` (which
is already a desktop dependency) for the worker contract and invokes MLX-LM
as a subprocess so a cancelled job can be terminated without touching a
user-owned source directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_SAFE_OUTPUT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")
_REDACTED = "[redacted]"
_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
_ACTIVE_STATUSES = {"queued", "running", "pausing", "paused", "awaiting_delivery"}
_LOCAL_RUNNING_STATUSES = {"queued", "running", "pausing", "paused", "cancelling"}


@dataclass(frozen=True)
class QuantizationRequest:
    """Stable service-level request contract shared by local and Worker paths."""

    source: dict[str, Any]
    engine: str
    worker_id: str = ""
    preset_id: str = "mlx-balanced"
    overrides: dict[str, Any] = field(default_factory=dict)
    output_name: str = ""


@dataclass(frozen=True)
class QuantizationCapabilities:
    """Capability response exposed to the native guided workbench."""

    engines: list[dict[str, Any]] = field(default_factory=list)
    workers: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class QuantizationDelivery:
    """Explicit post-completion delivery choice."""

    job_id: str
    action: str


@dataclass(frozen=True)
class QuantizationJob:
    """Serializable job snapshot used by integrations and diagnostics."""

    id: str
    status: str
    stage: str
    progress: float
    engine: str
    artifact: dict[str, Any] | None = None


def _now() -> float:
    return time.time()


def _iso(value: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(value or _now()))


def _safe_name(value: str, fallback: str = "quantized-model") -> str:
    value = str(value or "").strip()
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-_")
    return value[:120] or fallback


def _redact(value: Any) -> Any:
    """Redact credentials and auth headers from worker responses/logs."""

    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in ("token", "secret", "password", "authorization")):
                result[str(key)] = _REDACTED
            else:
                result[str(key)] = _redact(item)
        return result
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        # Avoid leaking a bearer token in an otherwise useful error string.
        value = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~+/-]+", "Bearer " + _REDACTED, value)
        return value[:4000]
    return value


class _KeychainSecretStore:
    """Small Keychain adapter with a process-local fallback for tests/Linux."""

    _memory: dict[str, str] = {}
    _memory_lock = threading.RLock()
    _service = "com.token-workshed.quant-worker"

    @classmethod
    def _key(cls, account: str) -> str:
        return f"{cls._service}:{account}"

    @classmethod
    def set(cls, account: str, value: str) -> None:
        account = str(account or "").strip()
        if not account:
            raise ValueError("Keychain account cannot be empty.")
        with cls._memory_lock:
            cls._memory[cls._key(account)] = str(value or "")
        if sys.platform != "darwin" or not value:
            return
        # ``security`` receives the secret through stdin, so it never appears
        # in argv or the process list.  A failure is not silently ignored: the
        # in-memory value still lets the current process continue, while the
        # caller receives a clear warning via ``keychain_persisted``.
        try:
            subprocess.run(
                ["/usr/bin/security", "delete-generic-password", "-s", cls._service, "-a", account],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=5,
            )
            subprocess.run(
                ["/usr/bin/security", "add-generic-password", "-U", "-s", cls._service, "-a", account, "-w"],
                input=f"{value}\n",
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return

    @classmethod
    def get(cls, account: str) -> str:
        account = str(account or "").strip()
        if not account:
            return ""
        if sys.platform == "darwin":
            try:
                result = subprocess.run(
                    ["/usr/bin/security", "find-generic-password", "-w", "-s", cls._service, "-a", account],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    check=False,
                    timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()
            except (OSError, subprocess.SubprocessError):
                pass
        with cls._memory_lock:
            return cls._memory.get(cls._key(account), "")

    @classmethod
    def delete(cls, account: str) -> None:
        with cls._memory_lock:
            cls._memory.pop(cls._key(str(account or "").strip()), None)
        if sys.platform == "darwin":
            with suppress(OSError, subprocess.SubprocessError):
                subprocess.run(
                    ["/usr/bin/security", "delete-generic-password", "-s", cls._service, "-a", str(account or "").strip()],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=5,
                )


class QuantizationService:
    """Persistent local/remote quantization job manager."""

    def __init__(
        self,
        *,
        artifact_root: str | Path | None = None,
        state_path: str | Path | None = None,
        worker_state_path: str | Path | None = None,
        python_executable: str | None = None,
        request_timeout_s: float = 25.0,
    ) -> None:
        configured_root = str(os.environ.get("TOKEN_WORKSHED_QUANTIZATION_ROOT", "") or "").strip()
        default_root = (
            Path(configured_root).expanduser()
            if configured_root
            else Path.home()
            / "Library"
            / "Application Support"
            / "token-workshed"
            / "quantized"
        )
        self.artifact_root = Path(artifact_root or default_root).expanduser().resolve()
        self.temp_root = self.artifact_root.parent / f".{self.artifact_root.name}-tmp"
        default_state = os.environ.get("TOKEN_WORKSHED_QUANTIZATION_STATE", "")
        self.state_path = Path(state_path or default_state or (self.artifact_root.parent / "quantization_jobs.json")).expanduser().resolve()
        default_workers = os.environ.get("TOKEN_WORKSHED_QUANTIZATION_WORKERS", "")
        self.worker_state_path = Path(worker_state_path or default_workers or (self.artifact_root.parent / "quantization_workers.json")).expanduser().resolve()
        self.python_executable = str(python_executable or sys.executable)
        self.request_timeout_s = max(3.0, float(request_timeout_s))
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._worker_configs: dict[str, dict[str, str]] = {}
        self._local_active_id = ""
        self._load_state()

    # ------------------------------------------------------------------
    # Persistent state and worker configuration
    # ------------------------------------------------------------------
    def _load_state(self) -> None:
        with self._lock:
            try:
                raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                raw = {}
            jobs = raw.get("jobs", raw) if isinstance(raw, dict) else {}
            if isinstance(jobs, dict):
                for job_id, job in jobs.items():
                    if isinstance(job, dict) and str(job_id).strip():
                        restored = _redact(job)
                        if restored.get("status") in _LOCAL_RUNNING_STATUSES and restored.get("engine") != "vllm_cuda":
                            restored["status"] = "failed"
                            restored["stage"] = "recovered"
                            restored["error"] = "Manager restarted before the job could be reattached."
                        self._jobs[str(job_id)] = restored
            try:
                workers = json.loads(self.worker_state_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                workers = {}
            if isinstance(workers, dict):
                for worker_id, worker in workers.items():
                    if isinstance(worker, dict) and _SAFE_ID.fullmatch(str(worker_id)):
                        url = str(worker.get("url", "") or "").strip()
                        if url.startswith("https://"):
                            self._worker_configs[str(worker_id)] = {
                                "id": str(worker_id),
                                "label": str(worker.get("label", "") or worker_id)[:120],
                                "url": url.rstrip("/"),
                            }
            self._rebuild_local_active_locked()
            remote_jobs = [
                (job_id, job)
                for job_id, job in self._jobs.items()
                if job.get("engine") == "vllm_cuda"
                and job.get("status") in _ACTIVE_STATUSES
                and job.get("remote_job_id")
            ]
            for job_id, _job in remote_jobs:
                self._cancel_events.setdefault(job_id, threading.Event())
        for job_id, job in remote_jobs:
            request = {
                "source": deepcopy(job.get("source") or {}),
                "engine": "vllm_cuda",
                "worker_id": str(job.get("worker_id", "")),
                "preset_id": str(job.get("preset_id", "cuda-balanced")),
                "overrides": deepcopy(job.get("overrides") or {}),
                "output_name": str(job.get("output_name", "quantized-model")),
                "remote_job_id": str(job.get("remote_job_id", "")),
            }
            threading.Thread(
                target=self._run_remote,
                args=(job_id, request),
                daemon=True,
                name=f"quantization-reconnect-{job_id[:8]}",
            ).start()

    def _write_json_atomic(self, path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temporary, path)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary)

    def _persist_locked(self) -> None:
        serializable = {"version": 1, "jobs": _redact(self._jobs)}
        self._write_json_atomic(self.state_path, serializable)

    def _persist_workers_locked(self) -> None:
        self._write_json_atomic(self.worker_state_path, _redact(self._worker_configs))

    def _rebuild_local_active_locked(self) -> None:
        self._local_active_id = ""
        for job_id, job in self._jobs.items():
            if job.get("engine") == "mlx_local" and job.get("status") in _LOCAL_RUNNING_STATUSES:
                self._local_active_id = job_id
                break

    def configure_worker(self, worker_id: str, label: str, url: str, token: str) -> dict[str, Any]:
        worker_id = str(worker_id or "").strip()
        label = str(label or worker_id).strip()[:120]
        url = str(url or "").strip().rstrip("/")
        token = str(token or "").strip()
        if not _SAFE_ID.fullmatch(worker_id):
            raise ValueError("Worker id must contain only letters, numbers, '.', '_' or '-'.")
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("Worker URL must use HTTPS.")
        if not token:
            raise ValueError("Worker access token cannot be empty.")
        with self._lock:
            self._worker_configs[worker_id] = {"id": worker_id, "label": label or worker_id, "url": url}
            _KeychainSecretStore.set(worker_id, token)
            self._persist_workers_locked()
            return self._worker_public_locked(worker_id)

    def remove_worker(self, worker_id: str) -> bool:
        with self._lock:
            existed = self._worker_configs.pop(str(worker_id or "").strip(), None) is not None
            if existed:
                _KeychainSecretStore.delete(worker_id)
                self._persist_workers_locked()
            return existed

    def _worker_public_locked(self, worker_id: str, *, status: str = "configured") -> dict[str, Any]:
        worker = dict(self._worker_configs.get(worker_id, {}))
        if not worker:
            return {"id": worker_id, "status": "missing"}
        return {"id": worker_id, "label": worker.get("label", worker_id), "url": worker.get("url", ""), "status": status, "auth": bool(_KeychainSecretStore.get(worker_id))}

    def workers(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._worker_public_locked(worker_id) for worker_id in sorted(self._worker_configs)]

    # ------------------------------------------------------------------
    # Capabilities, validation, and public job operations
    # ------------------------------------------------------------------
    def capabilities(self) -> dict[str, Any]:
        local = {
            "id": "mlx_local",
            "label": "MLX · This Mac",
            "kind": "local",
            "status": "ready" if sys.platform == "darwin" else "available",
            "source_kinds": ["huggingface", "local_path"],
            "presets": [
                {"id": "mlx-fast", "label": "Fast", "bits": 4, "group_size": 64, "mode": "affine"},
                {"id": "mlx-balanced", "label": "Balanced", "bits": 4, "group_size": 128, "mode": "affine"},
                {"id": "mlx-quality", "label": "Quality", "bits": 8, "group_size": 64, "mode": "affine"},
                {"id": "custom", "label": "Custom", "bits": 4, "group_size": 64, "mode": "affine"},
            ],
            "advanced": {"bits": [4, 8], "group_sizes": [32, 64, 128, 256], "modes": ["affine", "mxfp4"]},
        }
        with self._lock:
            worker_ids = sorted(self._worker_configs)
            workers = [self._worker_public_locked(worker_id) for worker_id in worker_ids]
        remote_engines: list[dict[str, Any]] = []
        for worker_id in worker_ids:
            worker = next((item for item in workers if item.get("id") == worker_id), {})
            try:
                response = self._worker_request(worker_id, "GET", "/v1/quantization/capabilities")
                remote = response.json()
                if not isinstance(remote, dict):
                    raise RuntimeError("Worker capability response was not an object.")
                remote_engines.append({
                    "id": "vllm_cuda",
                    "label": f"vLLM/CUDA · {worker.get('label', worker_id)}",
                    "kind": "remote",
                    "worker_id": worker_id,
                    "status": "ready",
                    "source_kinds": ["huggingface"],
                    "presets": remote.get("presets", []),
                    "advanced": remote.get("advanced", {}),
                    "pause": bool(remote.get("pause", False)),
                    "worker": remote.get("worker", {}),
                })
                worker["status"] = "ready"
            except Exception as exc:
                worker["status"] = "unreachable"
                worker["error"] = str(_redact(exc))
                remote_engines.append({
                    "id": "vllm_cuda",
                    "label": f"vLLM/CUDA · {worker.get('label', worker_id)}",
                    "kind": "remote",
                    "worker_id": worker_id,
                    "status": "unreachable",
                    "source_kinds": ["huggingface"],
                    "presets": [],
                    "advanced": {},
                    "pause": False,
                })
        return {
            "ok": True,
            "engines": [local, *remote_engines],
            "workers": workers,
            "warnings": (["No CUDA Worker configured. Add an HTTPS Worker to enable remote quantization."] if not workers else []),
            "managed_model_root": str(self.artifact_root),
        }

    def _normalize_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Quantization request must be an object.")
        source = payload.get("source")
        if not isinstance(source, dict):
            raise ValueError("Quantization source is required.")
        kind = str(source.get("kind", "")).strip().lower()
        ref = str(source.get("ref", "")).strip()
        revision = str(source.get("revision", "")).strip()[:200]
        if kind not in {"huggingface", "local_path"}:
            raise ValueError("Source kind must be 'huggingface' or 'local_path'.")
        if not ref or len(ref) > 4096:
            raise ValueError("Source reference cannot be empty.")
        engine = str(payload.get("engine", "mlx_local")).strip().lower()
        if engine not in {"mlx_local", "vllm_cuda"}:
            raise ValueError("Engine must be 'mlx_local' or 'vllm_cuda'.")
        worker_id = str(payload.get("worker_id", "")).strip()
        preset_id = str(payload.get("preset_id", "mlx-balanced")).strip() or "mlx-balanced"
        overrides = payload.get("overrides")
        if not isinstance(overrides, dict):
            overrides = {}
        requested_output = str(payload.get("output_name", "") or "").strip()
        if requested_output and not _SAFE_OUTPUT.fullmatch(requested_output):
            raise ValueError("Output name contains unsupported characters.")
        output_name = requested_output
        if not output_name:
            base = ref.rstrip("/").split("/")[-1] if kind == "huggingface" else Path(ref).name
            output_name = _safe_name(f"{base}-{preset_id}")
        normalized = {
            "source": {"kind": kind, "ref": ref, "revision": revision},
            "engine": engine,
            "worker_id": worker_id,
            "preset_id": preset_id,
            "overrides": deepcopy(overrides),
            "output_name": output_name,
        }
        if kind == "local_path":
            source_path = Path(ref).expanduser().resolve()
            normalized["source_path"] = str(source_path)
        return normalized

    def _preset(self, request: dict[str, Any]) -> dict[str, Any]:
        bits = int(request.get("overrides", {}).get("bits", 0) or 0)
        group_size = int(request.get("overrides", {}).get("group_size", 0) or 0)
        mode = str(request.get("overrides", {}).get("mode", "") or "").strip()
        algorithm = str(request.get("overrides", {}).get("algorithm", "") or "").strip()
        defaults = {
            "bits": 4,
            "group_size": 128,
            "mode": "compressed-tensors" if request.get("engine") == "vllm_cuda" else "affine",
            "algorithm": "awq",
        }
        capability_engines = self.capabilities().get("engines", [])
        selected_engine = next(
            (item for item in capability_engines if item.get("id") == request.get("engine")),
            {},
        )
        available_presets = [
            item
            for item in selected_engine.get("presets", [])
            if isinstance(item, dict) and str(item.get("id", "")).strip()
        ]
        if not available_presets:
            raise ValueError("The selected executor has no available quantization presets.")
        available_ids = {str(item.get("id")) for item in available_presets}
        if request.get("preset_id") not in available_ids:
            raise ValueError("The selected quantization preset is not advertised by the executor.")
        for item in available_presets:
            if item.get("id") == request.get("preset_id"):
                defaults.update({
                    key: item.get(key)
                    for key in ("bits", "group_size", "mode", "algorithm")
                    if item.get(key) is not None
                })
                break
        bits = bits or int(defaults["bits"])
        group_size = group_size or int(defaults["group_size"])
        mode = mode or str(defaults["mode"])
        advanced = selected_engine.get("advanced", {})
        if not isinstance(advanced, dict):
            advanced = {}

        def advertised_numbers(key: str) -> set[int]:
            values = advanced.get(key)
            if not isinstance(values, list):
                return set()
            result: set[int] = set()
            for value in values:
                with suppress(TypeError, ValueError):
                    result.add(int(value))
            return result

        advertised_bits = advertised_numbers("bits")
        advertised_group_sizes = advertised_numbers("group_sizes")
        if advertised_bits and bits not in advertised_bits:
            raise ValueError("The selected bit width is not advertised by the executor.")
        if advertised_group_sizes and group_size not in advertised_group_sizes:
            raise ValueError("The selected group size is not advertised by the executor.")

        advertised_modes = {
            str(value).strip()
            for value in advanced.get("modes", [])
            if str(value).strip()
        }
        if advertised_modes and mode not in advertised_modes:
            raise ValueError("The selected quantization mode is not advertised by the executor.")
        algorithm = algorithm or str(defaults.get("algorithm", "awq"))
        if request.get("engine") == "vllm_cuda":
            advertised_algorithms = {
                str(value).strip()
                for value in advanced.get("algorithms", [])
                if str(value).strip()
            }
            if not advertised_algorithms:
                raise ValueError("The Worker did not advertise any quantization algorithms.")
            if algorithm not in advertised_algorithms:
                raise ValueError("The selected quantization algorithm is not advertised by the Worker.")
        return {"bits": bits, "group_size": group_size, "mode": mode, "algorithm": algorithm}

    def preflight(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            request = self._normalize_request(payload)
        except ValueError as exc:
            return {"ok": False, "errors": [str(exc)], "warnings": [], "estimate": {}}
        errors: list[str] = []
        warnings: list[str] = []
        try:
            preset = self._preset(request)
        except ValueError as exc:
            # Keep the source/Worker diagnostics below visible even when the
            # executor has not advertised presets yet (for example while a
            # Worker is offline).  A conservative estimate is still useful
            # for the preflight rail, while the error keeps Start disabled.
            errors.append(str(exc))
            preset = {
                "bits": 4,
                "group_size": 128,
                "mode": "compressed-tensors" if request["engine"] == "vllm_cuda" else "affine",
                "algorithm": "awq",
            }
        source = request["source"]
        if request["engine"] == "vllm_cuda":
            if source["kind"] != "huggingface":
                errors.append("CUDA Worker accepts Hugging Face Hub sources only.")
            with self._lock:
                worker = self._worker_configs.get(request["worker_id"])
            if not worker:
                errors.append("Select a configured CUDA Worker.")
            elif not str(worker.get("url", "")).startswith("https://"):
                errors.append("Worker URL must use HTTPS.")
            elif not _KeychainSecretStore.get(request["worker_id"]):
                errors.append("Worker access token is not available in Keychain.")
        else:
            if request["source"]["kind"] == "local_path":
                source_path = Path(request["source_path"])
                if not source_path.is_dir():
                    errors.append("Local source must be an existing folder.")
                elif not os.access(source_path, os.R_OK):
                    errors.append("Local source folder is not readable.")
            with self._lock:
                if self._local_active_id:
                    errors.append("Another local MLX quantization job is already running.")
        output_path = self.artifact_root / request["output_name"]
        if output_path.exists():
            errors.append("Output name already exists in the managed quantized model directory.")
        estimated_input = 0
        if request["source"]["kind"] == "local_path":
            source_path = Path(request["source_path"])
            if source_path.is_dir():
                for child in source_path.rglob("*"):
                    if child.is_file():
                        with suppress(OSError):
                            estimated_input += child.stat().st_size
        estimated_output = max(256 * 1024 * 1024, int(estimated_input * (preset["bits"] / 16.0))) if estimated_input else 1024 * 1024 * 1024
        try:
            free = shutil.disk_usage(self.artifact_root.parent).free
        except OSError:
            free = 0
        if request["engine"] == "mlx_local" and free and free < estimated_output:
            errors.append("Not enough free disk space for the estimated output.")
        if request["source"]["kind"] == "huggingface" and not request["source"]["revision"]:
            warnings.append("The default Hub revision will be used.")
        return {
            "ok": not errors,
            "errors": errors,
            "warnings": warnings,
            "selected_profile": {"engine": request["engine"], "preset_id": request["preset_id"], **preset},
            "estimate": {
                "estimated_input_bytes": estimated_input,
                "estimated_output_bytes": estimated_output,
                "free_bytes": free,
                "compression_ratio": round(16 / preset["bits"], 2),
                "output_path": str(output_path),
            },
        }

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = self._normalize_request(payload)
        check = self.preflight(payload)
        if not check.get("ok"):
            raise ValueError(" ".join(str(item) for item in check.get("errors", [])) or "Quantization preflight failed.")
        job_id = uuid.uuid4().hex
        job = {
            "id": job_id,
            "engine": request["engine"],
            "source": deepcopy(request["source"]),
            "worker_id": request["worker_id"],
            "preset_id": request["preset_id"],
            "overrides": deepcopy(request["overrides"]),
            "output_name": request["output_name"],
            "status": "queued",
            "stage": "queued",
            "progress": 0.0,
            "speed": "",
            "eta_seconds": None,
            "message": "Queued for quantization.",
            "logs": [],
            "artifact": None,
            "delivery_options": ["register", "reveal"] if request["engine"] == "mlx_local" else ["retain", "download"],
            "allowed_actions": ["cancel"] if request["engine"] == "mlx_local" else ["cancel"],
            "error": "",
            "created_at": _iso(),
            "updated_at": _iso(),
        }
        cancel_event = threading.Event()
        with self._lock:
            if request["engine"] == "mlx_local" and self._local_active_id:
                active = self._jobs.get(self._local_active_id, {})
                if active.get("status") in _LOCAL_RUNNING_STATUSES:
                    raise ValueError("Another local MLX quantization job is already running.")
            if (self.artifact_root / request["output_name"]).exists():
                raise FileExistsError("Output name already exists in the managed directory.")
            self._jobs[job_id] = job
            self._cancel_events[job_id] = cancel_event
            if request["engine"] == "mlx_local":
                self._local_active_id = job_id
            self._persist_locked()
        target = self._run_local if request["engine"] == "mlx_local" else self._run_remote
        thread = threading.Thread(target=target, args=(job_id, request), daemon=True, name=f"quantization-{job_id[:8]}")
        thread.start()
        return self.job(job_id) or job

    def job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._jobs.get(str(job_id or ""))
            return deepcopy(value) if value else None

    def jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [deepcopy(item) for item in sorted(self._jobs.values(), key=lambda x: str(x.get("created_at", "")), reverse=True)]

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for key, value in changes.items():
                job[key] = deepcopy(value)
            job["updated_at"] = _iso()
            self._persist_locked()

    def _log(self, job_id: str, message: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            logs = list(job.get("logs", []))
            logs.append(str(_redact(message)).strip()[:1000])
            job["logs"] = logs[-120:]
            job["updated_at"] = _iso()
            self._persist_locked()

    def action(self, job_id: str, action: str) -> dict[str, Any]:
        action = str(action or "").strip().lower()
        if action not in {"cancel", "pause", "resume"}:
            raise ValueError("Unsupported quantization action.")
        with self._lock:
            job = self._jobs.get(str(job_id or ""))
            if not job:
                raise KeyError("Quantization job not found.")
            if job.get("status") in _TERMINAL_STATUSES | {"awaiting_delivery"}:
                return deepcopy(job)
            engine = job.get("engine")
            if action == "pause" and engine == "mlx_local":
                raise ValueError("Local MLX quantization cannot be paused; cancel the job instead.")
            event = self._cancel_events.setdefault(str(job_id), threading.Event())
            if action == "cancel":
                event.set()
                job["status"] = "cancelling"
                job["stage"] = "cancelling"
                job["message"] = "Cancellation requested."
            elif action == "pause":
                job["status"] = "pausing"
                job["stage"] = "pausing"
            else:
                job["status"] = "running"
                job["stage"] = "running"
            self._persist_locked()
            worker_id = str(job.get("worker_id", ""))
            remote_job_id = str(job.get("remote_job_id", ""))
        if engine == "vllm_cuda" and worker_id and remote_job_id:
            try:
                self._worker_post(worker_id, f"/v1/quantization/jobs/{remote_job_id}/action", {"action": action})
            except Exception as exc:
                self._log(str(job_id), f"Worker action failed: {_redact(exc)}")
        return self.job(str(job_id)) or {}

    def deliver(self, job_id: str, action: str) -> dict[str, Any]:
        action = str(action or "").strip().lower()
        with self._lock:
            job = self._jobs.get(str(job_id or ""))
            if not job:
                raise KeyError("Quantization job not found.")
            if job.get("status") not in {"awaiting_delivery", "completed"}:
                raise ValueError("Job is not ready for delivery.")
            engine = job.get("engine")
        if engine == "mlx_local" and action not in {"register", "reveal"}:
            raise ValueError("MLX delivery supports register or reveal.")
        if engine == "vllm_cuda" and action not in {"retain", "download"}:
            raise ValueError("CUDA delivery supports retain or download.")
        if action == "register":
            artifact = (self.job(job_id) or {}).get("artifact") or {}
            path = Path(str(artifact.get("path", ""))).resolve()
            if not self._is_owned_artifact(path):
                raise ValueError("Quantized artifact is outside the managed directory.")
            with self._lock:
                registry_path = self.artifact_root / "registry.json"
                try:
                    registry = json.loads(registry_path.read_text(encoding="utf-8"))
                except (OSError, ValueError, TypeError):
                    registry = {}
                if not isinstance(registry, dict):
                    registry = {}
                model_id = f"quantized/{(self.job(job_id) or {}).get('output_name', path.name)}"
                registry[model_id] = {"id": model_id, "path": str(path), "engine": "mlx_local", "created_at": _iso()}
                self._write_json_atomic(registry_path, registry)
            self._update(job_id, status="completed", stage="registered", delivery="register", managed_model_id=model_id, message="Registered in the managed Token Workshed model directory.")
        elif action == "reveal":
            artifact = (self.job(job_id) or {}).get("artifact") or {}
            path = Path(str(artifact.get("path", ""))).resolve()
            if not self._is_owned_artifact(path):
                raise ValueError("Quantized artifact is outside the managed directory.")
            if sys.platform == "darwin":
                with suppress(OSError):
                    subprocess.Popen(
                        ["/usr/bin/open", "-R", str(path)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
            self._update(job_id, status="completed", stage="ready", delivery="reveal", message="Artifact is ready to reveal in Finder.")
        elif action == "retain":
            self._update(job_id, status="completed", stage="retained", delivery="retain", message="Artifact retained on the CUDA Worker.")
        elif action == "download":
            self._download_remote_artifact(job_id)
            self._update(job_id, status="completed", stage="downloaded", delivery="download", message="Artifact downloaded to the managed Token Workshed model directory.")
        return self.job(job_id) or {}

    def managed_model_ids(self) -> list[str]:
        registry_path = self.artifact_root / "registry.json"
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return []
        if not isinstance(registry, dict):
            return []
        return sorted(str(key) for key, item in registry.items() if isinstance(item, dict) and self._is_owned_artifact(Path(str(item.get("path", "")))))

    def resolve_model(self, model_id: str) -> str | None:
        target = str(model_id or "").strip()
        if not target:
            return None
        registry_path = self.artifact_root / "registry.json"
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        item = registry.get(target) if isinstance(registry, dict) else None
        path = Path(str(item.get("path", ""))).resolve() if isinstance(item, dict) else Path()
        return str(path) if item and self._is_owned_artifact(path) else None

    def _is_owned_artifact(self, path: Path) -> bool:
        try:
            path.relative_to(self.artifact_root.resolve())
        except ValueError:
            return False
        return path.exists()

    # ------------------------------------------------------------------
    # Local MLX runner
    # ------------------------------------------------------------------
    def _run_local(self, job_id: str, request: dict[str, Any]) -> None:
        temp_dir: Path | None = None
        try:
            source = request["source"]
            source_ref = str(source.get("ref", ""))
            if source.get("kind") == "local_path":
                source_path = Path(request["source_path"]).resolve()
                if not source_path.is_dir():
                    raise ValueError("Local source folder is unavailable.")
                source_ref = str(source_path)
            self.temp_root.mkdir(parents=True, exist_ok=True)
            temp_dir = Path(tempfile.mkdtemp(prefix=f"{job_id}-", dir=str(self.temp_root)))
            self._update(job_id, status="running", stage="converting", progress=0.05, message="MLX-LM conversion started.")
            preset = self._preset(request)
            command = [
                self.python_executable,
                "-m",
                "mlx_lm.convert",
                "--hf-path",
                source_ref,
                "--mlx-path",
                str(temp_dir),
                "-q",
                "--q-bits",
                str(preset["bits"]),
                "--q-group-size",
                str(preset["group_size"]),
                "--q-mode",
                str(preset["mode"]),
            ]
            revision = str(source.get("revision", "") or "").strip()
            if revision:
                command.extend(["--revision", revision])
            self._log(job_id, "Running MLX-LM conversion (token omitted from logs).")
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
            with self._lock:
                self._processes[job_id] = process
            stream = process.stdout
            while stream is not None:
                if self._cancel_events.get(job_id, threading.Event()).is_set():
                    process.terminate()
                    break
                line = stream.readline()
                if not line:
                    break
                self._log(job_id, line.rstrip())
                with self._lock:
                    progress = min(0.92, float(self._jobs.get(job_id, {}).get("progress", 0.05)) + 0.01)
                self._update(job_id, progress=progress, stage="converting")
            return_code = process.wait(timeout=10)
            with self._lock:
                self._processes.pop(job_id, None)
            if self._cancel_events.get(job_id, threading.Event()).is_set():
                self._update(job_id, status="cancelled", stage="cancelled", progress=0.0, message="Local MLX quantization cancelled.")
                return
            if return_code != 0:
                raise RuntimeError(f"MLX-LM converter exited with code {return_code}.")
            artifact = self._promote_local_artifact(job_id, temp_dir, request)
            self._update(job_id, status="awaiting_delivery", stage="validate", progress=1.0, artifact=artifact, message="Quantization finished. Choose how to deliver the artifact.")
        except Exception as exc:
            self._update(job_id, status="failed", stage="failed", error=str(_redact(exc)), message="Quantization failed.")
        finally:
            with self._lock:
                self._processes.pop(job_id, None)
                if self._local_active_id == job_id:
                    self._local_active_id = ""
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)
            self._persist_locked()

    def _promote_local_artifact(self, job_id: str, temp_dir: Path, request: dict[str, Any]) -> dict[str, Any]:
        required = [temp_dir / "config.json"]
        weight_files = [item for item in temp_dir.rglob("*") if item.is_file() and item.suffix.lower() in {".safetensors", ".npz", ".bin"}]
        if not required[0].is_file() or not weight_files:
            raise RuntimeError("Converter completed without a config.json and weight file.")
        final = self.artifact_root / request["output_name"]
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        if final.exists():
            raise FileExistsError("Output name already exists in the managed directory.")
        os.replace(temp_dir, final)
        digest = hashlib.sha256()
        total = 0
        for item in final.rglob("*"):
            if not item.is_file():
                continue
            total += item.stat().st_size
            with item.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        return {"path": str(final), "name": request["output_name"], "bytes": total, "sha256": digest.hexdigest(), "engine": "mlx_local"}

    # ------------------------------------------------------------------
    # Remote Worker runner
    # ------------------------------------------------------------------
    def _worker_request(self, worker_id: str, method: str, path: str, payload: dict[str, Any] | None = None, *, stream: bool = False) -> requests.Response:
        with self._lock:
            worker = dict(self._worker_configs.get(worker_id, {}))
        if not worker:
            raise RuntimeError("CUDA Worker is not configured.")
        url = str(worker.get("url", "")).rstrip("/") + "/" + str(path).lstrip("/")
        if not url.startswith("https://"):
            raise RuntimeError("Worker URL must use HTTPS.")
        token = _KeychainSecretStore.get(worker_id)
        if not token:
            raise RuntimeError("CUDA Worker token is unavailable in Keychain.")
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        response = requests.request(method, url, json=payload, headers=headers, timeout=self.request_timeout_s, stream=stream)
        if response.status_code >= 400:
            try:
                detail: Any = response.json()
            except ValueError:
                detail = response.text[:500]
            raise RuntimeError(str(_redact(detail)))
        return response

    def _worker_post(self, worker_id: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._worker_request(worker_id, "POST", path, payload)
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("Worker returned an invalid JSON object.")
        return data

    def _run_remote(self, job_id: str, request: dict[str, Any]) -> None:
        try:
            worker_id = request["worker_id"]
            remote_id = str(request.get("remote_job_id", "") or "").strip()
            if not remote_id:
                data = self._worker_post(worker_id, "/v1/quantization/jobs", {key: request[key] for key in ("source", "preset_id", "overrides", "output_name")})
                remote_id = str(data.get("id", data.get("job_id", ""))).strip()
            if not remote_id:
                raise RuntimeError("Worker did not return a job id.")
            allowed_actions = ["cancel"]
            with suppress(Exception):
                capability_response = self._worker_request(
                    worker_id,
                    "GET",
                    "/v1/quantization/capabilities",
                ).json()
                if isinstance(capability_response, dict) and bool(capability_response.get("pause")):
                    allowed_actions.append("pause")
            self._update(job_id, status="running", stage="submitted", progress=0.02, remote_job_id=remote_id, allowed_actions=allowed_actions, message="Submitted to CUDA Worker.")
            while True:
                if self._cancel_events.get(job_id, threading.Event()).is_set():
                    try:
                        self._worker_post(worker_id, f"/v1/quantization/jobs/{remote_id}/action", {"action": "cancel"})
                    except Exception as exc:
                        self._log(job_id, f"Worker cancellation failed: {_redact(exc)}")
                    self._update(job_id, status="cancelled", stage="cancelled", message="CUDA Worker job cancelled.")
                    return
                status = self._worker_request(worker_id, "GET", f"/v1/quantization/jobs/{remote_id}").json()
                if not isinstance(status, dict):
                    raise RuntimeError("Worker returned an invalid job object.")
                remote_status = str(status.get("status", "running"))
                progress = float(status.get("progress", 0.0) or 0.0)
                changes = {"status": "awaiting_delivery" if remote_status in {"completed", "awaiting_delivery"} else remote_status, "stage": str(status.get("stage", remote_status)), "progress": max(0.0, min(1.0, progress)), "eta_seconds": status.get("eta_seconds"), "message": str(status.get("message", "")), "remote_status": remote_status}
                if status.get("artifact"):
                    changes["artifact"] = _redact(status.get("artifact"))
                if status.get("deployment"):
                    changes["deployment"] = _redact(status.get("deployment"))
                self._update(job_id, **changes)
                if remote_status in {"completed", "awaiting_delivery"}:
                    return
                if remote_status in {"failed", "cancelled"}:
                    self._update(job_id, status=remote_status, stage=remote_status, error=str(_redact(status.get("error", "Worker job failed."))))
                    return
                time.sleep(1.5)
        except Exception as exc:
            self._update(job_id, status="failed", stage="failed", error=str(_redact(exc)), message="CUDA Worker job failed.")

    def _download_remote_artifact(self, job_id: str) -> None:
        job = self.job(job_id) or {}
        worker_id = str(job.get("worker_id", ""))
        remote_id = str(job.get("remote_job_id", ""))
        if not worker_id or not remote_id:
            raise ValueError("Remote job identity is missing.")
        response = self._worker_request(worker_id, "GET", f"/v1/quantization/jobs/{remote_id}/artifact", stream=True)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        destination = self.artifact_root / str(job.get("output_name", f"quantized-{job_id}"))
        if destination.exists():
            raise FileExistsError("Output name already exists in the managed directory.")
        self.temp_root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f"{job_id}-download-", dir=str(self.temp_root))) / "artifact.zip"
        digest = hashlib.sha256()
        total = 0
        expected_digest = str((job.get("artifact") or {}).get("sha256", "") or "").strip().lower()
        try:
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
            if expected_digest and digest.hexdigest().lower() != expected_digest:
                raise RuntimeError("Worker artifact checksum does not match its metadata.")
            if not zipfile.is_zipfile(temporary):
                raise RuntimeError("Worker artifact is not a ZIP model package.")
            extracted = Path(tempfile.mkdtemp(prefix=f"{job_id}-extract-", dir=str(self.temp_root)))
            try:
                with zipfile.ZipFile(temporary) as archive:
                    for member in archive.infolist():
                        target = (extracted / member.filename).resolve()
                        try:
                            target.relative_to(extracted.resolve())
                        except ValueError as exc:
                            raise RuntimeError("Worker artifact contains an unsafe path.") from exc
                    archive.extractall(extracted)
                os.replace(extracted, destination)
            finally:
                shutil.rmtree(extracted, ignore_errors=True)
        finally:
            shutil.rmtree(temporary.parent, ignore_errors=True)
        self._update(job_id, artifact={"path": str(destination), "name": destination.name, "bytes": total, "sha256": digest.hexdigest(), "engine": "vllm_cuda", "archive": "zip"})

    def shutdown(self) -> None:
        with self._lock:
            for event in self._cancel_events.values():
                event.set()
            processes = list(self._processes.values())
        for process in processes:
            with suppress(OSError):
                process.terminate()
        with self._lock:
            self._persist_locked()


__all__ = [
    "QuantizationCapabilities",
    "QuantizationDelivery",
    "QuantizationJob",
    "QuantizationRequest",
    "QuantizationService",
]
