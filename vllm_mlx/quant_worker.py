# SPDX-License-Identifier: Apache-2.0
"""Standalone HTTPS-ready CUDA quantization Worker contract.

The desktop manager proxies this API and never sends a Hugging Face token.
Deploy this module behind TLS (or a TLS reverse proxy) with
``TOKEN_WORKSHED_QUANT_WORKER_TOKEN``.  A production deployment can set
``TOKEN_WORKSHED_QUANT_WORKER_COMMAND`` to an LLM Compressor wrapper; the
wrapper receives a request JSON file and writes the artifact into the
workspace supplied by the Worker.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shlex
import shutil
import subprocess
import threading
import time
import uuid
import zipfile
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field


def _iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class WorkerSource(BaseModel):
    kind: str = "huggingface"
    ref: str = Field(min_length=1, max_length=4096)
    revision: str = Field(default="", max_length=200)


class WorkerJobRequest(BaseModel):
    source: WorkerSource
    preset_id: str = Field(default="cuda-balanced", max_length=96)
    overrides: dict[str, Any] = Field(default_factory=dict)
    output_name: str = Field(default="quantized-model", max_length=120)


class WorkerActionRequest(BaseModel):
    action: str


def _safe_name(value: str) -> str:
    result = "".join(char if char.isalnum() or char in "._-" else "-" for char in str(value or ""))
    return result.strip(".-_")[:120] or "quantized-model"


class _WorkerJobs:
    def __init__(self, root: Path, runner: Callable[[dict[str, Any], Path, threading.Event], None] | None = None) -> None:
        self.root = root.resolve()
        self.workspace = self.root / "workspaces"
        self.artifacts = self.root / "artifacts"
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.runner = runner
        self.lock = threading.RLock()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.cancel: dict[str, threading.Event] = {}
        self.processes: dict[str, subprocess.Popen[str]] = {}

    def create(self, payload: WorkerJobRequest) -> dict[str, Any]:
        if payload.source.kind != "huggingface":
            raise ValueError("CUDA Worker accepts Hugging Face Hub sources only.")
        job_id = uuid.uuid4().hex
        workspace = self.workspace / job_id
        job = {
            "id": job_id,
            "status": "queued",
            "stage": "queued",
            "progress": 0.0,
            "eta_seconds": None,
            "message": "Queued on CUDA Worker.",
            "error": "",
            "source": payload.source.model_dump(),
            "preset_id": payload.preset_id,
            "overrides": payload.overrides,
            "output_name": _safe_name(payload.output_name),
            "artifact": None,
            "deployment": None,
            "created_at": _iso(),
            "updated_at": _iso(),
        }
        with self.lock:
            self.jobs[job_id] = job
            self.cancel[job_id] = threading.Event()
        threading.Thread(target=self._run, args=(job_id, workspace), daemon=True, name=f"quant-worker-{job_id[:8]}").start()
        return dict(job)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self.lock:
            job = self.jobs.get(job_id)
            return dict(job) if job else None

    def update(self, job_id: str, **changes: Any) -> None:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return
            job.update(changes)
            job["updated_at"] = _iso()

    def _run(self, job_id: str, workspace: Path) -> None:
        try:
            workspace.mkdir(parents=True, exist_ok=True)
            self.update(job_id, status="running", stage="compressing", progress=0.05, message="LLM Compressor job started.")
            job = self.get(job_id) or {}
            if self.runner is not None:
                self.runner(job, workspace, self.cancel[job_id])
            else:
                self._run_configured_command(job, workspace, self.cancel[job_id])
            if self.cancel[job_id].is_set():
                self.update(job_id, status="cancelled", stage="cancelled", progress=0.0, message="Worker job cancelled.")
                return
            request_file = workspace / "request.json"
            # The request envelope is only for the configured runner; it is
            # not a model artifact and should never be the sole "success"
            # signal or get retained with the deployable weights.
            with suppress(FileNotFoundError):
                request_file.unlink()
            files = [path for path in workspace.rglob("*") if path.is_file()]
            if not files:
                raise RuntimeError("LLM Compressor runner produced no artifact files.")
            archive = self.artifacts / f"{job_id}-{_safe_name(str(job.get('output_name', 'quantized-model')))}.zip"
            self._zip_workspace(workspace, archive)
            retained = self.artifacts / job_id
            os.replace(workspace, retained)
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            bytes_total = archive.stat().st_size
            deployment = {
                "command": f"vllm serve {shlex.quote(str(retained))} --quantization compressed-tensors",
                "artifact": archive.name,
                "path": str(retained),
                "engine": "vllm",
                "quantization": "compressed-tensors",
            }
            self.update(job_id, status="completed", stage="ready", progress=1.0, message="Artifact ready for delivery.", artifact={"name": archive.name, "bytes": bytes_total, "sha256": digest}, deployment=deployment)
        except Exception as exc:
            self.update(job_id, status="failed", stage="failed", error=str(exc)[:1000], message="Worker job failed.")
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def _run_configured_command(self, job: dict[str, Any], workspace: Path, cancel: threading.Event) -> None:
        configured = str(os.environ.get("TOKEN_WORKSHED_QUANT_WORKER_COMMAND", "")).strip()
        if not configured:
            raise RuntimeError("LLM Compressor runner is not configured; set TOKEN_WORKSHED_QUANT_WORKER_COMMAND.")
        request_file = workspace / "request.json"
        request_file.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
        command = shlex.split(configured) + ["--request", str(request_file), "--output", str(workspace)]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        with self.lock:
            self.processes[job["id"]] = process
        try:
            while True:
                if cancel.is_set():
                    process.terminate()
                    break
                line = process.stdout.readline() if process.stdout is not None else ""
                if not line:
                    if process.poll() is not None:
                        break
                    time.sleep(0.1)
                    continue
                self.update(job["id"], message=line.rstrip()[:1000], progress=min(0.95, float(self.get(job["id"]).get("progress", 0.05)) + 0.01))
            code = process.wait(timeout=10)
            if code != 0 and not cancel.is_set():
                raise RuntimeError(f"LLM Compressor runner exited with code {code}.")
        finally:
            with self.lock:
                self.processes.pop(job["id"], None)

    def _zip_workspace(self, workspace: Path, archive: Path) -> None:
        temporary = archive.with_suffix(archive.suffix + ".tmp")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            for path in workspace.rglob("*"):
                if path.is_file() and path.name != "request.json":
                    handle.write(path, path.relative_to(workspace))
        os.replace(temporary, archive)

    def action(self, job_id: str, action: str) -> dict[str, Any]:
        action = str(action or "").strip().lower()
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                raise KeyError("Worker job not found.")
            if job.get("status") in {"completed", "failed", "cancelled"}:
                raise ValueError("Worker job is not active.")
            if action == "cancel":
                self.cancel[job_id].set()
                job["status"] = "cancelling"
                job["stage"] = "cancelling"
            elif action == "pause":
                # Pause support is intentionally capability-gated.  This first
                # Worker exposes cancel only because LLM Compressor subprocess
                # jobs cannot be safely suspended across GPU contexts.
                raise ValueError("This Worker does not support pause.")
            elif action == "resume":
                raise ValueError("This Worker does not support pause.")
            else:
                raise ValueError("Unsupported Worker action.")
            job["updated_at"] = _iso()
            return dict(job)

    def artifact_path(self, job_id: str) -> Path | None:
        job = self.get(job_id) or {}
        artifact = job.get("artifact") or {}
        name = str(artifact.get("name", "")).strip()
        path = (self.artifacts / name).resolve() if name else Path()
        try:
            path.relative_to(self.artifacts.resolve())
        except ValueError:
            return None
        return path if path.is_file() else None


def create_quant_worker_app(*, root: str | Path | None = None, token: str | None = None, runner: Callable[[dict[str, Any], Path, threading.Event], None] | None = None) -> FastAPI:
    root_path = Path(root or os.environ.get("TOKEN_WORKSHED_QUANT_WORKER_ROOT", "quant-worker-data")).expanduser().resolve()
    expected_token = str(token or os.environ.get("TOKEN_WORKSHED_QUANT_WORKER_TOKEN", "")).strip()
    jobs = _WorkerJobs(root_path, runner=runner)
    app = FastAPI(title="Token Workshed CUDA Quant Worker", version="0.2.0")
    app.state.jobs = jobs

    def authenticate(authorization: str | None) -> None:
        if not expected_token:
            raise HTTPException(status_code=503, detail="Worker token is not configured.")
        scheme, _, value = str(authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(value.strip(), expected_token):
            raise HTTPException(status_code=401, detail="Invalid Worker authorization.")

    @app.get("/v1/quantization/capabilities")
    def capabilities(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        authenticate(authorization)
        return {"ok": True, "engine": "vllm_cuda", "worker": {"name": os.environ.get("TOKEN_WORKSHED_QUANT_WORKER_NAME", "CUDA Quant Worker"), "gpu": os.environ.get("TOKEN_WORKSHED_QUANT_WORKER_GPU", "unknown"), "memory": os.environ.get("TOKEN_WORKSHED_QUANT_WORKER_MEMORY", "unknown")}, "presets": [{"id": "cuda-fast", "label": "Fast", "algorithm": "smoothquant", "bits": 8}, {"id": "cuda-balanced", "label": "Balanced", "algorithm": "awq", "bits": 4}, {"id": "cuda-quality", "label": "Quality", "algorithm": "gptq", "bits": 4}], "advanced": {"algorithms": ["awq", "gptq", "smoothquant"], "bits": [4, 8], "group_sizes": [32, 64, 128, 256], "modes": ["compressed-tensors"]}, "pause": False}

    @app.post("/v1/quantization/jobs")
    def create_job(payload: WorkerJobRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        authenticate(authorization)
        try:
            return {"ok": True, **jobs.create(payload)}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/quantization/jobs/{job_id}")
    def get_job(job_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        authenticate(authorization)
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Worker job not found.")
        return {"ok": True, **job}

    @app.post("/v1/quantization/jobs/{job_id}/action")
    def action(job_id: str, payload: WorkerActionRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        authenticate(authorization)
        try:
            return {"ok": True, **jobs.action(job_id, payload.action)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/quantization/jobs/{job_id}/artifact")
    def artifact(job_id: str, authorization: str | None = Header(default=None)) -> FileResponse:
        authenticate(authorization)
        path = jobs.artifact_path(job_id)
        if path is None:
            raise HTTPException(status_code=404, detail="Worker artifact is not ready.")
        return FileResponse(path, media_type="application/zip", filename=path.name)

    @app.get("/healthz")
    def healthz() -> JSONResponse:
        return JSONResponse({"ok": True, "service": "token-workshed-quant-worker"})

    return app


def main() -> None:
    import uvicorn

    host = os.environ.get("TOKEN_WORKSHED_QUANT_WORKER_HOST", "0.0.0.0")
    port = int(os.environ.get("TOKEN_WORKSHED_QUANT_WORKER_PORT", "8787"))
    uvicorn.run(create_quant_worker_app(), host=host, port=port)


__all__ = ["create_quant_worker_app", "main"]
