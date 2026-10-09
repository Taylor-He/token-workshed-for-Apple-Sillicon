# SPDX-License-Identifier: Apache-2.0
"""Contract and safety tests for the 0.2.0 quantization workbench."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vllm_mlx.quant_worker import create_quant_worker_app
from vllm_mlx.quantization import QuantizationService


def _service(tmp_path: Path) -> QuantizationService:
    return QuantizationService(
        artifact_root=tmp_path / "managed",
        state_path=tmp_path / "jobs.json",
        worker_state_path=tmp_path / "workers.json",
        request_timeout_s=3,
    )


def test_capabilities_and_preflight_keep_local_and_remote_boundaries(tmp_path: Path) -> None:
    service = _service(tmp_path)
    capabilities = service.capabilities()
    assert capabilities["ok"] is True
    assert {item["id"] for item in capabilities["engines"]} == {"mlx_local"}

    local = tmp_path / "source"
    local.mkdir()
    (local / "config.json").write_text("{}", encoding="utf-8")
    preflight = service.preflight(
        {
            "source": {"kind": "local_path", "ref": str(local)},
            "engine": "mlx_local",
            "preset_id": "mlx-balanced",
            "overrides": {},
            "output_name": "demo-4bit",
        }
    )
    assert preflight["ok"] is True
    assert preflight["estimate"]["compression_ratio"] == 4.0

    remote = service.preflight(
        {
            "source": {"kind": "local_path", "ref": str(local)},
            "engine": "vllm_cuda",
            "worker_id": "worker",
            "preset_id": "cuda-balanced",
            "overrides": {},
            "output_name": "remote",
        }
    )
    assert remote["ok"] is False
    assert any("Hub" in error for error in remote["errors"])
    invalid_name = service.preflight(
        {
            "source": {"kind": "huggingface", "ref": "org/model"},
            "engine": "mlx_local",
            "preset_id": "mlx-balanced",
            "output_name": "../escape",
        }
    )
    assert invalid_name["ok"] is False
    assert any("Output name" in error for error in invalid_name["errors"])


def test_worker_configuration_requires_https_and_does_not_persist_token(tmp_path: Path) -> None:
    service = _service(tmp_path)
    with pytest.raises(ValueError, match="HTTPS"):
        service.configure_worker("worker", "GPU", "http://worker.local", "secret")
    service.configure_worker("worker", "GPU", "https://worker.local/api", "secret")
    persisted = json.loads((tmp_path / "workers.json").read_text(encoding="utf-8"))
    assert persisted["worker"]["url"] == "https://worker.local/api"
    assert "secret" not in (tmp_path / "workers.json").read_text(encoding="utf-8")
    remote = service.preflight(
        {
            "source": {"kind": "huggingface", "ref": "org/model"},
            "engine": "vllm_cuda",
            "worker_id": "worker",
            "preset_id": "cuda-balanced",
            "overrides": {},
            "output_name": "remote",
        }
    )
    # A configured Worker is not considered ready until its capability
    # contract can be reached; the UI should keep Start disabled while it is
    # offline instead of silently accepting an unknown preset.
    assert remote["ok"] is False
    assert any("Worker" in error or "presets" in error for error in remote["errors"])


def test_managed_registry_is_path_bounded(tmp_path: Path) -> None:
    service = _service(tmp_path)
    artifact = service.artifact_root / "demo"
    artifact.mkdir(parents=True)
    (artifact / "config.json").write_text("{}", encoding="utf-8")
    registry = service.artifact_root / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "quantized/demo": {
                    "path": str(artifact),
                    "engine": "mlx_local",
                },
                "outside": {"path": str(tmp_path), "engine": "mlx_local"},
            }
        ),
        encoding="utf-8",
    )
    assert service.managed_model_ids() == ["quantized/demo"]
    assert service.resolve_model("quantized/demo") == str(artifact.resolve())
    assert service.resolve_model("outside") is None


def test_remote_download_extracts_inside_managed_directory(tmp_path: Path) -> None:
    service = _service(tmp_path)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("config.json", "{}")
        handle.writestr("model.safetensors", "weights")

    class Response:
        def iter_content(self, chunk_size: int):
            del chunk_size
            yield archive.getvalue()

    service._jobs["job"] = {
        "id": "job",
        "worker_id": "worker",
        "remote_job_id": "remote",
        "output_name": "remote-model",
        "status": "awaiting_delivery",
        "artifact": {"sha256": hashlib.sha256(archive.getvalue()).hexdigest()},
    }
    service._worker_request = lambda *args, **kwargs: Response()  # type: ignore[method-assign]
    service._download_remote_artifact("job")
    destination = service.artifact_root / "remote-model"
    assert (destination / "config.json").is_file()
    assert (destination / "model.safetensors").is_file()


def test_worker_contract_auth_pause_capability_and_artifact(tmp_path: Path) -> None:
    def runner(job: dict, workspace: Path, _cancel) -> None:
        (workspace / "model.safetensors").write_bytes(b"weights")
        (workspace / "config.json").write_text("{}", encoding="utf-8")

    app = create_quant_worker_app(root=tmp_path / "worker", token="worker-secret", runner=runner)
    with TestClient(app) as client:
        assert client.get("/v1/quantization/capabilities").status_code == 401
        headers = {"Authorization": "Bearer worker-secret"}
        capabilities = client.get("/v1/quantization/capabilities", headers=headers)
        assert capabilities.status_code == 200
        assert capabilities.json()["pause"] is False
        response = client.post(
            "/v1/quantization/jobs",
            headers=headers,
            json={
                "source": {"kind": "huggingface", "ref": "org/model"},
                "preset_id": "cuda-balanced",
                "output_name": "demo",
            },
        )
        assert response.status_code == 200
        job_id = response.json()["id"]
        for _ in range(20):
            status = client.get(f"/v1/quantization/jobs/{job_id}", headers=headers).json()
            if status["status"] == "completed":
                break
        else:
            pytest.fail(f"worker job did not complete: {status}")
        assert "compressed-tensors" in status["deployment"]["command"]
        assert Path(status["deployment"]["path"]).is_dir()
        artifact = client.get(f"/v1/quantization/jobs/{job_id}/artifact", headers=headers)
        assert artifact.status_code == 200
        assert artifact.content.startswith(b"PK")
        pause = client.post(
            f"/v1/quantization/jobs/{job_id}/action",
            headers=headers,
            json={"action": "pause"},
        )
        assert pause.status_code == 400
