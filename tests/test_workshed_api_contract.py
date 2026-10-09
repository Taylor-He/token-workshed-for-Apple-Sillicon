"""Contract coverage for schema-driven Workshed manager APIs."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx import desktop_ui
from vllm_mlx.desktop_ui import _register_workshed_contract_routes
from vllm_mlx.workshed import WorkshedService


class _Manager:
    _snapshot_has_required_files = desktop_ui.LocalModelServerManager._snapshot_has_required_files

    def discover_models(self) -> list[str]:
        return ["org/local-model", "quantized/managed", "workshed/managed"]

    def search_community_models(self, query: str, limit: int) -> list[dict[str, Any]]:
        assert limit >= 1
        return [
            {"id": "org/local-model", "source": "local", "local": True},
            {
                "id": "org/hub-model",
                "source": "community",
                "developer": "Org",
                "size_label": "3B",
                "quantization_label": "BF16",
                "downloads": 120,
                "likes": 7,
                "pipeline_tag": "text-generation",
            },
        ]


class _Quantization:
    def managed_model_ids(self) -> list[str]:
        return ["quantized/managed"]

    def resolve_model(self, _model_id: str) -> None:
        return None

    def capabilities(self) -> dict[str, Any]:
        return {
            "ok": True,
            "engines": [
                {
                    "id": "mlx_local",
                    "label": "MLX · This Mac",
                    "presets": [
                        {
                            "id": "mlx-balanced",
                            "label": "Balanced",
                            "bits": 4,
                            "group_size": 128,
                            "mode": "affine",
                        }
                    ],
                },
                {
                    "id": "vllm_cuda",
                    "label": "CUDA · Worker",
                    "worker_id": "worker-a",
                    "presets": [{"id": "cuda-quality", "label": "Quality", "bits": 8}],
                },
            ],
        }


class _Workshed:
    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path

    def managed_model_ids(self) -> list[str]:
        return ["workshed/managed"]

    def resolve_model(self, model_id: str) -> str | None:
        if model_id == "workshed/managed" and self.model_path is not None:
            return str(self.model_path)
        return None

    def capabilities(self) -> dict[str, Any]:
        return {
            "ok": True,
            "runtime": {"local_only": True},
            "blocks": [
                {
                    "type_id": "quantize_mlx",
                    "presets": [
                        {
                            "id": "mlx-balanced",
                            "label": "Balanced",
                            "params": {
                                "bits": 4,
                                "group_size": 128,
                                "mode": "affine",
                            },
                        }
                    ],
                }
            ],
        }

    def run_receipt(self, run_id: str) -> dict[str, Any]:
        if run_id == "missing":
            raise KeyError("Workshed run not found.")
        return {
            "schema_version": 2,
            "run_id": run_id,
            "status": "running",
            "finalized": False,
        }


def _client(
    *,
    manager: Any | None = None,
    quantization: Any | None = None,
    workshed: Any | None = None,
) -> TestClient:
    app = FastAPI()
    _register_workshed_contract_routes(
        app,
        manager=manager or _Manager(),
        quantization_service=quantization or _Quantization(),
        workshed_service=workshed or _Workshed(),
        manager_api_token="manager-token",
    )
    return TestClient(app)


def _headers() -> dict[str, str]:
    return {"x-token-workshed-manager-token": "manager-token"}


def test_options_contract_exposes_exactly_the_seven_allowlisted_providers() -> None:
    provider_schema = desktop_ui.WorkshedOptionsPayload.model_json_schema()["properties"][
        "provider"
    ]

    assert set(provider_schema["enum"]) == {
        "models.local",
        "models.managed",
        "models.hub",
        "datasets.hub",
        "quantization.presets",
        "lm_eval.tasks",
        "model.target_modules",
    }


def test_capability_catalog_only_references_registered_option_providers(
    tmp_path: Path,
) -> None:
    allowed = set(
        desktop_ui.WorkshedOptionsPayload.model_json_schema()["properties"]["provider"][
            "enum"
        ]
    )
    capabilities = WorkshedService(root=tmp_path).capabilities()
    referenced: set[str] = set()

    for block in capabilities["blocks"]:
        properties = block["param_schema"].get("properties", {})
        for field_schema in properties.values():
            x_ui = field_schema.get("x-ui", {})
            if x_ui.get("options_provider"):
                referenced.add(x_ui["options_provider"])
            provider_by = x_ui.get("options_provider_by", {})
            referenced.update(
                provider
                for key, provider in provider_by.items()
                if key != "field" and provider
            )

    assert referenced == allowed

    quantize = next(
        block for block in capabilities["blocks"] if block["type_id"] == "quantize_mlx"
    )
    preset_ui = quantize["param_schema"]["properties"]["preset_id"]["x-ui"]
    assert preset_ui["options_provider"] == "quantization.presets"
    assert preset_ui["options_context"] == {"engine": "mlx_local"}


def test_options_require_manager_auth_and_strict_provider_allowlist() -> None:
    with _client() as client:
        unauthorized = client.post(
            "/api/manager/workshed/options",
            json={"provider": "models.local"},
        )
        assert unauthorized.status_code == 401

        unsupported = client.post(
            "/api/manager/workshed/options",
            headers=_headers(),
            json={
                "provider": "packages.install",
                "context": {"url": "https://example.test"},
            },
        )
        assert unsupported.status_code == 422

        extra = client.post(
            "/api/manager/workshed/options",
            headers=_headers(),
            json={"provider": "models.local", "package": "arbitrary"},
        )
        assert extra.status_code == 422


def test_local_and_managed_model_options_are_separate(monkeypatch, tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"model")
    monkeypatch.setattr(
        "huggingface_hub.scan_cache_dir",
        lambda: types.SimpleNamespace(repos=[types.SimpleNamespace(
            repo_id="org/local-model", repo_type="model", revisions=[types.SimpleNamespace(
                snapshot_path=snapshot, commit_hash="a" * 40, refs=["main"], last_modified=1,
            )],
        )]),
    )
    with _client() as client:
        local = client.post(
            "/api/manager/workshed/options",
            headers=_headers(),
            json={
                "provider": "models.local",
                "query": "local",
                "limit": 24,
                "context": {},
            },
        )
        assert local.status_code == 200
        assert local.json()["items"] == [
            {
                "id": str(snapshot),
                "label": "org/local-model",
                "description": "Local snapshot · aaaaaaaa",
                "metadata": {
                    "source": "local", "ref": str(snapshot), "hub_id": "org/local-model", "commit_hash": "a" * 40,
                },
            }
        ]
        assert local.json()["next_cursor"] == ""

        managed = client.post(
            "/api/manager/workshed/options",
            headers=_headers(),
            json={"provider": "models.managed", "query": "managed"},
        )
        assert managed.status_code == 200
        assert {item["id"] for item in managed.json()["items"]} == {
            "quantized/managed",
            "workshed/managed",
        }


def test_hub_models_reuse_manager_search_without_returning_local_rows() -> None:
    with _client() as client:
        response = client.post(
            "/api/manager/workshed/options",
            headers=_headers(),
            json={"provider": "models.hub", "query": "hub"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert [item["id"] for item in payload["items"]] == ["org/hub-model"]
        assert payload["items"][0]["metadata"]["source"] == "hub"


def test_hub_offline_is_a_clear_empty_result_instead_of_500() -> None:
    manager = _Manager()

    def offline(_query: str, _limit: int) -> list[dict[str, Any]]:
        raise OSError("offline")

    manager.search_community_models = offline  # type: ignore[method-assign]
    with _client(manager=manager) as client:
        response = client.post(
            "/api/manager/workshed/options",
            headers=_headers(),
            json={"provider": "models.hub"},
        )
        assert response.status_code == 200
        assert response.json()["items"] == []
        assert "offline" in response.json()["message"].lower()


def test_quantization_presets_are_capability_driven_and_context_filtered() -> None:
    with _client() as client:
        response = client.post(
            "/api/manager/workshed/options",
            headers=_headers(),
            json={
                "provider": "quantization.presets",
                "context": {"engine": "mlx_local"},
            },
        )
        assert response.status_code == 200
        assert response.json()["items"] == [
            {
                "id": "mlx-balanced",
                "label": "Balanced",
                "description": "4 bit · 128 group · affine",
                "metadata": {
                    "source": "workshed_capabilities",
                    "engine": "mlx_local",
                    "worker_id": "",
                    "bits": 4,
                    "group_size": 128,
                    "mode": "affine",
                },
            }
        ]


def test_quantization_preset_options_never_expose_remote_workers() -> None:
    """Workshed is local-only even though the legacy facade still knows Workers."""

    with _client() as client:
        unfiltered = client.post(
            "/api/manager/workshed/options",
            headers=_headers(),
            json={"provider": "quantization.presets"},
        )
        assert unfiltered.status_code == 200
        assert {item["metadata"]["engine"] for item in unfiltered.json()["items"]} == {
            "mlx_local"
        }
        assert all(not item["metadata"]["worker_id"] for item in unfiltered.json()["items"])

        remote = client.post(
            "/api/manager/workshed/options",
            headers=_headers(),
            json={
                "provider": "quantization.presets",
                "context": {"engine": "vllm_cuda", "worker_id": "worker-a"},
            },
        )
        assert remote.status_code == 200
        assert remote.json()["items"] == []
        assert "this mac only" in remote.json()["message"].lower()


def test_quantization_preset_lookup_does_not_probe_legacy_workers() -> None:
    class WorkerAwareQuantization(_Quantization):
        def __init__(self) -> None:
            self.capability_probe_count = 0

        def capabilities(self) -> dict[str, Any]:
            self.capability_probe_count += 1
            return super().capabilities()

    quantization = WorkerAwareQuantization()
    with _client(quantization=quantization, workshed=_Workshed()) as client:
        response = client.post(
            "/api/manager/workshed/options",
            headers=_headers(),
            json={"provider": "quantization.presets"},
        )

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["items"]] == ["mlx-balanced"]
    assert quantization.capability_probe_count == 0


def test_dataset_and_lm_eval_providers_keep_the_uniform_contract(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        desktop_ui,
        "_workshed_hub_dataset_options",
        lambda query, limit: (
            [
                desktop_ui._workshed_option_item(
                    "org/dataset",
                    "org/dataset",
                    "Dataset",
                    {"source": "hub"},
                )
            ],
            "",
        ),
    )
    monkeypatch.setattr(
        desktop_ui,
        "_workshed_lm_eval_task_options",
        lambda query, limit: (
            [
                desktop_ui._workshed_option_item(
                    "mmlu",
                    "mmlu",
                    "lm-eval task",
                    {"source": "lm_eval"},
                )
            ],
            "",
        ),
    )

    with _client() as client:
        for provider, expected_id in (
            ("datasets.hub", "org/dataset"),
            ("lm_eval.tasks", "mmlu"),
        ):
            response = client.post(
                "/api/manager/workshed/options",
                headers=_headers(),
                json={"provider": provider, "query": ""},
            )
            assert response.status_code == 200
            payload = response.json()
            assert payload["provider"] == provider
            assert payload["items"][0]["id"] == expected_id
            assert payload["next_cursor"] == ""


def test_lm_eval_provider_filters_installed_tasks_through_release_allowlist(
    monkeypatch: Any,
) -> None:
    tasks_module = types.ModuleType("lm_eval.tasks")

    class TaskManager:
        all_tasks = ["gsm8k", "mmlu", "unreviewed_new_task"]

    tasks_module.TaskManager = TaskManager  # type: ignore[attr-defined]
    package = types.ModuleType("lm_eval")
    package.tasks = tasks_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lm_eval", package)
    monkeypatch.setitem(sys.modules, "lm_eval.tasks", tasks_module)

    with _client() as client:
        response = client.post(
            "/api/manager/workshed/options",
            headers=_headers(),
            json={"provider": "lm_eval.tasks", "query": ""},
        )

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["items"]] == ["gsm8k", "mmlu"]
    assert all(
        item["metadata"]["allowlist_version"] == "0.2.0"
        for item in response.json()["items"]
    )


def test_target_modules_use_registered_model_config_and_reject_urls(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "llama"}),
        encoding="utf-8",
    )
    workshed = _Workshed(tmp_path)
    with _client(workshed=workshed) as client:
        response = client.post(
            "/api/manager/workshed/options",
            headers=_headers(),
            json={
                "provider": "model.target_modules",
                "query": "q_proj",
                "context": {
                    "source": "managed",
                    "ref": "workshed/managed",
                    "revision": "",
                },
            },
        )
        assert response.status_code == 200
        assert [item["id"] for item in response.json()["items"]] == ["q_proj"]

        rejected = client.post(
            "/api/manager/workshed/options",
            headers=_headers(),
            json={
                "provider": "model.target_modules",
                "context": {"source": "hub", "ref": "https://example.test/model"},
            },
        )
        assert rejected.status_code == 200
        assert rejected.json()["items"] == []
        assert "not accepted" in rejected.json()["message"]


def test_run_receipt_contract_supports_in_progress_runs_and_404() -> None:
    with _client() as client:
        response = client.get(
            "/api/manager/workshed/runs/run-1/receipt",
            headers=_headers(),
        )
        assert response.status_code == 200
        assert response.json() == {
            "ok": True,
            "receipt": {
                "schema_version": 2,
                "run_id": "run-1",
                "status": "running",
                "finalized": False,
            },
        }

        missing = client.get(
            "/api/manager/workshed/runs/missing/receipt",
            headers=_headers(),
        )
        assert missing.status_code == 404
        assert "not found" in missing.json()["detail"].lower()


def test_run_receipt_route_uses_the_real_workshed_service(tmp_path: Path) -> None:
    """Do not let a test double hide drift between the route and service API."""

    service = WorkshedService(root=tmp_path)
    with service._lock:
        service._runs["real-run"] = {
            "id": "real-run",
            "status": "running",
            "work_order_id": "work-order-1",
            "work_order_revision": 3,
            "work_order_hash": "work-order-hash",
            "plan_digest": "plan-digest",
            "catalog_digest": "catalog-digest",
            "capability_digest": "capability-digest",
            "created_at": "2026-08-22T00:00:00Z",
            "steps": [],
            "artifacts": [],
        }
    try:
        with _client(workshed=service) as client:
            active = client.get(
                "/api/manager/workshed/runs/real-run/receipt",
                headers=_headers(),
            )
            missing = client.get(
                "/api/manager/workshed/runs/not-a-run/receipt",
                headers=_headers(),
            )
        assert active.status_code == 200
        assert active.json()["receipt"] == {
            "schema_version": 2,
            "run_id": "real-run",
            "status": "running",
            "finalized": False,
            "work_order_id": "work-order-1",
            "work_order_revision": 3,
            "work_order_hash": "work-order-hash",
            "plan_digest": "plan-digest",
            "catalog_digest": "catalog-digest",
            "capability_digest": "capability-digest",
            "created_at": "2026-08-22T00:00:00Z",
            "finalized_at": "",
            "steps": [],
            "artifacts": [],
        }
        assert missing.status_code == 404
        assert "not found" in missing.json()["detail"].lower()
    finally:
        service.shutdown()


def test_real_workshed_capabilities_are_strictly_local(tmp_path: Path) -> None:
    capabilities = WorkshedService(root=tmp_path).capabilities()

    assert capabilities["runtime"]["local_only"] is True
    assert {runner["id"] for runner in capabilities["runners"]} == {
        "mlx_local",
        "hf_mps",
    }
    assert all(runner["kind"] == "local" for runner in capabilities["runners"])
    serialized = json.dumps(capabilities).lower()
    assert "vllm_cuda" not in serialized
    assert "worker_id" not in serialized
