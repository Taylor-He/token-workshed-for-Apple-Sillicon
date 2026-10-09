"""Managed Workshed discovery and narrowly scoped, recoverable cleanup."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_mlx.desktop_ui import LocalModelServerManager, _CombinedManagedModelResolver
from vllm_mlx.workshed import WorkshedService


def _managed_store(tmp_path: Path) -> tuple[WorkshedService, str, str, Path]:
    service = WorkshedService(root=tmp_path / "workshed")
    digest = "a" * 64
    path = service.root / "artifacts" / digest
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    content_id = f"workshed/{digest[:24]}"
    alias_id = "workshed/ui-quantization-test"
    service._artifacts = {
        content_id: {"id": content_id, "kind": "model", "path": str(path), "sha256": digest, "registered": False},
        alias_id: {"id": alias_id, "kind": "model", "path": str(path), "registered": True, "parent_artifact_ids": [content_id]},
    }
    service._runs["test-run"] = {"id": "test-run", "status": "completed", "artifacts": [content_id, alias_id]}
    service._persist_locked()
    return service, content_id, alias_id, path


def _manager(monkeypatch, service: WorkshedService) -> LocalModelServerManager:
    monkeypatch.setattr("huggingface_hub.scan_cache_dir", lambda: SimpleNamespace(repos=[]))
    quantization = SimpleNamespace(managed_model_ids=lambda: [], resolve_model=lambda _id: None)
    manager = LocalModelServerManager("python", "127.0.0.1", 8000)
    manager.set_managed_model_resolver(_CombinedManagedModelResolver(quantization, service))
    return manager


def test_discovery_accepts_callable_object_and_bound_method(monkeypatch, tmp_path: Path) -> None:
    service, _, alias_id, _ = _managed_store(tmp_path)
    manager = _manager(monkeypatch, service)
    assert manager.discover_models(False) == [alias_id]
    manager.set_managed_model_resolver(service.resolve_model)
    assert manager.discover_models(False) == [alias_id]


def test_delete_refuses_active_model_and_an_alias_of_its_running_path(monkeypatch, tmp_path: Path) -> None:
    service, _, alias_id, path = _managed_store(tmp_path)
    manager = _manager(monkeypatch, service)
    manager._active_model = alias_id
    assert manager.delete_local_model(alias_id)[0] is False
    manager._active_model = str(path)
    assert manager.delete_local_model(alias_id)[0] is False
    assert service.resolve_model(alias_id) == str(path)


def test_model_delete_only_unregisters_exact_alias_and_retains_shared_content(monkeypatch, tmp_path: Path) -> None:
    service, content_id, alias_id, path = _managed_store(tmp_path)
    sibling_id = "workshed/keep-this-model"
    service._artifacts[sibling_id] = {**deepcopy(service._artifacts[alias_id]), "id": sibling_id}
    manager = _manager(monkeypatch, service)
    ok, message, models = manager.delete_local_model(alias_id)
    assert ok is True and "Unregistered" in message
    assert models == [sibling_id]
    assert path.is_dir()
    assert service._artifacts[alias_id]["deletion_kind"] == "alias"
    with pytest.raises(ValueError, match="still referenced"):
        service.artifact_action(content_id, "trash")
    assert path.is_dir()


def test_trash_moves_only_unreferenced_owned_content_and_keeps_history(monkeypatch, tmp_path: Path) -> None:
    service, content_id, alias_id, path = _managed_store(tmp_path)
    manager = _manager(monkeypatch, service)
    assert manager.delete_local_model(alias_id)[0] is True
    monkeypatch.setattr(service, "_trash_directory", lambda: tmp_path / "Trash")
    result = service.artifact_action(content_id, "trash")
    assert not path.exists()
    assert (Path(result["trash_path"]) / "config.json").is_file()
    assert result["deleted"] is True and result["deletion_kind"] == "trash"
    assert service.artifact_action(content_id, "trash") == result
    assert service.get_run("test-run")["artifact_lifecycle"][content_id]["deleted"] is True
    restored = WorkshedService(root=service.root)
    assert restored.managed_model_ids() == []
    assert restored.get_run("test-run")["status"] == "completed"
    assert any(item["id"] == content_id and item["deleted"] for item in restored.artifacts())


@pytest.mark.parametrize("reference", ["artifact", "order", "run"])
def test_trash_refuses_dependent_references(monkeypatch, tmp_path: Path, reference: str) -> None:
    service, content_id, alias_id, path = _managed_store(tmp_path)
    service.unregister_model(alias_id)
    monkeypatch.setattr(service, "_trash_directory", lambda: tmp_path / "Trash")
    if reference == "artifact":
        service._artifacts["workshed/dependent"] = {"id": "workshed/dependent", "parent_artifact_ids": [content_id]}
    elif reference == "order":
        service._work_orders["saved"] = {"blocks": [{"params": {"ref": content_id}}]}
    else:
        service._runs["active"] = {"status": "running", "inputs": {"model": str(path)}}
    with pytest.raises(ValueError, match="referenced"):
        service.artifact_action(content_id, "trash")
    assert path.is_dir()


@pytest.mark.parametrize("unsafe", ["source", "outside", "root", "symlink", "child_symlink", "digest"])
def test_trash_refuses_sources_paths_and_symlinks(monkeypatch, tmp_path: Path, unsafe: str) -> None:
    service, content_id, alias_id, path = _managed_store(tmp_path)
    service.unregister_model(alias_id)
    monkeypatch.setattr(service, "_trash_directory", lambda: tmp_path / "Trash")
    artifact = service._artifacts[content_id]
    if unsafe == "source":
        artifact["source_kind"] = "local"
    elif unsafe == "outside":
        artifact["path"] = str(tmp_path)
    elif unsafe == "root":
        artifact["path"] = str(service.root / "artifacts")
    elif unsafe == "symlink":
        link = service.root / "artifacts" / ("b" * 64)
        link.symlink_to(path, target_is_directory=True)
        artifact["path"] = str(link)
    elif unsafe == "child_symlink":
        (path / "outside").symlink_to(tmp_path)
    else:
        artifact["sha256"] = "untrusted"
    with pytest.raises(PermissionError):
        service.artifact_action(content_id, "trash")
    assert (path / "config.json").is_file()


def test_deleted_alias_cannot_be_registered_again_by_artifact_action(tmp_path: Path) -> None:
    service, _, alias_id, _ = _managed_store(tmp_path)
    service.unregister_model(alias_id)
    with pytest.raises(ValueError, match="tombstone"):
        service.artifact_action(alias_id, "register")


def test_trash_rolls_back_directory_and_lifecycle_if_persistence_fails(monkeypatch, tmp_path: Path) -> None:
    service, content_id, alias_id, path = _managed_store(tmp_path)
    service.unregister_model(alias_id)
    monkeypatch.setattr(service, "_trash_directory", lambda: tmp_path / "Trash")

    def fail_persistence() -> None:
        raise OSError("simulated disk failure")

    monkeypatch.setattr(service, "_persist_locked", fail_persistence)
    with pytest.raises(OSError, match="disk failure"):
        service.artifact_action(content_id, "trash")
    assert path.is_dir()
    assert not service._artifacts[content_id].get("deleted")
    assert "artifact_lifecycle" not in service._runs["test-run"]


@pytest.mark.parametrize("ref", ["org/repo", "/missing/workshed-test-model"])
def test_preflight_rejects_invalid_local_model_before_run(tmp_path: Path, ref: str) -> None:
    service = WorkshedService(root=tmp_path / "workshed")
    order = service.create_work_order({"name": "local-test"})
    order["blocks"].extend([
        {"id": "source", "type_id": "load_model", "lane": "source", "order": 1,
         "params": {"source_kind": "local", "ref": ref}},
        {"id": "deliver", "type_id": "register_model", "lane": "deliver", "order": 2, "params": {}},
    ])
    preflight = service.preflight({"work_order": order})
    assert preflight["ok"] is False
    assert any(item["code"] == "local_source" for item in preflight["errors"])
