# SPDX-License-Identifier: Apache-2.0
"""Automatic cleanup behavior for incomplete local model snapshots."""

from __future__ import annotations

from types import SimpleNamespace

from vllm_mlx.desktop_ui import LocalModelServerManager


def test_discovery_requests_automatic_integrity_cleanup(monkeypatch) -> None:
    manager = LocalModelServerManager("python", "127.0.0.1", 8000)
    calls: list[tuple[bool, bool]] = []

    def fake_cleanup(*, force: bool = False, apply_delete: bool = False):
        calls.append((force, apply_delete))
        return {"checked": True, "removed": [], "pending": [], "errors": []}

    monkeypatch.setattr(manager, "_cleanup_incomplete_models_locked", fake_cleanup)
    monkeypatch.setattr(
        "huggingface_hub.scan_cache_dir",
        lambda: SimpleNamespace(repos=[]),
    )

    assert manager.discover_models() == []
    assert calls == [(False, True)]


def test_incomplete_snapshot_is_removed_during_discovery(monkeypatch, tmp_path) -> None:
    removed_hashes: list[str] = []

    class DeleteStrategy:
        def execute(self) -> None:
            removed_hashes.extend(["deadbeef"])
            cache.repos = []

    cache = SimpleNamespace()
    cache.repos = [
        SimpleNamespace(
            repo_type="model",
            repo_id="owner/incomplete-model",
            revisions=[
                SimpleNamespace(
                    snapshot_path=str(tmp_path / "partial-snapshot"),
                    commit_hash="deadbeef",
                )
            ],
            repo_path=str(tmp_path / "owner--incomplete-model"),
        )
    ]

    def delete_revisions(*hashes: str) -> DeleteStrategy:
        assert hashes == ("deadbeef",)
        return DeleteStrategy()

    cache.delete_revisions = delete_revisions
    monkeypatch.setattr("huggingface_hub.scan_cache_dir", lambda: cache)

    manager = LocalModelServerManager("python", "127.0.0.1", 8000)
    assert manager.discover_models() == []
    assert removed_hashes == ["deadbeef"]

    report = manager.model_integrity_report()
    assert report["removed"] == [
        {"repo_id": "owner/incomplete-model", "reason": "incomplete"}
    ]
    assert report["pending"] == []
    assert report["requires_confirmation"] is False
