# SPDX-License-Identifier: Apache-2.0
"""Short-lived index for checkpoints already present in the Hugging Face cache."""

from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=8)
def _snapshot_index(root: str, time_bucket: int) -> dict[str, tuple[str, str]]:
    """Share one Hub scan across model lookups for ten seconds."""
    try:
        from huggingface_hub import scan_cache_dir

        repos = scan_cache_dir(cache_dir=root).repos
    except Exception:
        return {}
    index = {}
    for repo in repos:
        model_id = str(getattr(repo, "repo_id", "") or "")
        if not model_id or getattr(repo, "repo_type", "model") != "model":
            continue
        revisions = list(getattr(repo, "revisions", ()))
        refs = getattr(repo, "refs", {})
        main = refs.get("main") if isinstance(refs, dict) else None
        revision = next((item for item in revisions if item.commit_hash == main), None)
        if revision is None and revisions:
            revision = revisions[0] if len(revisions) == 1 else (
                max(revisions, key=lambda item: item.size_on_disk) if not main else None
            )
        if revision is not None:
            index[model_id] = (str(revision.snapshot_path), str(revision.commit_hash))
    return index


def invalidate_local_snapshot_cache() -> None:
    """Drop cached Hub metadata after NextCore downloads or deletes a checkpoint."""
    _snapshot_index.cache_clear()


def local_huggingface_snapshot(model_id: str) -> tuple[str, str]:
    """Return a cached Hub model's main snapshot path and commit, if present."""
    model_id = str(model_id or "").strip()
    if not model_id or "/" not in model_id or model_id.startswith(("/", ".")):
        return "", ""
    roots = [
        Path(os.environ[name]).expanduser()
        for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE")
        if os.environ.get(name, "").strip()
    ]
    if home := os.environ.get("HF_HOME", "").strip():
        roots.append(Path(home).expanduser() / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    bucket = int(time.monotonic() // 10)
    for root in roots:
        if root.is_dir():
            key = str(root.resolve())
            snapshot = _snapshot_index(key, bucket).get(model_id)
            if snapshot is not None:
                if Path(snapshot[0]).is_dir():
                    return snapshot
                invalidate_local_snapshot_cache()
    return "", ""
