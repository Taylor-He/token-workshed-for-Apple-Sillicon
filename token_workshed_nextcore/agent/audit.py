"""Minimal, secret-safe audit trail for first-party Agent tool execution."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


def audit_tool_call(
    *,
    app_data_dir: Path,
    run_id: str,
    tool: str,
    arguments: dict[str, Any],
    outcome: str,
    duration_ms: float,
) -> None:
    """Store metadata only; never persist prompts, arguments, or tool output."""
    encoded = json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str).encode()
    record = {
        "time": int(time.time()),
        "run_id": run_id[:80],
        "tool": tool[:80],
        "arguments_sha256": hashlib.sha256(encoded).hexdigest(),
        "outcome": outcome[:40],
        "duration_ms": round(max(0.0, duration_ms), 1),
    }
    path = app_data_dir / "agent" / "audit.jsonl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK:
            fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        # Auditing must not leak secrets into logs or turn a tool result into a
        # false success. The tool caller reports the actual result separately.
        return
