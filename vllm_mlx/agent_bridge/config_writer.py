"""Runtime config writers for OpenClaw and Hermes."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from .defaults import BridgePaths


@dataclass
class WriteResult:
    written: bool
    path: str
    backup_path: str | None
    reason: str = ""


def _backup_path(path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return path.with_name(f"{path.name}.bak.{timestamp}")


def _write_text_with_backup(path: Path, content: str) -> WriteResult:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    if path.exists():
        backup = _backup_path(path)
        shutil.copy2(path, backup)
    path.write_text(content, encoding="utf-8")
    return WriteResult(
        written=True,
        path=str(path),
        backup_path=str(backup) if backup else None,
        reason="",
    )


def detect_api_mode(base_url: str) -> str:
    normalized = (base_url or "").strip().lower().rstrip("/")
    if "api.openai.com" in normalized or "api.x.ai" in normalized:
        return "codex_responses"
    if normalized.endswith("/anthropic"):
        return "anthropic_messages"
    return "chat_completions"


def is_openai_compatible_url(base_url: str) -> bool:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if not parsed.netloc:
        return False
    if detect_api_mode(base_url) == "anthropic_messages":
        return False
    return True


def build_openclaw_config(
    *,
    model_id: str,
    base_url_v1: str,
    context_window: int,
    max_tokens: int,
    api_key: str,
    workspace_dir: str,
    repo_root: str,
) -> dict[str, Any]:
    provider_model = f"openai/{model_id}"
    return {
        "gateway": {"mode": "local"},
        "agents": {
            "defaults": {
                "workspace": workspace_dir,
                "repoRoot": repo_root,
                "timeoutSeconds": 1200,
                "model": {"primary": provider_model},
                "models": {provider_model: {"alias": "token-workshed-local"}},
            }
        },
        "models": {
            "mode": "merge",
            "providers": {
                "openai": {
                    "api": "openai-completions",
                    "baseUrl": base_url_v1,
                    "authHeader": False,
                    "apiKey": api_key or "token-workshed-local",
                    "models": [
                        {
                            "id": model_id,
                            "name": model_id,
                            "reasoning": False,
                            "input": ["text", "image"],
                            "cost": {
                                "input": 0,
                                "output": 0,
                                "cacheRead": 0,
                                "cacheWrite": 0,
                            },
                            "contextWindow": int(max(1, context_window)),
                            "maxTokens": int(max(1, max_tokens)),
                        }
                    ],
                }
            },
        },
        "tools": {"web": {"search": {"enabled": False}, "fetch": {"enabled": True}}},
    }


def write_openclaw_config(
    *,
    paths: BridgePaths,
    model_id: str,
    base_url_v1: str,
    context_window: int,
    max_tokens: int,
    api_key: str,
    dry_run: bool,
) -> WriteResult:
    config_payload = build_openclaw_config(
        model_id=model_id,
        base_url_v1=base_url_v1,
        context_window=context_window,
        max_tokens=max_tokens,
        api_key=api_key,
        workspace_dir=str(paths.openclaw_workspace_dir),
        repo_root=str(paths.root_dir),
    )
    serialized = json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n"
    if dry_run:
        return WriteResult(
            written=False,
            path=str(paths.openclaw_config_path),
            backup_path=None,
            reason="dry-run",
        )
    return _write_text_with_backup(paths.openclaw_config_path, serialized)


def write_hermes_config(
    *,
    paths: BridgePaths,
    model_id: str,
    base_url_v1: str,
    api_key: str,
    dry_run: bool,
) -> dict[str, WriteResult]:
    if not is_openai_compatible_url(base_url_v1):
        reason = (
            "Hermes config skipped: base URL is not OpenAI-compatible; manual setup required."
        )
        return {
            "env": WriteResult(
                written=False,
                path=str(paths.hermes_env_path),
                backup_path=None,
                reason=reason,
            ),
            "config": WriteResult(
                written=False,
                path=str(paths.hermes_config_path),
                backup_path=None,
                reason=reason,
            ),
        }

    mode = detect_api_mode(base_url_v1)
    env_lines = [
        f"OPENAI_BASE_URL={base_url_v1}",
        f"OPENROUTER_BASE_URL={base_url_v1}",
        f"OPENAI_API_KEY={api_key or 'token-workshed-local'}",
        "HERMES_INFERENCE_PROVIDER=custom",
        "HERMES_QUIET=1",
        "NO_COLOR=1",
        "FORCE_COLOR=0",
    ]
    env_content = "\n".join(env_lines) + "\n"
    config_payload = {
        "model": {
            "provider": "custom",
            "default": model_id,
            "base_url": base_url_v1,
            "api_mode": mode,
            "api_key": api_key or "token-workshed-local",
        }
    }
    config_content = yaml.safe_dump(config_payload, sort_keys=False, allow_unicode=False)

    if dry_run:
        return {
            "env": WriteResult(
                written=False,
                path=str(paths.hermes_env_path),
                backup_path=None,
                reason="dry-run",
            ),
            "config": WriteResult(
                written=False,
                path=str(paths.hermes_config_path),
                backup_path=None,
                reason="dry-run",
            ),
        }

    env_result = _write_text_with_backup(paths.hermes_env_path, env_content)
    cfg_result = _write_text_with_backup(paths.hermes_config_path, config_content)
    return {"env": env_result, "config": cfg_result}


def write_result_to_dict(result: WriteResult) -> dict[str, Any]:
    return asdict(result)
