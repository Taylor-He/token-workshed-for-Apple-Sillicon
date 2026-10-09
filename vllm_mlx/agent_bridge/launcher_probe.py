"""Launch and health-check helpers for local vLLM endpoints."""

from __future__ import annotations

import json
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import requests


def _is_port_open(host: str, port: int, timeout: float = 0.6) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        return sock.connect_ex((host, port)) == 0
    except OSError:
        return False
    finally:
        sock.close()


def _check_models(base_url: str, timeout_seconds: float) -> tuple[bool, str, list[str]]:
    try:
        response = requests.get(
            f"{base_url.rstrip('/')}/v1/models",
            timeout=(3.0, timeout_seconds),
        )
    except Exception as exc:
        return False, f"/v1/models request failed: {exc}", []
    if not response.ok:
        return False, f"/v1/models HTTP {response.status_code}", []
    try:
        payload = response.json()
    except Exception as exc:
        return False, f"/v1/models non-JSON response: {exc}", []
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return False, "/v1/models missing data list", []
    models: list[str] = []
    for item in data:
        if isinstance(item, dict):
            model_id = item.get("id")
            if isinstance(model_id, str) and model_id.strip():
                models.append(model_id.strip())
    return True, "", models


def _check_chat_completion(
    *,
    base_url: str,
    model: str,
    timeout_seconds: float,
) -> tuple[bool, str]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "reply with: ok"}],
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": False,
    }
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            json=payload,
            timeout=(3.0, timeout_seconds),
        )
    except Exception as exc:
        return False, f"/v1/chat/completions request failed: {exc}"
    if not response.ok:
        body_preview = response.text[:220] if isinstance(response.text, str) else ""
        return (
            False,
            f"/v1/chat/completions HTTP {response.status_code}: {body_preview}",
        )
    try:
        payload = response.json()
    except Exception as exc:
        return False, f"/v1/chat/completions non-JSON response: {exc}"
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices:
        return False, "/v1/chat/completions missing choices"
    return True, ""


def _model_id_matches(expected: str, served: str) -> bool:
    """Accept the common repo-id/served-alias forms returned by /v1/models."""
    expected_text = str(expected or "").strip().rstrip("/").lower()
    served_text = str(served or "").strip().rstrip("/").lower()
    if not expected_text or not served_text:
        return False
    if (
        expected_text == served_text
        or expected_text.endswith(f"/{served_text}")
        or served_text.endswith(f"/{expected_text}")
    ):
        return True

    # Local MLX launches can expose an absolute snapshot path while the UI
    # keeps the Hugging Face repo id (or vice versa).  Matching the final model
    # directory is safe here because the full id/path comparison above still
    # takes precedence and we only use this for a served-model alias.
    expected_leaf = Path(expected_text).name
    served_leaf = Path(served_text).name
    return bool(expected_leaf and expected_leaf == served_leaf)


def _served_model_alias(expected: str, served_models: list[str]) -> str | None:
    """Return the server's exact id for a requested model alias."""
    for served in served_models:
        if _model_id_matches(expected, served):
            return served
    return None


def launch_and_probe(
    *,
    command: list[str],
    host: str,
    port: int,
    base_url: str,
    expected_model: str,
    startup_timeout_seconds: float,
    request_timeout_seconds: float,
    reuse_running: bool,
    stop_after_check: bool,
) -> dict[str, Any]:
    process: subprocess.Popen[str] | None = None
    reused_existing = False
    launched = False
    logs_tail = ""

    if _is_port_open(host, port):
        if not reuse_running:
            return {
                "ok": False,
                "launched": False,
                "reused_existing": False,
                "pid": None,
                "error": f"Server port {port} is already occupied.",
                "models": [],
                "logs_tail": "",
            }
        reused_existing = True
    else:
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            launched = True
        except Exception as exc:
            return {
                "ok": False,
                "launched": False,
                "reused_existing": False,
                "pid": None,
                "error": f"Failed to launch vllm process: {exc}",
                "models": [],
                "logs_tail": "",
            }

    models: list[str] = []
    deadline = time.time() + max(5.0, startup_timeout_seconds)
    models_ok = False
    models_error = ""
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            out, err = process.communicate(timeout=1.0)
            logs_tail = "\n".join((out or "", err or "")).strip()[-2000:]
            return {
                "ok": False,
                "launched": launched,
                "reused_existing": reused_existing,
                "pid": process.pid,
                "error": f"vLLM process exited before ready (code {process.returncode}).",
                "models": [],
                "logs_tail": logs_tail,
            }

        models_ok, models_error, models = _check_models(
            base_url=base_url,
            timeout_seconds=max(4.0, request_timeout_seconds),
        )
        if models_ok:
            break
        time.sleep(0.5)

    if not models_ok:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except Exception:
                process.kill()
        return {
            "ok": False,
            "launched": launched,
            "reused_existing": reused_existing,
            "pid": process.pid if process else None,
            "error": f"Startup health check failed: {models_error or 'timeout waiting /v1/models'}",
            "models": models,
            "logs_tail": logs_tail,
        }

    served_model = _served_model_alias(expected_model, models)
    model_present = served_model is not None
    if not model_present:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except Exception:
                process.kill()
        available = ", ".join(models[:6]) if models else "none"
        return {
            "ok": False,
            "launched": launched,
            "reused_existing": reused_existing,
            "pid": process.pid if process else None,
            "error": (
                f"Expected model '{expected_model}' is not served; "
                f"available model(s): {available}."
            ),
            "models": models,
            "model_present": False,
            "served_model": None,
            "logs_tail": logs_tail,
        }

    chat_ok, chat_error = _check_chat_completion(
        base_url=base_url,
        model=served_model or expected_model,
        timeout_seconds=max(4.0, request_timeout_seconds),
    )
    if not chat_ok:
        if process is not None and process.poll() is None and stop_after_check:
            process.terminate()
        return {
            "ok": False,
            "launched": launched,
            "reused_existing": reused_existing,
            "pid": process.pid if process else None,
            "error": f"Chat health check failed: {chat_error}",
            "models": models,
            "served_model": served_model,
            "logs_tail": logs_tail,
        }

    if process is not None and process.poll() is None and stop_after_check:
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except Exception:
            process.kill()

    return {
        "ok": True,
        "launched": launched,
        "reused_existing": reused_existing,
        "pid": process.pid if process else None,
        "error": "",
        "models": models,
        "model_present": True,
        "served_model": served_model,
        "logs_tail": logs_tail,
        "health_summary": json.dumps(
            {"models_count": len(models), "model_present": model_present}
        ),
    }
