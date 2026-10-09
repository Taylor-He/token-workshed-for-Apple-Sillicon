#!/usr/bin/env python3
"""Run A-E timeout diagnosis matrix for vllm-mlx + OpenClaw."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

import requests


KEYWORDS = (
    "timeout",
    "tool_call",
    "tool_calls",
    "function_call",
    "retry",
    "repair",
    "invalid json",
    "parse",
    "schema",
    "abort",
    "max_tokens",
    "finish_reason",
    "stream",
)

HTTP_CONNECT_TIMEOUT_SECONDS = 5
HTTP_READ_TIMEOUT_SECONDS = 90
OPENCLAW_UI_CASE_DEADLINE_SECONDS = 90
DEFAULT_REPORT_DIR = Path(".openclaw") / "diagnostics"
EMBEDDED_STAGE_RE = re.compile(
    r"\[embedded-stage\]\s+stage=(?P<stage>[a-z_]+)\s+"
    r"elapsed_ms=(?P<elapsed_ms>\d+)\s+"
    r"runId=(?P<run_id>[^\s]+)\s+sessionId=(?P<session_id>[^\s]+)"
)


def _timestamp_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _truncate(text: Any, limit: int = 320) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."


def _extract_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
    if isinstance(content, dict) and isinstance(content.get("text"), str):
        return content["text"]
    return str(content or "")


def _safe_json(response: requests.Response) -> tuple[dict[str, Any] | None, str | None]:
    try:
        data = response.json()
    except Exception as exc:  # pragma: no cover - network parse failure
        return None, str(exc)
    if not isinstance(data, dict):
        return None, "response is not a JSON object"
    return data, None


def _summarize_tool_calls(message: dict[str, Any]) -> dict[str, Any]:
    tool_calls = message.get("tool_calls")
    has_tool_calls = isinstance(tool_calls, list) and len(tool_calls) > 0
    invalid_arguments = False
    invalid_reason = ""
    if has_tool_calls:
        for item in tool_calls:
            if not isinstance(item, dict):
                continue
            fn = item.get("function")
            if not isinstance(fn, dict):
                continue
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    parsed = json.loads(args)
                except Exception as exc:
                    invalid_arguments = True
                    invalid_reason = f"tool_calls.arguments invalid JSON: {exc}"
                    break
                if not isinstance(parsed, dict):
                    invalid_arguments = True
                    invalid_reason = "tool_calls.arguments JSON is not object"
                    break
    content_text = _extract_content_text(message.get("content", "")).strip()
    return {
        "has_tool_calls": has_tool_calls,
        "tool_calls_count": len(tool_calls) if isinstance(tool_calls, list) else 0,
        "content_preview": _truncate(content_text, limit=240),
        "literal_tool_call_markup": bool(
            re.search(r"(?is)<\s*tool_call\b|\[calling tool:", content_text)
        ),
        "invalid_json": invalid_arguments,
        "invalid_json_reason": invalid_reason,
    }


def _count_keywords(text: str) -> dict[str, int]:
    lower = str(text or "").lower()
    return {word: lower.count(word) for word in KEYWORDS}


def _read_log_delta(path: Path, start_offset: int) -> tuple[str, int]:
    if not path.exists():
        return "", start_offset
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(start_offset)
        delta = handle.read()
        end = handle.tell()
    return delta, end


def _resolve_finish_reason(case: dict[str, Any]) -> str:
    finish_reason = str(case.get("finish_reason") or "").strip()
    if finish_reason:
        return finish_reason
    stop_reason = str(case.get("stop_reason") or "").strip()
    if stop_reason:
        return stop_reason
    return ""


def _normalize_agent_session_id(
    conversation_id: str,
    *,
    selected_model: str,
    runtime: str,
) -> str:
    raw_conv = str(conversation_id or "").strip() or "main"
    raw_model = str(selected_model or "").strip() or "default"
    runtime_tag = "hermes" if str(runtime or "").strip().lower() == "hermes" else "openclaw"
    raw = f"{raw_conv}|{runtime_tag}|{raw_model}"
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", raw).strip("-.")
    if not normalized:
        normalized = "main"
    return f"token-workshed-{normalized[:96]}"


def _parse_embedded_stage_events(log_text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in str(log_text or "").splitlines():
        match = EMBEDDED_STAGE_RE.search(line)
        if not match:
            continue
        elapsed_raw = str(match.group("elapsed_ms") or "").strip()
        elapsed_ms: int | None
        if elapsed_raw.isdigit():
            elapsed_ms = int(elapsed_raw)
        else:
            elapsed_ms = None
        events.append(
            {
                "stage": str(match.group("stage") or "").strip(),
                "elapsed_ms": elapsed_ms,
                "run_id": str(match.group("run_id") or "").strip(),
                "session_id": str(match.group("session_id") or "").strip(),
                "line": line.strip(),
            }
        )
    return events


def _extract_stage_token(line: str, key: str) -> str:
    match = re.search(rf"\b{re.escape(key)}=(?P<value>[^\s]+)", str(line or ""))
    if not match:
        return ""
    return str(match.group("value") or "").strip()


def _extract_stage_int(line: str, key: str) -> int | None:
    raw = _extract_stage_token(line, key)
    if not raw:
        return None
    if raw.isdigit():
        return int(raw)
    return None


def _stage_diagnostics(
    *,
    log_text: str,
    expected_session_id: str,
    expected_run_id: str = "",
) -> dict[str, Any]:
    events = _parse_embedded_stage_events(log_text)
    scoped = [event for event in events if event.get("session_id") == expected_session_id]
    expected_run_id_trimmed = str(expected_run_id or "").strip()
    if not scoped and expected_run_id_trimmed:
        scoped = [
            event
            for event in events
            if str(event.get("run_id") or "").strip() == expected_run_id_trimmed
        ]
    if not scoped:
        return {
            "embedded_stage_events_count": 0,
            "embedded_run_id": "",
            "embedded_session_id": expected_session_id,
            "embedded_last_stage": "",
            "embedded_last_stage_elapsed_ms": None,
            "embedded_seen_run_cleanup_done": False,
            "embedded_seen_run_timeout": False,
            "embedded_seen_compaction_timeout": False,
            "embedded_first_chunk_deadline_ms": None,
            "embedded_tools_count": None,
            "embedded_tools_mode": "",
            "embedded_tools_count_label": "",
            "embedded_stage_tail": [],
        }
    latest_run_id = (
        expected_run_id_trimmed
        if expected_run_id_trimmed
        else str(scoped[-1].get("run_id") or "").strip()
    )
    scoped_run = [
        event for event in scoped if str(event.get("run_id") or "").strip() == latest_run_id
    ]
    if not scoped_run:
        latest_run_id = str(scoped[-1].get("run_id") or "").strip()
        scoped_run = [
            event for event in scoped if str(event.get("run_id") or "").strip() == latest_run_id
        ]
    if not scoped_run:
        scoped_run = scoped
    last = scoped_run[-1]
    stage_names = [str(event.get("stage") or "").strip() for event in scoped_run]
    latest_model_request = next(
        (
            event
            for event in reversed(scoped_run)
            if str(event.get("stage") or "").strip() == "model_request_start"
        ),
        None,
    )
    first_chunk_deadline_ms = None
    tools_count = None
    tools_mode = ""
    tools_count_label = ""
    tools_count_value: Any = None
    if isinstance(latest_model_request, dict):
        model_line = str(latest_model_request.get("line") or "")
        first_chunk_deadline_ms = _extract_stage_int(model_line, "first_chunk_deadline_ms")
        tools_count = _extract_stage_int(model_line, "tools_count")
        tools_mode = _extract_stage_token(model_line, "tools_mode")
        if tools_mode == "full":
            tools_count_label = "full"
            tools_count_value = "full"
        elif tools_count is not None:
            tools_count_label = str(tools_count)
            tools_count_value = tools_count

    return {
        "embedded_stage_events_count": len(scoped_run),
        "embedded_run_id": latest_run_id,
        "embedded_session_id": expected_session_id,
        "embedded_last_stage": str(last.get("stage") or "").strip(),
        "embedded_last_stage_elapsed_ms": last.get("elapsed_ms"),
        "embedded_seen_run_cleanup_done": "run_cleanup_done" in stage_names,
        "embedded_seen_run_timeout": "run_timeout" in stage_names,
        "embedded_seen_compaction_timeout": "compaction_timeout" in stage_names,
        "embedded_first_chunk_deadline_ms": first_chunk_deadline_ms,
        "embedded_tools_count": tools_count_value,
        "embedded_tools_mode": tools_mode,
        "embedded_tools_count_label": tools_count_label,
        "embedded_stage_tail": stage_names[-8:],
    }


def _wait_for_embedded_cleanup(
    *,
    gateway_log_path: Path,
    start_offset: int,
    expected_session_id: str,
    expected_run_id: str = "",
    wait_timeout_seconds: int = 45,
) -> dict[str, Any]:
    wait_started = time.perf_counter()
    offset = start_offset
    last_diag: dict[str, Any] = {
        "embedded_stage_events_count": 0,
        "embedded_run_id": "",
        "embedded_session_id": expected_session_id,
        "embedded_last_stage": "",
        "embedded_last_stage_elapsed_ms": None,
        "embedded_seen_run_cleanup_done": False,
        "embedded_seen_run_timeout": False,
        "embedded_seen_compaction_timeout": False,
        "embedded_first_chunk_deadline_ms": None,
        "embedded_tools_count": None,
        "embedded_tools_mode": "",
        "embedded_tools_count_label": "",
        "embedded_stage_tail": [],
    }

    while (time.perf_counter() - wait_started) < float(max(1, wait_timeout_seconds)):
        delta, offset = _read_log_delta(gateway_log_path, offset)
        if delta:
            last_diag = _stage_diagnostics(
                log_text=delta,
                expected_session_id=expected_session_id,
                expected_run_id=expected_run_id,
            )
            if last_diag.get("embedded_seen_run_cleanup_done"):
                break
        time.sleep(0.5)

    return {
        "timeout_cleanup_wait_ms": round((time.perf_counter() - wait_started) * 1000.0, 2),
        "timeout_cleanup_wait_seconds": max(1, wait_timeout_seconds),
        "timeout_cleanup_observed": bool(last_diag.get("embedded_seen_run_cleanup_done")),
        "timeout_last_stage": str(last_diag.get("embedded_last_stage") or ""),
        "timeout_last_stage_elapsed_ms": last_diag.get("embedded_last_stage_elapsed_ms"),
        "timeout_stage_tail": list(last_diag.get("embedded_stage_tail") or []),
        "timeout_run_id": str(last_diag.get("embedded_run_id") or ""),
        "timeout_session_id": expected_session_id,
    }


def _resolve_case_status(case: dict[str, Any]) -> str:
    raw_status = str(case.get("status") or "").strip().lower()
    if raw_status in {"ok", "timeout", "error"}:
        return raw_status
    if bool(case.get("timed_out")):
        return "timeout"
    if bool(case.get("ok")):
        return "ok"
    return "error"


def _normalize_case_result(case: dict[str, Any], *, case_name: str, tools_label: str) -> dict[str, Any]:
    result = dict(case)
    result["case"] = case_name
    result["name"] = case_name
    result["tools"] = tools_label

    elapsed = result.get("elapsed_ms")
    try:
        elapsed_float = float(elapsed)
    except (TypeError, ValueError):
        elapsed_float = 0.0
    result["elapsed_ms"] = round(elapsed_float, 2)

    if "timed_out" not in result:
        result["timed_out"] = False
    result["status"] = _resolve_case_status(result)

    finish_reason = _resolve_finish_reason(result)
    result["finish_reason"] = finish_reason

    return result


def _report_path_from_args(raw_report_file: str) -> Path:
    trimmed = str(raw_report_file or "").strip()
    if trimmed:
        return Path(trimmed).expanduser()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return DEFAULT_REPORT_DIR / f"timeout_matrix_{stamp}.json"


def _write_report(report_path: Path, report: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = report_path.with_suffix(report_path.suffix + ".tmp")
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    temp_path.write_text(serialized, encoding="utf-8")
    temp_path.replace(report_path)


def _check_http_get(url: str) -> tuple[bool, str]:
    try:
        response = requests.get(
            url,
            timeout=(HTTP_CONNECT_TIMEOUT_SECONDS, HTTP_READ_TIMEOUT_SECONDS),
        )
    except Exception as exc:
        return False, str(exc)
    if not response.ok:
        return False, f"HTTP {response.status_code}: {_truncate(response.text, 180)}"
    return True, ""


def _check_http_post(url: str, payload: dict[str, Any]) -> tuple[bool, str]:
    try:
        response = requests.post(
            url,
            json=payload,
            timeout=(HTTP_CONNECT_TIMEOUT_SECONDS, HTTP_READ_TIMEOUT_SECONDS),
        )
    except Exception as exc:
        return False, str(exc)
    if not response.ok:
        return False, f"HTTP {response.status_code}: {_truncate(response.text, 180)}"
    return True, ""


def _run_preflight(*, server_url: str, ui_url: str) -> dict[str, Any]:
    vllm_url = f"{server_url.rstrip('/')}/v1/models"
    vllm_ok, vllm_error = _check_http_get(vllm_url)

    ui_get_url = f"{ui_url.rstrip('/')}/api/status"
    ui_get_ok, ui_get_error = _check_http_get(ui_get_url)

    ui_post_url = f"{ui_url.rstrip('/')}/api/chat/title"
    ui_post_payload = {"messages": [], "fallback": "preflight"}
    ui_post_ok, ui_post_error = _check_http_post(ui_post_url, ui_post_payload)

    ui_ok = bool(ui_get_ok and ui_post_ok)
    ui_reason = ""
    if not ui_ok:
        if not ui_get_ok:
            ui_reason = f"GET failed: {ui_get_error}"
        elif not ui_post_ok:
            ui_reason = f"POST failed: {ui_post_error}"

    return {
        "vllm": {
            "ok": bool(vllm_ok),
            "url": vllm_url,
            "error": vllm_error or None,
        },
        "openclaw_ui": {
            "ok": ui_ok,
            "get_check": {
                "ok": bool(ui_get_ok),
                "url": ui_get_url,
                "error": ui_get_error or None,
            },
            "post_check": {
                "ok": bool(ui_post_ok),
                "url": ui_post_url,
                "payload": ui_post_payload,
                "error": ui_post_error or None,
            },
            "error": ui_reason or None,
        },
    }


def _build_case_start_line(*, name: str, tools: str, timeout_seconds: int) -> str:
    return f"[case start] name={name} tools={tools} timeout={timeout_seconds}s"


def _build_case_done_line(case: dict[str, Any]) -> str:
    name = str(case.get("case") or "")
    elapsed = float(case.get("elapsed_ms") or 0.0)
    status = _resolve_case_status(case)
    finish_reason = _resolve_finish_reason(case)
    return (
        f"[case done] name={name} elapsed_ms={elapsed:.2f} "
        f"status={status} finish_reason={finish_reason}"
    )


def run_direct_case(
    *,
    case_id: str,
    server_url: str,
    model: str,
    message: str,
    with_tool: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "max_tokens": 256,
        "temperature": 0,
        "stream": False,
    }
    if with_tool:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "tw_echo",
                    "description": "Echo a short text.",
                    "parameters": {
                        "type": "object",
                        "required": ["text"],
                        "properties": {"text": {"type": "string"}},
                    },
                },
            }
        ]
        payload["tool_choice"] = "required"

    started = time.perf_counter()
    result: dict[str, Any] = {
        "case": case_id,
        "mode": "vllm_direct",
        "request": {
            "endpoint": f"{server_url.rstrip('/')}/v1/chat/completions",
            "model": model,
            "with_tool": with_tool,
            "max_tokens": payload["max_tokens"],
            "temperature": payload["temperature"],
            "message_preview": _truncate(message, limit=200),
            "timeout": {
                "connect_seconds": HTTP_CONNECT_TIMEOUT_SECONDS,
                "read_seconds": HTTP_READ_TIMEOUT_SECONDS,
            },
        },
    }
    try:
        response = requests.post(
            f"{server_url.rstrip('/')}/v1/chat/completions",
            json=payload,
            timeout=(HTTP_CONNECT_TIMEOUT_SECONDS, HTTP_READ_TIMEOUT_SECONDS),
        )
    except requests.exceptions.Timeout as exc:
        elapsed = (time.perf_counter() - started) * 1000.0
        result.update(
            {
                "ok": False,
                "timed_out": True,
                "status": "timeout",
                "elapsed_ms": round(elapsed, 2),
                "error": str(exc),
            }
        )
        return result
    except Exception as exc:  # pragma: no cover - network failure
        elapsed = (time.perf_counter() - started) * 1000.0
        result.update(
            {
                "ok": False,
                "timed_out": False,
                "status": "error",
                "elapsed_ms": round(elapsed, 2),
                "error": str(exc),
            }
        )
        return result

    elapsed = (time.perf_counter() - started) * 1000.0
    result["elapsed_ms"] = round(elapsed, 2)
    result["http_status"] = int(response.status_code)

    data, parse_error = _safe_json(response)
    if data is None:
        result.update(
            {
                "ok": False,
                "timed_out": False,
                "status": "error",
                "error": f"json parse failed: {parse_error}",
                "body_preview": _truncate(response.text, limit=240),
            }
        )
        return result

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        result.update(
            {
                "ok": False,
                "timed_out": False,
                "status": "error",
                "error": "response missing choices",
                "response_preview": _truncate(json.dumps(data, ensure_ascii=False), 240),
            }
        )
        return result

    first = choices[0] if isinstance(choices[0], dict) else {}
    message_payload = first.get("message") if isinstance(first, dict) else {}
    if not isinstance(message_payload, dict):
        message_payload = {}
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}

    result.update(
        {
            "ok": bool(response.ok),
            "timed_out": False,
            "status": "ok" if bool(response.ok) else "error",
            "finish_reason": (
                str(first.get("finish_reason", "")).strip().lower()
                if isinstance(first.get("finish_reason"), str)
                else None
            ),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "tool_call_summary": _summarize_tool_calls(message_payload),
        }
    )
    return result


def run_agent_stream_case(
    *,
    case_id: str,
    ui_url: str,
    server_url: str,
    model: str,
    message: str,
    runtime: str,
    gateway_log_path: Path,
    case_deadline_seconds: int,
    tool_mode: str = "full",
    allowed_tools: list[str] | None = None,
    model_first_chunk_timeout_ms: int | None = None,
) -> dict[str, Any]:
    conversation_id = f"diag-{runtime}-{case_id.lower()}-{uuid.uuid4().hex[:8]}"
    expected_session_id = _normalize_agent_session_id(
        conversation_id,
        selected_model=model,
        runtime=runtime,
    )
    expected_run_id = f"{expected_session_id}-run-{uuid.uuid4().hex[:12]}"
    payload = {
        "message": message,
        "history": [],
        "server_url": server_url,
        "model": model,
        "openclaw_enabled": True,
        "agent_runtime": runtime,
        "conversation_id": conversation_id,
        "session_id": expected_session_id,
        "run_id": expected_run_id,
        "max_tokens": 1024,
        "temperature": 0.2,
        "tool_mode": tool_mode,
    }
    if isinstance(allowed_tools, list):
        payload["allowed_tools"] = [str(name).strip() for name in allowed_tools if str(name).strip()]
    if isinstance(model_first_chunk_timeout_ms, int) and model_first_chunk_timeout_ms > 0:
        payload["model_first_chunk_timeout_ms"] = model_first_chunk_timeout_ms

    log_offset = gateway_log_path.stat().st_size if gateway_log_path.exists() else 0
    log_cursor = log_offset
    log_chunks: list[str] = []
    started = time.perf_counter()
    timestamps: dict[str, float] = {}
    thinking_parts: list[str] = []
    answer_parts: list[str] = []
    done_payload: dict[str, Any] | None = None
    error_payload: dict[str, Any] | None = None
    event_counts: dict[str, int] = {}
    parse_errors = 0
    status_code: int | None = None

    result: dict[str, Any] = {
        "case": case_id,
        "mode": f"agent_{runtime}",
        "request": {
            "endpoint": f"{ui_url.rstrip('/')}/api/chat/stream",
            "server_url": server_url,
            "model": model,
            "conversation_id": conversation_id,
            "expected_session_id": expected_session_id,
            "expected_run_id": expected_run_id,
            "message_preview": _truncate(message, limit=240),
            "tool_mode": tool_mode,
            "allowed_tools": payload.get("allowed_tools", []),
            "model_first_chunk_timeout_ms": payload.get("model_first_chunk_timeout_ms"),
            "timeout": {
                "connect_seconds": HTTP_CONNECT_TIMEOUT_SECONDS,
                "read_seconds": HTTP_READ_TIMEOUT_SECONDS,
                "case_deadline_seconds": case_deadline_seconds,
            },
        },
    }

    def _collect_log_delta() -> str:
        nonlocal log_cursor
        delta, log_cursor = _read_log_delta(gateway_log_path, log_cursor)
        if delta:
            log_chunks.append(delta)
        return delta

    def _collect_log_excerpt(limit: int = 800) -> dict[str, Any]:
        _collect_log_delta()
        full_log = "".join(log_chunks)
        diagnostics = _stage_diagnostics(
            log_text=full_log,
            expected_session_id=expected_session_id,
            expected_run_id=expected_run_id,
        )
        return {
            "keyword_hits": _count_keywords(full_log),
            "log_excerpt": _truncate(full_log, limit=limit),
            **diagnostics,
        }

    try:
        response = requests.post(
            f"{ui_url.rstrip('/')}/api/chat/stream",
            json=payload,
            timeout=(HTTP_CONNECT_TIMEOUT_SECONDS, HTTP_READ_TIMEOUT_SECONDS),
            stream=True,
        )
        status_code = int(response.status_code)
        with response:
            if not response.ok:
                body = _truncate(response.text, limit=280)
                result.update(
                    {
                        "ok": False,
                        "timed_out": False,
                        "status": "error",
                        "http_status": status_code,
                        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
                        "error": f"HTTP {status_code}: {body}",
                    }
                )
                result.update(_collect_log_excerpt(limit=600))
                return result

            for raw_line in response.iter_lines(decode_unicode=True):
                elapsed_seconds = time.perf_counter() - started
                if elapsed_seconds > float(case_deadline_seconds):
                    try:
                        response.close()
                    except Exception:
                        pass
                    result.update(
                        {
                            "ok": False,
                            "timed_out": True,
                            "status": "timeout",
                            "http_status": status_code,
                            "elapsed_ms": round(elapsed_seconds * 1000.0, 2),
                            "error": (
                                "OpenClaw/UI stream exceeded hard deadline "
                                f"{case_deadline_seconds}s"
                            ),
                        }
                    )
                    result.update(_collect_log_excerpt(limit=600))
                    result.update(
                        _wait_for_embedded_cleanup(
                            gateway_log_path=gateway_log_path,
                            start_offset=log_cursor,
                            expected_session_id=expected_session_id,
                            expected_run_id=expected_run_id,
                            wait_timeout_seconds=45,
                        )
                    )
                    result.update(_collect_log_excerpt(limit=800))
                    return result

                if raw_line is None:
                    continue
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    parse_errors += 1
                    continue
                if not isinstance(event, dict):
                    continue
                event_type = str(event.get("type", "")).strip()
                if not event_type:
                    continue
                now_ms = (time.perf_counter() - started) * 1000.0
                timestamps.setdefault(event_type, round(now_ms, 2))
                event_counts[event_type] = int(event_counts.get(event_type, 0)) + 1

                if event_type == "thinking_delta":
                    text = str(event.get("text", "") or "")
                    if text:
                        thinking_parts.append(text)
                elif event_type == "answer_delta":
                    text = str(event.get("text", "") or "")
                    if text:
                        answer_parts.append(text)
                elif event_type == "done":
                    done_payload = event
                    break
                elif event_type == "error":
                    error_payload = event
                    break
    except requests.exceptions.Timeout as exc:
        result.update(
            {
                "ok": False,
                "timed_out": True,
                "status": "timeout",
                "http_status": status_code,
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
                "error": str(exc),
            }
        )
        result.update(_collect_log_excerpt(limit=600))
        result.update(
            _wait_for_embedded_cleanup(
                gateway_log_path=gateway_log_path,
                start_offset=log_cursor,
                expected_session_id=expected_session_id,
                expected_run_id=expected_run_id,
                wait_timeout_seconds=45,
            )
        )
        result.update(_collect_log_excerpt(limit=800))
        return result
    except Exception as exc:  # pragma: no cover - network failure
        result.update(
            {
                "ok": False,
                "timed_out": False,
                "status": "error",
                "http_status": status_code,
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
                "error": str(exc),
            }
        )
        result.update(_collect_log_excerpt(limit=600))
        return result

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    thinking_text = "".join(thinking_parts).strip()
    answer_text = "".join(answer_parts).strip()
    combined = "\n".join([thinking_text, answer_text]).strip()
    metrics = done_payload.get("metrics") if isinstance(done_payload, dict) else {}
    if not isinstance(metrics, dict):
        metrics = {}

    retry_flags = {
        key: value
        for key, value in metrics.items()
        if "retry" in str(key).lower() or "repair" in str(key).lower()
    }

    result.update(
        {
            "ok": done_payload is not None and error_payload is None,
            "timed_out": bool(
                error_payload
                and "timed out" in str(error_payload.get("message", "")).lower()
            ),
            "status": (
                "timeout"
                if bool(
                    error_payload
                    and "timed out" in str(error_payload.get("message", "")).lower()
                )
                else ("ok" if done_payload is not None and error_payload is None else "error")
            ),
            "http_status": status_code,
            "elapsed_ms": round(elapsed_ms, 2),
            "event_counts": event_counts,
            "stage_timing_ms": {
                "start": timestamps.get("start"),
                "first_thinking_delta": timestamps.get("thinking_delta"),
                "first_answer_delta": timestamps.get("answer_delta"),
                "done": timestamps.get("done"),
                "error": timestamps.get("error"),
            },
            "stream_parse_errors": parse_errors,
            "thinking_preview": _truncate(thinking_text, limit=280),
            "answer_preview": _truncate(answer_text, limit=280),
            "literal_tool_call_markup": bool(
                re.search(r"(?is)<\s*tool_call\b|\[calling tool:", combined)
            ),
            "finish_reason": (
                str(done_payload.get("finish_reason", "")).strip().lower()
                if isinstance(done_payload, dict)
                and isinstance(done_payload.get("finish_reason"), str)
                else None
            ),
            "stop_reason": (
                str(done_payload.get("stop_reason", "")).strip().lower()
                if isinstance(done_payload, dict)
                and isinstance(done_payload.get("stop_reason"), str)
                else None
            ),
            "prompt_tokens": metrics.get("prompt_tokens"),
            "completion_tokens": metrics.get("completion_tokens"),
            "total_tokens": metrics.get("total_tokens"),
            "tokens_per_second": metrics.get("tokens_per_second"),
            "agent_metrics": metrics,
            "retry_or_repair_flags": retry_flags,
            "error_payload": error_payload,
        }
    )

    if result.get("timed_out"):
        result.update(
            _wait_for_embedded_cleanup(
                gateway_log_path=gateway_log_path,
                start_offset=log_cursor,
                expected_session_id=expected_session_id,
                expected_run_id=expected_run_id,
                wait_timeout_seconds=45,
            )
        )

    result.update(_collect_log_excerpt(limit=800))
    return result


def _is_fast(case: dict[str, Any], threshold_ms: float) -> bool:
    return bool(case.get("ok")) and float(case.get("elapsed_ms") or 0.0) <= threshold_ms


def _is_slow_or_timeout(case: dict[str, Any], threshold_ms: float) -> bool:
    if bool(case.get("timed_out")) or not bool(case.get("ok")):
        return True
    return float(case.get("elapsed_ms") or 0.0) > threshold_ms


def classify_root_cause(cases: dict[str, dict[str, Any]], threshold_ms: float) -> list[str]:
    notes: list[str] = []
    a = cases.get("A", {})
    b = cases.get("B", {})
    d = cases.get("D", {})
    e = cases.get("E", {})

    if _is_fast(a, threshold_ms) and _is_slow_or_timeout(b, threshold_ms):
        notes.append(
            "A快/B慢或超时：优先怀疑 vLLM-MLX tool parser、chat template、tool schema/arguments 处理路径。"
        )
    if _is_fast(a, threshold_ms) and _is_fast(b, threshold_ms) and _is_slow_or_timeout(d, threshold_ms):
        notes.append(
            "A/B都快但D慢或超时：优先怀疑 OpenClaw 请求格式、agent loop、重试链路，而非底层 vLLM 推理速度。"
        )
    if _is_fast(d, threshold_ms) and _is_slow_or_timeout(e, threshold_ms):
        notes.append(
            "D快/E慢或超时：优先怀疑完整工具集过大（schema 太大）或某个工具定义/返回触发 retry/repair 循环。"
        )

    if not notes:
        notes.append("矩阵未触发单一明确模式，需要结合关键词命中和日志分段继续细分。")
    return notes


def _compute_diagnosis(cases: list[dict[str, Any]], threshold_ms: float) -> list[str]:
    cases_by_id = {str(item.get("case")): item for item in cases}
    diagnosis = classify_root_cause(cases_by_id, threshold_ms=threshold_ms)
    if len(cases) < 5:
        return [f"Partial run: only {len(cases)}/5 cases completed.", *diagnosis]
    return diagnosis


def render_table(cases: list[dict[str, Any]]) -> str:
    headers = [
        "Case",
        "OK",
        "TimedOut",
        "Elapsed(ms)",
        "Finish",
        "ToolCalls",
        "PromptTok",
        "CompTok",
        "RetryFlags",
    ]
    rows = [headers]
    for case in cases:
        tool_calls_count = ""
        tool_summary = case.get("tool_call_summary")
        if isinstance(tool_summary, dict):
            tool_calls_count = str(tool_summary.get("tool_calls_count", ""))
        retry_flags = case.get("retry_or_repair_flags")
        retry_text = ""
        if isinstance(retry_flags, dict) and retry_flags:
            retry_text = ",".join(sorted(str(k) for k in retry_flags.keys()))
        rows.append(
            [
                str(case.get("case", "")),
                "Y" if case.get("ok") else "N",
                "Y" if case.get("timed_out") else "N",
                f"{float(case.get('elapsed_ms') or 0.0):.1f}",
                str(case.get("finish_reason") or case.get("stop_reason") or ""),
                tool_calls_count,
                str(case.get("prompt_tokens") or ""),
                str(case.get("completion_tokens") or ""),
                retry_text,
            ]
        )

    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    lines = []
    for row in rows:
        line = " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))
        lines.append(line)
    return "\n".join(lines)


def _build_case_specs(
    *,
    server_url: str,
    ui_url: str,
    model: str,
    runtime: str,
    gateway_log_path: Path,
    case_timeout_seconds: int,
    include_agent_cases: bool,
    model_first_chunk_timeout_ms: int | None,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "name": "A",
            "tools": "none",
            "runner": run_direct_case,
            "kwargs": {
                "case_id": "A",
                "server_url": server_url,
                "model": model,
                "message": "Say 'pong' only.",
                "with_tool": False,
            },
            "timeout_seconds": case_timeout_seconds,
        },
        {
            "name": "B",
            "tools": "single",
            "runner": run_direct_case,
            "kwargs": {
                "case_id": "B",
                "server_url": server_url,
                "model": model,
                "message": "Call tw_echo with text='ok'. Return structured tool call.",
                "with_tool": True,
            },
            "timeout_seconds": case_timeout_seconds,
        },
    ]
    if include_agent_cases:
        specs.extend(
            [
                {
                    "name": "C",
                    "tools": "none",
                    "runner": run_agent_stream_case,
                    "kwargs": {
                        "case_id": "C",
                        "ui_url": ui_url,
                        "server_url": server_url,
                        "model": model,
                        "message": "Who are you? Reply in one short paragraph.",
                        "runtime": runtime,
                        "gateway_log_path": gateway_log_path,
                        "case_deadline_seconds": case_timeout_seconds,
                        "tool_mode": "none",
                        "allowed_tools": [],
                        "model_first_chunk_timeout_ms": model_first_chunk_timeout_ms,
                    },
                    "timeout_seconds": case_timeout_seconds,
                },
                {
                    "name": "D",
                    "tools": "single",
                    "runner": run_agent_stream_case,
                    "kwargs": {
                        "case_id": "D",
                        "ui_url": ui_url,
                        "server_url": server_url,
                        "model": model,
                        "message": (
                            "Use exactly one exec tool call to run `echo tool_ok`, "
                            "then return only the command output."
                        ),
                        "runtime": runtime,
                        "gateway_log_path": gateway_log_path,
                        "case_deadline_seconds": case_timeout_seconds,
                        "tool_mode": "single",
                        "allowed_tools": ["exec"],
                        "model_first_chunk_timeout_ms": model_first_chunk_timeout_ms,
                    },
                    "timeout_seconds": case_timeout_seconds,
                },
                {
                    "name": "E",
                    "tools": "full",
                    "runner": run_agent_stream_case,
                    "kwargs": {
                        "case_id": "E",
                        "ui_url": ui_url,
                        "server_url": server_url,
                        "model": model,
                        "message": (
                            "Search the web for Steve Jobs, verify from at least two sources, "
                            "then summarize key facts."
                        ),
                        "runtime": runtime,
                        "gateway_log_path": gateway_log_path,
                        "case_deadline_seconds": case_timeout_seconds,
                        "tool_mode": "full",
                        "allowed_tools": [],
                        "model_first_chunk_timeout_ms": model_first_chunk_timeout_ms,
                    },
                    "timeout_seconds": case_timeout_seconds,
                },
            ]
        )
    return specs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Minimal reproduction matrix for timeout root-cause diagnosis.",
    )
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--ui-url", default="http://127.0.0.1:7862")
    parser.add_argument("--model", required=True, help="Model ID served by vllm-mlx.")
    parser.add_argument("--runtime", default="openclaw", choices=("openclaw", "hermes"))
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=180,
        help="Requested timeout hint (OpenClaw/UI hard capped to 90s).",
    )
    parser.add_argument("--slow-threshold-ms", type=float, default=35000.0)
    parser.add_argument(
        "--gateway-log-path",
        default="integrations/openclaw/.token-workshed-state/gateway.log",
        help="OpenClaw gateway log file used for keyword delta scan.",
    )
    parser.add_argument("--report-file", default="")
    parser.add_argument(
        "--model-first-chunk-timeout-ms",
        type=int,
        default=None,
        help=(
            "Override embedded first chunk timeout for agent cases. "
            "If omitted, reads OPENCLAW_MODEL_FIRST_CHUNK_TIMEOUT_MS from env."
        ),
    )
    args = parser.parse_args()

    server_url = str(args.server_url).rstrip("/")
    ui_url = str(args.ui_url).rstrip("/")
    gateway_log_path = Path(args.gateway_log_path).expanduser()
    model_first_chunk_timeout_ms = args.model_first_chunk_timeout_ms
    if model_first_chunk_timeout_ms is None:
        raw_first_chunk_timeout = os.environ.get("OPENCLAW_MODEL_FIRST_CHUNK_TIMEOUT_MS", "").strip()
        if raw_first_chunk_timeout:
            try:
                parsed_first_chunk_timeout = int(raw_first_chunk_timeout)
            except ValueError:
                parsed_first_chunk_timeout = 0
            if parsed_first_chunk_timeout > 0:
                model_first_chunk_timeout_ms = parsed_first_chunk_timeout

    # Hard cap per requirement/assumption: agent-side cases never exceed 90s.
    _ = max(1, int(args.timeout_seconds or 0))
    effective_case_timeout_seconds = OPENCLAW_UI_CASE_DEADLINE_SECONDS

    report_path = _report_path_from_args(str(args.report_file or ""))
    emit_json_to_stdout = not bool(str(args.report_file or "").strip())

    cases: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "status": "running",
        "reason": None,
        "started_at": _timestamp_now(),
        "ended_at": None,
        "server_url": server_url,
        "ui_url": ui_url,
        "model": args.model,
        "runtime": args.runtime,
        "model_first_chunk_timeout_ms": model_first_chunk_timeout_ms,
        "slow_threshold_ms": float(args.slow_threshold_ms),
        "preflight": {},
        "skipped_cases": [],
        "cases": [],
        "diagnosis": [],
        "error": None,
    }
    _write_report(report_path, report)

    finalized = False

    try:
        preflight = _run_preflight(server_url=server_url, ui_url=ui_url)
        report["preflight"] = preflight
        _write_report(report_path, report)

        vllm_ok = bool((preflight.get("vllm") or {}).get("ok"))
        if not vllm_ok:
            report["status"] = "environment_not_ready"
            report["reason"] = "vllm_server_unreachable"
            report["ended_at"] = _timestamp_now()
            report["error"] = (
                str((preflight.get("vllm") or {}).get("error") or "").strip() or None
            )
            report["cases"] = cases
            report["diagnosis"] = [
                "Preflight failed: vLLM server unreachable, matrix aborted before case execution."
            ]
            _write_report(report_path, report)
            finalized = True
            print(
                "[preflight] status=environment_not_ready reason=vllm_server_unreachable",
                flush=True,
            )
            print(f"Saved report: {report_path}", flush=True)
            if emit_json_to_stdout:
                print("\nJSON report:", flush=True)
                print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
            return 2

        ui_ok = bool((preflight.get("openclaw_ui") or {}).get("ok"))
        if not ui_ok:
            report["status"] = "openclaw_ui_unreachable"
            report["reason"] = "openclaw_ui_unreachable"
            report["skipped_cases"] = ["C", "D", "E"]
            report["error"] = (
                str((preflight.get("openclaw_ui") or {}).get("error") or "").strip() or None
            )
            _write_report(report_path, report)
            print(
                "[preflight] status=openclaw_ui_unreachable skip_cases=C,D,E",
                flush=True,
            )

        specs = _build_case_specs(
            server_url=server_url,
            ui_url=ui_url,
            model=args.model,
            runtime=args.runtime,
            gateway_log_path=gateway_log_path,
            case_timeout_seconds=effective_case_timeout_seconds,
            include_agent_cases=ui_ok,
            model_first_chunk_timeout_ms=model_first_chunk_timeout_ms,
        )

        for spec in specs:
            name = str(spec["name"])
            tools = str(spec["tools"])
            timeout_seconds = int(spec["timeout_seconds"])
            print(
                _build_case_start_line(
                    name=name,
                    tools=tools,
                    timeout_seconds=timeout_seconds,
                ),
                flush=True,
            )

            case_started = time.perf_counter()
            runner = spec["runner"]
            kwargs = dict(spec.get("kwargs") or {})

            try:
                raw_case = runner(**kwargs)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - case_started) * 1000.0
                raw_case = {
                    "case": name,
                    "ok": False,
                    "timed_out": False,
                    "status": "error",
                    "elapsed_ms": round(elapsed_ms, 2),
                    "error": f"Unhandled case exception: {exc}",
                }

            if "elapsed_ms" not in raw_case:
                raw_case["elapsed_ms"] = round((time.perf_counter() - case_started) * 1000.0, 2)

            case_result = _normalize_case_result(raw_case, case_name=name, tools_label=tools)
            print(_build_case_done_line(case_result), flush=True)

            cases.append(case_result)
            report["cases"] = cases
            report["diagnosis"] = _compute_diagnosis(cases, threshold_ms=float(args.slow_threshold_ms))
            _write_report(report_path, report)

        if ui_ok:
            report["status"] = "completed"
            report["reason"] = None
            report["error"] = None
        else:
            report["status"] = "openclaw_ui_unreachable"
            report["reason"] = "openclaw_ui_unreachable"
        report["ended_at"] = _timestamp_now()
        report["cases"] = cases
        report["diagnosis"] = _compute_diagnosis(cases, threshold_ms=float(args.slow_threshold_ms))
        _write_report(report_path, report)
        finalized = True

        print(render_table(cases), flush=True)
        print("\nDiagnosis:", flush=True)
        for note in report["diagnosis"]:
            print(f"- {note}", flush=True)

        print(f"\nSaved report: {report_path}", flush=True)
        if emit_json_to_stdout:
            print("\nJSON report:", flush=True)
            print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

        return 0

    except KeyboardInterrupt:
        report["status"] = "interrupted"
        report["reason"] = "interrupted"
        report["ended_at"] = _timestamp_now()
        report["error"] = "KeyboardInterrupt"
        report["cases"] = cases
        report["diagnosis"] = _compute_diagnosis(cases, threshold_ms=float(args.slow_threshold_ms))
        _write_report(report_path, report)
        finalized = True
        print("\nInterrupted. Partial report flushed.", flush=True)
        print(f"Saved report: {report_path}", flush=True)
        return 130
    except Exception as exc:
        report["status"] = "failed"
        report["reason"] = "exception"
        report["ended_at"] = _timestamp_now()
        report["error"] = str(exc)
        report["cases"] = cases
        report["diagnosis"] = _compute_diagnosis(cases, threshold_ms=float(args.slow_threshold_ms))
        _write_report(report_path, report)
        finalized = True
        print(f"\nFatal error: {exc}", flush=True)
        print(f"Saved report: {report_path}", flush=True)
        return 1
    finally:
        if not finalized:
            report.setdefault("ended_at", _timestamp_now())
            report["cases"] = cases
            report["diagnosis"] = _compute_diagnosis(cases, threshold_ms=float(args.slow_threshold_ms))
            if report.get("status") == "running":
                report["status"] = "failed"
                report["reason"] = report.get("reason") or "unspecified_failure"
                report["error"] = report.get("error") or "Unspecified failure"
            _write_report(report_path, report)


if __name__ == "__main__":
    raise SystemExit(main())
