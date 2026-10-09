"""Strict tool-calling smoke tests."""

from __future__ import annotations

import json
import re
from typing import Any

import requests

_LITERAL_TOOL_CALL_RE = re.compile(
    r"<tool_call|</tool_call>|\[calling tool:", re.IGNORECASE
)


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
    return ""


def _required_tool_choice_rejected(response: Any) -> bool:
    """Return whether the server rejected only the strict choice mode.

    A few OpenAI-compatible servers implement structured tool calls but do not
    implement ``tool_choice=required``.  That is a recoverable compatibility
    issue: retrying with ``auto`` still lets the model decide whether to emit a
    structured call, while the parser/argument checks below remain strict.
    """
    try:
        status = int(getattr(response, "status_code", 0) or 0)
    except (TypeError, ValueError):
        status = 0
    if status not in {400, 422}:
        return False
    body = str(getattr(response, "text", "") or "").lower()
    mentions_choice = "tool_choice" in body or "tool choice" in body
    if not mentions_choice:
        return False
    return any(
        marker in body
        for marker in (
            "required",
            "unsupported",
            "not support",
            "invalid",
            "unrecognized",
        )
    )


def _plain_chat_fallback(
    *,
    endpoint: str,
    model: str,
    timeout_seconds: float,
) -> tuple[bool, str]:
    """Check that the model can still answer without tool schemas.

    A model can be perfectly usable for an agent conversation while not
    supporting OpenAI's structured ``tool_calls`` contract.  Configure should
    distinguish that degraded capability from a dead model/server instead of
    turning every parser mismatch into a hard failure.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the word: ok"}],
        "temperature": 0.0,
        "max_tokens": 32,
        "stream": False,
    }
    try:
        response = requests.post(
            endpoint,
            json=payload,
            timeout=(3.0, max(4.0, float(timeout_seconds))),
        )
    except Exception as exc:
        return False, f"plain-chat fallback request failed: {exc}"
    if not response.ok:
        body_preview = response.text[:220] if isinstance(response.text, str) else ""
        detail = f"HTTP {response.status_code}"
        if body_preview:
            detail = f"{detail}: {body_preview}"
        return False, f"plain-chat fallback failed ({detail})"
    try:
        data = response.json()
    except Exception as exc:
        return False, f"plain-chat fallback returned non-JSON data: {exc}"
    choices = data.get("choices") if isinstance(data, dict) else None
    first = choices[0] if isinstance(choices, list) and choices else None
    message = first.get("message") if isinstance(first, dict) else None
    if not isinstance(message, dict):
        return False, "plain-chat fallback returned no assistant message"
    text = _flatten_content(message.get("content"))
    if not text.strip():
        return False, "plain-chat fallback returned an empty assistant message"
    return True, ""


def _degraded_text_result(
    *,
    reason: str,
    fallback_reason: str = "",
) -> dict[str, Any]:
    """Build a successful-but-limited tool capability result."""
    warning = str(reason or "Structured tool calls were not observed.").strip()
    if fallback_reason:
        warning = f"{warning} {fallback_reason.strip()}".strip()
    return {
        "ok": True,
        "reason": warning,
        "tool_calls": 0,
        "tool_calling": "degraded",
        "degraded": True,
        "warnings": [warning],
        "failure_category": "tool_call_compatibility",
    }


def _try_plain_chat_degraded(
    *,
    allow_text_fallback: bool,
    endpoint: str,
    model: str,
    timeout_seconds: float,
    reason: str,
) -> dict[str, Any] | None:
    """Turn a structural probe failure into a limited profile when chat works."""
    if not allow_text_fallback or not endpoint or not model:
        return None
    fallback_ok, fallback_reason = _plain_chat_fallback(
        endpoint=endpoint,
        model=model,
        timeout_seconds=timeout_seconds,
    )
    if not fallback_ok:
        return None
    return _degraded_text_result(reason=reason, fallback_reason=fallback_reason)


def _parse_smoke_response(
    response: Any,
    *,
    allow_text_fallback: bool = False,
    endpoint: str = "",
    model: str = "",
    timeout_seconds: float = 20.0,
) -> dict[str, Any]:
    """Validate one completion response and return a normalized smoke result.

    ``allow_text_fallback`` is intentionally opt-in for the low-level helper.
    The Configure path enables it, while callers that need a strict structured
    tool-call assertion can keep the historical behaviour.
    """
    if not response.ok:
        body_preview = response.text[:220] if isinstance(response.text, str) else ""
        recovered = _try_plain_chat_degraded(
            allow_text_fallback=allow_text_fallback,
            endpoint=endpoint,
            model=model,
            timeout_seconds=timeout_seconds,
            reason=(
                f"Tool-call probe HTTP {response.status_code}; "
                "model remains usable for plain agent responses."
            ),
        )
        if recovered is not None:
            return recovered
        return {
            "ok": False,
            "reason": f"Smoke HTTP {response.status_code}: {body_preview}",
            "tool_calls": 0,
            "tool_calling": "failed",
        }
    try:
        data = response.json()
    except Exception as exc:
        recovered = _try_plain_chat_degraded(
            allow_text_fallback=allow_text_fallback,
            endpoint=endpoint,
            model=model,
            timeout_seconds=timeout_seconds,
            reason="Tool-call probe returned non-JSON data; plain chat remains usable.",
        )
        if recovered is not None:
            return recovered
        return {
            "ok": False,
            "reason": f"Smoke response is not JSON: {exc}",
            "tool_calls": 0,
            "tool_calling": "failed",
        }

    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        recovered = _try_plain_chat_degraded(
            allow_text_fallback=allow_text_fallback,
            endpoint=endpoint,
            model=model,
            timeout_seconds=timeout_seconds,
            reason="Tool-call probe returned no choices; plain chat remains usable.",
        )
        if recovered is not None:
            return recovered
        return {
            "ok": False,
            "reason": "Smoke response missing choices.",
            "tool_calls": 0,
            "tool_calling": "failed",
        }
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first, dict) else {}
    if not isinstance(message, dict):
        recovered = _try_plain_chat_degraded(
            allow_text_fallback=allow_text_fallback,
            endpoint=endpoint,
            model=model,
            timeout_seconds=timeout_seconds,
            reason="Tool-call probe returned no assistant message; plain chat remains usable.",
        )
        if recovered is not None:
            return recovered
        return {
            "ok": False,
            "reason": "Smoke response missing assistant message.",
            "tool_calls": 0,
            "tool_calling": "failed",
        }

    tool_calls = message.get("tool_calls")
    legacy_function_call = False
    if not isinstance(tool_calls, list) or not tool_calls:
        # Some older OpenAI-compatible servers still expose the pre-1.0
        # ``function_call`` field. Normalize it so Configure does not reject
        # a model whose structured contract is otherwise usable.
        legacy = message.get("function_call")
        if isinstance(legacy, dict) and legacy:
            tool_calls = [{"function": legacy}]
            legacy_function_call = True
    if not isinstance(tool_calls, list) or not tool_calls:
        text = _flatten_content(message.get("content"))
        if _LITERAL_TOOL_CALL_RE.search(text):
            if allow_text_fallback and endpoint and model:
                fallback_ok, fallback_reason = _plain_chat_fallback(
                    endpoint=endpoint,
                    model=model,
                    timeout_seconds=timeout_seconds,
                )
                if fallback_ok:
                    return _degraded_text_result(
                        reason=(
                            "Model emitted literal tool-call markup instead of "
                            "structured tool_calls."
                        ),
                        fallback_reason=fallback_reason,
                    )
            return {
                "ok": False,
                "reason": "Model emitted literal <tool_call> markup instead of structured tool_calls.",
                "tool_calls": 0,
                "tool_calling": "failed",
            }
        if text.strip():
            if allow_text_fallback and endpoint and model:
                return _degraded_text_result(
                    reason="Model returned plain text instead of structured tool_calls."
                )
            return {
                "ok": False,
                "reason": "Model returned plain text instead of structured tool_calls.",
                "tool_calls": 0,
                "tool_calling": "failed",
            }
        if allow_text_fallback and endpoint and model:
            fallback_ok, fallback_reason = _plain_chat_fallback(
                endpoint=endpoint,
                model=model,
                timeout_seconds=timeout_seconds,
            )
            if fallback_ok:
                return _degraded_text_result(
                    reason="Model returned an empty tool-call response.",
                    fallback_reason=fallback_reason,
                )
        return {
            "ok": False,
            "reason": "Model returned empty assistant message.",
            "tool_calls": 0,
            "tool_calling": "failed",
        }

    first_call = tool_calls[0] if isinstance(tool_calls[0], dict) else {}
    function = first_call.get("function") if isinstance(first_call, dict) else {}
    if isinstance(function, dict):
        raw_args = function.get("arguments")
    else:
        raw_args = first_call.get("arguments") if isinstance(first_call, dict) else None
    if raw_args is None:
        recovered = _try_plain_chat_degraded(
            allow_text_fallback=allow_text_fallback,
            endpoint=endpoint,
            model=model,
            timeout_seconds=timeout_seconds,
            reason="Model returned a tool call without arguments; tool use is limited.",
        )
        if recovered is not None:
            return recovered
        return {
            "ok": False,
            "reason": "tool_calls arguments missing.",
            "tool_calls": len(tool_calls),
            "tool_calling": "failed",
        }
    if isinstance(raw_args, dict):
        args_payload = raw_args
    elif isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except Exception as exc:
            recovered = _try_plain_chat_degraded(
                allow_text_fallback=allow_text_fallback,
                endpoint=endpoint,
                model=model,
                timeout_seconds=timeout_seconds,
                reason="Model returned invalid JSON tool arguments; tool use is limited.",
            )
            if recovered is not None:
                return recovered
            return {
                "ok": False,
                "reason": f"tool_calls arguments are not valid JSON: {exc}",
                "tool_calls": len(tool_calls),
                "tool_calling": "failed",
            }
        if not isinstance(parsed, dict):
            recovered = _try_plain_chat_degraded(
                allow_text_fallback=allow_text_fallback,
                endpoint=endpoint,
                model=model,
                timeout_seconds=timeout_seconds,
                reason="Model returned non-object tool arguments; tool use is limited.",
            )
            if recovered is not None:
                return recovered
            return {
                "ok": False,
                "reason": "tool_calls arguments JSON is not an object.",
                "tool_calls": len(tool_calls),
                "tool_calling": "failed",
            }
        args_payload = parsed
    else:
        recovered = _try_plain_chat_degraded(
            allow_text_fallback=allow_text_fallback,
            endpoint=endpoint,
            model=model,
            timeout_seconds=timeout_seconds,
            reason="Model returned an unsupported tool-argument shape; tool use is limited.",
        )
        if recovered is not None:
            return recovered
        return {
            "ok": False,
            "reason": "tool_calls arguments are neither JSON object nor JSON string.",
            "tool_calls": len(tool_calls),
            "tool_calling": "failed",
        }

    return {
        "ok": True,
        "reason": "",
        "tool_calls": len(tool_calls),
        "tool_calling": "passed",
        "tool_call_format": "function_call" if legacy_function_call else "tool_calls",
        "parsed_arguments": args_payload,
    }


def run_tool_call_smoke(
    *,
    base_url: str,
    model: str,
    timeout_seconds: float,
    allow_text_fallback: bool = False,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Call the provided function once. Return only structured tool_calls.",
            },
            {"role": "user", "content": "Call tw_smoke_tool with value='ok'."},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "tw_smoke_tool",
                    "description": "Smoke test tool.",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                },
            }
        ],
        "tool_choice": "required",
        "temperature": 0.0,
        # Thinking models can consume a small budget before emitting a tool
        # call.  Keep this bounded, but leave enough room for the parser probe.
        "max_tokens": 512,
        "stream": False,
    }
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            json=payload,
            timeout=(3.0, timeout_seconds),
        )
    except Exception as exc:
        return {
            "ok": False,
            "reason": f"Smoke request failed: {exc}",
            "tool_calls": 0,
            "tool_calling": "failed",
        }
    recovered_with_auto = False
    if _required_tool_choice_rejected(response):
        retry_payload = dict(payload)
        retry_payload["tool_choice"] = "auto"
        try:
            response = requests.post(
                f"{base_url.rstrip('/')}/v1/chat/completions",
                json=retry_payload,
                timeout=(3.0, timeout_seconds),
            )
            recovered_with_auto = True
        except Exception as exc:
            return {
                "ok": False,
                "reason": f"Smoke retry with tool_choice=auto failed: {exc}",
                "tool_calls": 0,
                "tool_calling": "failed",
                "recovery_attempted": True,
            }

    result = _parse_smoke_response(
        response,
        allow_text_fallback=allow_text_fallback,
        endpoint=f"{base_url.rstrip('/')}/v1/chat/completions",
        model=model,
        timeout_seconds=timeout_seconds,
    )
    if recovered_with_auto:
        result["recovery_attempted"] = True
        result["tool_choice_fallback"] = "auto"
        if result.get("ok") and result.get("tool_calling") == "passed":
            result["reason"] = ""
    return result
