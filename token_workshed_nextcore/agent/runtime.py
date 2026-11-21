"""Tool-calling loop for the NextCore app, backed only by SGLang."""

from __future__ import annotations

import base64
import binascii
import concurrent.futures
import io
import json
import queue
import re
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests
from PIL import Image, UnidentifiedImageError

from .audit import audit_tool_call
from .mcp_runtime import MCPRuntime
from .python_sandbox import execute_python
from .store import AgentStateStore
from .tools import (
    SUPPORTED_TOOL_NAMES,
    execute_tool,
    interrupt_computer_use,
    interrupt_web_requests,
    tool_definitions,
)

MAX_AGENT_STEPS = 16
MAX_TOOL_CALLS_PER_STEP = 16
MAX_TOOL_CALLS_PER_RUN = 64
MAX_TOOL_RESULT_CHARS = 32_000
POST_TOOL_SUMMARY_TIMEOUT_SECONDS = 90
MAX_POST_TOOL_SUMMARY_TOKENS = 512
MAX_EMPTY_RESPONSE_RETRIES = 0
MAX_THINKING_PREFILL_RETRIES = 2
MAX_POST_TOOL_EMPTY_RECOVERIES = 1
MAX_CONTEXT_BUDGET_RETRIES = 14
MAX_CHAT_CAPTURE_BYTES = 8 * 1024 * 1024
MAX_CHAT_CAPTURE_DIMENSIONS = (2560, 1600)
MAX_CHAT_CAPTURE_PIXELS = 60_000_000


def _is_safe_citation_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme.lower() in {"http", "https"}
        and bool(parsed.hostname)
        and not parsed.username
        and not parsed.password
        and not any(character.isspace() for character in value)
    )

_TOOL_LABELS = {
    "web_search": "Searching the web",
    "web_fetch": "Reading a webpage",
    "terminal": "Running a command",
    "file": "Working with files",
    "computer_use": "Operating the computer",
    "browser": "Using the browser",
}


class _ThinkTagFilter:
    """Stream only text outside complete or partial ``<think>`` regions."""

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._inside_think = False

    @staticmethod
    def _partial_suffix_length(value: str, token: str) -> int:
        # The delimiters are ASCII, and lower() preserves string offsets while
        # casefold() can expand unrelated Unicode characters before a match.
        lowered = value.lower()
        token = token.lower()
        for length in range(min(len(lowered), len(token) - 1), 0, -1):
            if lowered.endswith(token[:length]):
                return length
        return 0

    def feed(self, text: str, *, final: bool = False) -> str:
        self._buffer += text
        visible: list[str] = []
        while self._buffer:
            lowered = self._buffer.lower()
            if self._inside_think:
                close_at = lowered.find(self._CLOSE)
                if close_at >= 0:
                    self._buffer = self._buffer[close_at + len(self._CLOSE) :]
                    self._inside_think = False
                    continue
                if final:
                    self._buffer = ""
                    break
                keep = self._partial_suffix_length(self._buffer, self._CLOSE)
                self._buffer = self._buffer[-keep:] if keep else ""
                break

            open_at = lowered.find(self._OPEN)
            if open_at >= 0:
                visible.append(self._buffer[:open_at])
                self._buffer = self._buffer[open_at + len(self._OPEN) :]
                self._inside_think = True
                continue
            if final:
                visible.append(self._buffer)
                self._buffer = ""
                break
            keep = self._partial_suffix_length(self._buffer, self._OPEN)
            if keep:
                visible.append(self._buffer[:-keep])
                self._buffer = self._buffer[-keep:]
            else:
                visible.append(self._buffer)
                self._buffer = ""
            break
        return "".join(visible)


class AgentCancelledError(RuntimeError):
    pass


class SGLangContextWindowError(RuntimeError):
    """SGLang rejected a request because the prompt/output budget exceeded context."""


def _response_error_text(response: requests.Response) -> str:
    try:
        payload = response.json()
    except (ValueError, requests.RequestException):
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            detail = error.get("message") or error.get("detail")
        else:
            detail = error or payload.get("message") or payload.get("detail")
        if detail:
            return str(detail).strip()[:1600]
    try:
        return str(response.text or "").strip()[:1600]
    except requests.RequestException:
        return ""


def _looks_like_context_window_error(detail: str) -> bool:
    normalized = str(detail or "").casefold()
    has_context_reference = "context" in normalized or "token count" in normalized
    has_limit_signal = any(
        marker in normalized
        for marker in (
            "exceed",
            "longer than",
            "too long",
            "overflow",
            "maximum",
            "max context",
            "context_length_exceeded",
        )
    )
    return has_context_reference and has_limit_signal


def _looks_like_prompt_over_context_error(detail: str) -> bool:
    normalized = str(detail or "").casefold()
    return bool(
        re.search(
            r"\binput\b.*\bis longer than\b.*\bcontext length\b",
            normalized,
        )
    )


def _search_status_only(answer: str) -> bool:
    """Catch terse status acknowledgements that do not report search findings."""
    normalized = re.sub(r"[\s,，。.!！?？:：;；]+", "", str(answer or "")).casefold()
    return normalized in {
        "搜索完成",
        "搜索已完成",
        "搜索已经完成",
        "联网搜索完成",
        "联网搜索已完成",
        "已搜索",
        "已搜索完成",
        "我搜索了",
        "我已经搜索了",
        "searchcomplete",
        "searchiscomplete",
        "searchcompleted",
        "searchiscompleted",
        "searchdone",
        "searchisdone",
        "searchfinished",
        "searchisfinished",
        "thesearchiscomplete",
        "thesearchisdone",
        "isearched",
        "isearchedtheweb",
    }


def _parse_tool_arguments(raw_arguments: Any, *, tool_name: str) -> dict[str, Any]:
    """Decode a tool argument object, including SGLang's JSON-string wrapper."""
    if isinstance(raw_arguments, dict):
        return raw_arguments

    encoded = str(raw_arguments or "{}")
    try:
        parsed: Any = json.loads(encoded)
        # Some parser/model combinations wrap the object in a JSON string.
        # Decode one additional layer, but never coerce arrays or scalars into
        # tool arguments.
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"SGLang returned invalid arguments for {tool_name or 'tool'}."
        ) from exc

    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"SGLang returned non-object arguments for {tool_name or 'tool'}."
        )
    return parsed


class AgentRunRegistry:
    """Cancellation events keyed by the app generated chat run id."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, threading.Event] = {}

    def register(self, run_id: str) -> threading.Event:
        with self._lock:
            previous = self._runs.get(run_id)
            if previous is not None:
                previous.set()
            event = threading.Event()
            self._runs[run_id] = event
            return event

    def cancel(self, run_id: str) -> bool:
        with self._lock:
            event = self._runs.get(run_id)
            if event is None:
                return False
            event.set()
            return True

    def unregister(self, run_id: str, event: threading.Event) -> None:
        with self._lock:
            if self._runs.get(run_id) is event:
                self._runs.pop(run_id, None)

    def cancel_all(self) -> None:
        with self._lock:
            for event in self._runs.values():
                event.set()


class SGLangAgent:
    def __init__(
        self,
        *,
        server_url: str,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        app_data_dir: Path,
        run_id: str,
        tool_parser: str | None,
        cancel_event: threading.Event,
        reasoning_parser: str | None = None,
        required_tool: str | None = None,
        required_tool_arguments: dict[str, Any] | None = None,
        stream_visible_answer: bool = False,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        answer_sanitizer: Callable[[str], str] | None = None,
        allowed_tools: set[str] | None = None,
        mcp_runtime: MCPRuntime | None = None,
        delegate_depth: int = 0,
        authorization_scope: dict[str, Any] | None = None,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.model = model
        self.messages = [dict(item) for item in messages]
        self.max_tokens = max(1, min(int(max_tokens), 16_384))
        self.effective_max_tokens = self.max_tokens
        self.temperature = max(0.0, min(float(temperature), 2.0))
        self.app_data_dir = app_data_dir
        self.state_store = AgentStateStore(app_data_dir)
        self.allowed_tools = set(allowed_tools) if allowed_tools is not None else None
        self.mcp_runtime = mcp_runtime
        self.delegate_depth = max(0, min(int(delegate_depth), 1))
        self.authorization_scope = dict(authorization_scope or {})
        self.run_id = run_id
        self.tool_parser = str(tool_parser or "").strip().lower()
        self.reasoning_parser = str(reasoning_parser or "").strip().lower()
        self.required_tool = str(required_tool or "").strip() or None
        self.required_tool_arguments = dict(required_tool_arguments or {})
        self._single_tool_run = self.required_tool is not None
        self._capture_only_run = self.required_tool == "computer_use"
        if self.required_tool:
            if self.required_tool not in SUPPORTED_TOOL_NAMES:
                raise ValueError("The requested deterministic Agent tool is not supported.")
            if self._capture_only_run:
                if self.required_tool_arguments.get("action") != "capture":
                    raise ValueError("A forced computer-use request must be a read-only capture.")
                if self.required_tool_arguments.get("mode") != "vision":
                    raise ValueError("A forced screen capture must use vision mode.")
                self.required_tool_arguments = {
                    "action": "capture",
                    "mode": "vision",
                    **(
                        {"app": str(self.required_tool_arguments["app"]).strip()}
                        if self.required_tool_arguments.get("app")
                        else {}
                    ),
                }
        self.cancel_event = cancel_event
        self.stream_visible_answer = bool(stream_visible_answer)
        self.on_event = on_event or (lambda _event: None)
        self.answer_sanitizer = answer_sanitizer or (lambda text: text.strip())
        self.session = requests.Session()
        self._mcp_tools = mcp_runtime.tools() if mcp_runtime is not None else []
        self._tool_definitions = tool_definitions(self._mcp_tools, self.allowed_tools)
        if self.delegate_depth >= 1:
            self._tool_definitions = [item for item in self._tool_definitions if item.get("function", {}).get("name") != "delegate"]
        self.steps = 0
        self.tool_calls = 0
        self.tool_time_ms = 0.0
        self.streamed_answer: list[str] = []
        self.first_token_ms: float | None = None
        self.started_at = time.perf_counter()

    @property
    def tools_available(self) -> bool:
        return bool(self.tool_parser and self._tool_definitions)

    def run(self) -> dict[str, Any]:
        if not self.messages or self.messages[-1].get("role") != "user":
            raise ValueError("The Agent requires a final user message.")
        forced_tool_call = None
        if self.required_tool:
            forced_tool_call = {
                "id": f"nextcore-tool-{uuid.uuid4().hex}",
                "type": "function",
                "function": {
                    "name": self.required_tool,
                    "arguments": json.dumps(self.required_tool_arguments, ensure_ascii=False),
                },
            }
        force_final_answer = False
        empty_response_retries = 0
        thinking_prefill_retries = 0
        post_tool_recovery_attempted = False
        context_budget_retries = 0
        tool_rounds = 0
        max_model_turns = (
            MAX_AGENT_STEPS
            + 1
            + MAX_EMPTY_RESPONSE_RETRIES
            + MAX_THINKING_PREFILL_RETRIES
            + MAX_POST_TOOL_EMPTY_RECOVERIES
            + MAX_CONTEXT_BUDGET_RETRIES
        )
        for _ in range(max_model_turns):
            self._check_cancelled()
            if not force_final_answer and (
                tool_rounds >= MAX_AGENT_STEPS
                or self.tool_calls >= MAX_TOOL_CALLS_PER_RUN
            ):
                force_final_answer = True
                self.messages.append({
                    "role": "user",
                    "content": self._tool_budget_nudge(),
                })
                self.on_event({
                    "type": "tool_status",
                    "tool": "assistant",
                    "status": "running",
                    "message": self._tool_budget_status(),
                })
            if forced_tool_call is not None:
                response = {"content": "", "tool_calls": [forced_tool_call], "usage": {}}
                forced_tool_call = None
            else:
                if self.tool_calls:
                    self.on_event({
                        "type": "tool_status",
                        "tool": "assistant",
                        "status": "running",
                        "message": self._tool_summary_status(),
                    })
                try:
                    response = self._complete(force_final_answer=force_final_answer)
                except AgentCancelledError:
                    raise
                except SGLangContextWindowError as exc:
                    if _looks_like_prompt_over_context_error(str(exc)):
                        fallback = self._tool_result_fallback()
                        if fallback:
                            response = {
                                "content": fallback,
                                "tool_calls": [],
                                "usage": {},
                                "tool_result_fallback": True,
                            }
                        else:
                            raise RuntimeError(
                                "The full conversation itself exceeds the context window available in this SGLang model. "
                                "NextCore preserved the complete history and did not truncate it. "
                                "Start a new conversation or use a model configured with a larger context window."
                            ) from exc
                    elif (
                        context_budget_retries < MAX_CONTEXT_BUDGET_RETRIES
                        and self.effective_max_tokens > 1
                    ):
                        previous_budget = self.effective_max_tokens
                        self.effective_max_tokens = max(1, previous_budget // 2)
                        context_budget_retries += 1
                        self.on_event({
                            "type": "tool_status",
                            "tool": "assistant",
                            "status": "running",
                            "message": self._context_retry_status(previous_budget),
                        })
                        continue
                    else:
                        fallback = self._tool_result_fallback()
                        if fallback:
                            response = {
                                "content": fallback,
                                "tool_calls": [],
                                "usage": {},
                                "tool_result_fallback": True,
                            }
                        else:
                            raise RuntimeError(
                                "SGLang has no output-token headroom left for the full conversation. "
                                "NextCore preserved the complete history and did not truncate it. "
                                "Start a new conversation or use a model configured with a larger context window."
                            ) from exc
                except Exception:
                    fallback = self._tool_result_fallback()
                    if not fallback:
                        raise
                    response = {
                        "content": fallback,
                        "tool_calls": [],
                        "usage": {},
                        "tool_result_fallback": True,
                    }
            self.steps += 1
            tool_calls = response.get("tool_calls") or []
            if self._single_tool_run and self.tool_calls and tool_calls:
                raise RuntimeError("Only one explicitly routed tool call is allowed for this request.")
            if force_final_answer and tool_calls:
                # A model can ignore the tool-free final-answer turn and emit
                # another call anyway. Do not execute it or fail the chat; use
                # completed search output when available, then return a clear
                # budget notice instead of another max-tools error.
                fallback = self._tool_result_fallback() or self._forced_final_fallback()
                response = {
                    **response,
                    "content": fallback,
                    "tool_calls": [],
                    "tool_result_fallback": True,
                }
                tool_calls = []
            if not tool_calls:
                raw_answer = str(response.get("content") or "")
                answer = self.answer_sanitizer(raw_answer)
                search_status_only = bool(self.tool_calls and _search_status_only(answer))

                if (
                    not answer
                    and bool(response.get("reasoning_seen"))
                    and thinking_prefill_retries < MAX_THINKING_PREFILL_RETRIES
                    and not force_final_answer
                ):
                    thinking_prefill_retries += 1
                    self.messages.append(self._assistant_continuation_message(response))
                    self.on_event({
                        "type": "tool_status",
                        "tool": "assistant",
                        "status": "running",
                        "message": self._thinking_prefill_status(thinking_prefill_retries),
                    })
                    continue

                if (
                    not answer
                    and not self.tool_calls
                    and empty_response_retries < MAX_EMPTY_RESPONSE_RETRIES
                ):
                    empty_response_retries += 1
                    self.on_event({
                        "type": "tool_status",
                        "tool": "assistant",
                        "status": "running",
                        "message": self._empty_retry_status(empty_response_retries),
                    })
                    continue

                can_recover_after_tool = (
                    bool(self.tool_calls)
                    and not post_tool_recovery_attempted
                    and MAX_POST_TOOL_EMPTY_RECOVERIES > 0
                )
                if (not answer or search_status_only) and can_recover_after_tool:
                    post_tool_recovery_attempted = True
                    force_final_answer = True
                    self.on_event({
                        "type": "tool_status",
                        "tool": "assistant",
                        "status": "running",
                        "message": self._empty_recovery_status(search_status_only),
                    })
                    self.messages.append(self._assistant_continuation_message(response))
                    self.messages.append({
                        "role": "user",
                        "content": self._empty_recovery_nudge(search_status_only),
                    })
                    continue

                if search_status_only:
                    # A generic status is non-empty to the model API, but it
                    # still does not answer what the search found.
                    fallback = self._tool_result_fallback()
                    if fallback:
                        answer = fallback
                if not answer:
                    fallback = self._tool_result_fallback()
                    answer = fallback or (
                        "Screen capture completed. The screenshot is attached to this chat."
                        if self._capture_only_run
                        else "SGLang returned no visible answer."
                    )
                streamed_answer = "".join(self.streamed_answer)
                if self.stream_visible_answer and streamed_answer:
                    if streamed_answer != answer:
                        self.on_event({"type": "answer_replace", "text": answer})
                else:
                    self.on_event({"type": "answer_delta", "text": answer})
                elapsed = max(0.001, time.perf_counter() - self.started_at)
                usage = response.get("usage") or {}
                completion_tokens = usage.get("completion_tokens")
                metrics = {
                    "latency_ms": round(elapsed * 1000, 2),
                    "first_token_ms": self.first_token_ms,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": completion_tokens,
                    "total_tokens": usage.get("total_tokens"),
                    "effective_max_tokens": self.effective_max_tokens,
                    "tokens_per_second": (
                        round(float(completion_tokens) / elapsed, 2)
                        if isinstance(completion_tokens, (int, float)) and elapsed > 0
                        else None
                    ),
                    "agent_steps": self.steps,
                    "agent_tool_calls": self.tool_calls,
                    "agent_tool_time_ms": round(self.tool_time_ms, 1),
                }
                return {"reply": answer, "model": self.model, "metrics": metrics}

            if not self.tools_available and not self._single_tool_run:
                raise RuntimeError("SGLang returned tool calls although this model has no configured tool-call parser.")
            if tool_rounds >= MAX_AGENT_STEPS:
                force_final_answer = True
                self.messages.append(self._assistant_continuation_message(response))
                self.messages.append({"role": "user", "content": self._tool_budget_nudge()})
                self.on_event({
                    "type": "tool_status",
                    "tool": "assistant",
                    "status": "running",
                    "message": self._tool_budget_status(),
                })
                continue
            tool_rounds += 1
            force_final_answer = False
            if (
                len(tool_calls) > MAX_TOOL_CALLS_PER_STEP
                or self.tool_calls + len(tool_calls) > MAX_TOOL_CALLS_PER_RUN
            ):
                self.messages.append(self._assistant_continuation_message(response))
                self.messages.append({
                    "role": "user",
                    "content": self._tool_budget_nudge(
                        per_step=len(tool_calls) > MAX_TOOL_CALLS_PER_STEP
                    ),
                })
                force_final_answer = True
                self.on_event({
                    "type": "tool_status",
                    "tool": "assistant",
                    "status": "running",
                    "message": self._tool_budget_status(),
                })
                continue

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "tool_calls": tool_calls,
            }
            if response.get("content"):
                assistant_message["content"] = response["content"]
            if response.get("reasoning_content"):
                assistant_message["reasoning_content"] = response["reasoning_content"]
            self.messages.append(assistant_message)
            for call in tool_calls:
                self._check_cancelled()
                tool_id = str(call.get("id") or uuid.uuid4().hex)
                function = call.get("function") or {}
                name = str(function.get("name") or "")
                arguments = _parse_tool_arguments(
                    function.get("arguments"), tool_name=name
                )
                label = _TOOL_LABELS.get(name, "Running an Agent tool")
                self.on_event({"type": "tool_status", "tool": name, "status": "running", "message": f"{label}…"})
                tool_started = time.perf_counter()
                outcome = "ok"
                tool_error = ""
                try:
                    result = self._execute_tool_cancellable(name, arguments)
                    tool_error = _tool_result_error(name, result)
                    if tool_error:
                        outcome = "error"
                except (InterruptedError, AgentCancelledError) as exc:
                    outcome = "cancelled"
                    self._audit(name, arguments, outcome, tool_started)
                    raise AgentCancelledError("Stopped by user.") from exc
                except Exception as exc:
                    outcome = "error"
                    tool_error = str(exc).strip()[:1000]
                    result = json.dumps({"error": tool_error}, ensure_ascii=False)
                result, attachment = _extract_tool_attachment(name, arguments, str(result))
                if attachment is None and name == "computer_use":
                    try:
                        attachment_result = json.loads(result)
                    except (TypeError, ValueError):
                        attachment_result = {}
                    attachment_error = (
                        str(attachment_result.get("screenshot_display_error") or "").strip()
                        if isinstance(attachment_result, dict)
                        else ""
                    )
                    if attachment_error:
                        outcome = "error"
                        tool_error = attachment_error[:1000]
                self._audit(name, arguments, outcome, tool_started)
                if attachment is not None:
                    self.on_event({"type": "tool_attachment", "attachment": attachment})
                result = str(result)[:MAX_TOOL_RESULT_CHARS]
                self.tool_calls += 1
                self.on_event({
                    "type": "tool_status",
                    "tool": name,
                    "status": "failed" if outcome == "error" else "completed",
                    "message": f"{label}: {'failed' if outcome == 'error' else 'done'}",
                })
                if name == "web_search" and outcome != "error":
                    self._emit_search_results(result)
                self.messages.append({"role": "tool", "tool_call_id": tool_id, "name": name, "content": result})
                self.tool_time_ms += (time.perf_counter() - tool_started) * 1000.0
                if self._capture_only_run and outcome == "error":
                    detail = tool_error or "The read-only screen capture failed."
                    raise RuntimeError(f"Screen capture failed: {detail}")
                if self._single_tool_run and outcome == "error":
                    detail = tool_error or f"{label} failed."
                    raise RuntimeError(f"{label} failed: {detail}")
        fallback = self._tool_result_fallback() or self._processing_limit_fallback()
        if self.stream_visible_answer and self.streamed_answer:
            self.on_event({"type": "answer_replace", "text": fallback})
        else:
            self.on_event({"type": "answer_delta", "text": fallback})
        return {
            "reply": fallback,
            "model": self.model,
            "metrics": {
                "latency_ms": round(max(0.001, time.perf_counter() - self.started_at) * 1000.0, 2),
                "first_token_ms": self.first_token_ms,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "effective_max_tokens": self.effective_max_tokens,
                "tokens_per_second": None,
                "agent_steps": self.steps,
                "agent_tool_calls": self.tool_calls,
                "agent_tool_time_ms": round(self.tool_time_ms, 1),
            },
        }

    def _tool_budget_nudge(self, *, per_step: bool = False) -> str:
        if self._request_is_chinese():
            if per_step:
                return (
                    "你一次请求了过多工具。请停止调用工具，根据已有对话和已经返回的结果，"
                    "直接给出当前能完成的最终答复，并说明尚未完成的部分。"
                )
            return (
                "本次 Agent 的安全工具预算已用完。请停止调用工具，根据完整对话和已经返回的结果，"
                "直接给出当前能完成的最终答复，并说明尚未完成的部分。"
            )
        if per_step:
            return (
                "You requested too many tools in one step. Stop calling tools and answer using the full "
                "conversation and results already returned. State clearly what remains incomplete."
            )
        return (
            "The safe tool budget for this Agent run has been reached. Stop calling tools and provide the "
            "best final answer from the full conversation and results already returned. State what remains incomplete."
        )

    def _tool_budget_status(self) -> str:
        if self._request_is_chinese():
            return "工具预算已到上限，正在根据已完成的结果整理答复…"
        return "Tool budget reached. Preparing an answer from completed results…"

    def _forced_final_fallback(self) -> str:
        if self._request_is_chinese():
            return (
                "模型在整理最终答复时仍请求了新工具；NextCore 没有执行这次额外请求。"
                "请查看上方已完成的工具结果，或缩小任务范围后继续。"
            )
        return (
            "The model requested another tool while preparing its final answer. NextCore did not run it. "
            "Review the completed tool results above, or continue with a narrower request."
        )

    def _processing_limit_fallback(self) -> str:
        if self._request_is_chinese():
            return (
                "Agent 已停止继续处理这次请求。请查看上方已完成的工具结果，"
                "或缩小任务范围后继续。"
            )
        return (
            "The Agent stopped processing this request. Review the completed tool results above, "
            "or continue with a narrower request."
        )

    def _empty_recovery_nudge(self, status_only: bool) -> str:
        if self._request_is_chinese():
            if status_only:
                return (
                    "你刚才只报告了搜索状态，没有回答搜索发现。请依据完整对话和上方搜索结果，"
                    "直接总结实际发现并列出来源；不要再次调用工具，也不要只说搜索完成。"
                )
            return (
                "上一次工具结果已经返回，但没有生成可见答复。请依据完整对话和上方工具结果继续完成用户请求；"
                "不要再次调用工具，并直接给出面向用户的答复。"
            )
        if status_only:
            return (
                "Your last message only reported search status and did not answer what the search found. "
                "Use the full conversation and search results above to summarize the findings and cite sources. "
                "Do not call tools again or merely say the search is complete."
            )
        return (
            "The tool results are available above, but your last turn had no visible answer. "
            "Use the full conversation and those results to answer the user's request now. "
            "Do not call tools again; provide a user-facing response."
        )

    @staticmethod
    def _assistant_continuation_message(response: dict[str, Any]) -> dict[str, Any]:
        """Preserve an assistant turn and hidden SGLang reasoning for replay."""
        message: dict[str, Any] = {
            "role": "assistant",
            "content": str(response.get("content") or ""),
        }
        reasoning = str(response.get("reasoning_content") or "")
        if reasoning:
            message["reasoning_content"] = reasoning
        if not message["content"] and not reasoning:
            message["content"] = "(empty)"
        return message

    def _thinking_prefill_status(self, attempt: int) -> str:
        if self._request_is_chinese():
            return f"模型正在继续生成最终答复…（{attempt}/{MAX_THINKING_PREFILL_RETRIES}）"
        return f"Continuing after a reasoning-only turn… ({attempt}/{MAX_THINKING_PREFILL_RETRIES})"

    def _empty_retry_status(self, attempt: int) -> str:
        if self._request_is_chinese():
            return f"模型没有返回内容，正在重试…（{attempt}/{MAX_EMPTY_RESPONSE_RETRIES}）"
        return f"The model returned an empty response; retrying… ({attempt}/{MAX_EMPTY_RESPONSE_RETRIES})"

    def _empty_recovery_status(self, status_only: bool) -> str:
        if self._request_is_chinese():
            if status_only:
                return "模型只返回了搜索状态，正在要求它说明实际发现…"
            return "模型没有生成可见答复，正在根据工具结果重试…"
        if status_only:
            return "The model only reported search status; asking it to explain the findings…"
        return "The model returned no visible answer; retrying with the tool results…"

    def _context_retry_status(self, previous_budget: int) -> str:
        if self._request_is_chinese():
            return (
                "完整对话已保留；SGLang 提示回复预算超出当前上下文余量，"
                f"正在把最大回复长度从 {previous_budget} 调低到 {self.effective_max_tokens} token 后重试…"
            )
        return (
            "Keeping the full conversation; SGLang reported insufficient context headroom, "
            f"so the reply budget is being reduced from {previous_budget} to {self.effective_max_tokens} tokens…"
        )

    def _emit_search_results(self, result: str) -> None:
        """Put concrete web results in chat before asking the model to summarize."""
        try:
            payload = json.loads(result)
        except (TypeError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        source_results = payload.get("results")
        if not isinstance(source_results, list):
            return
        results = []
        for item in source_results[:5]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()[:240]
            url = str(item.get("url") or "").strip()[:1000]
            snippet = str(item.get("snippet") or "").strip()[:500]
            if title or url:
                results.append({"title": title or url, "url": url, "snippet": snippet})
        count = len(source_results)
        if self._request_is_chinese():
            message = f"搜索到 {count} 条结果" if count else "没有找到搜索结果"
            empty_message = "没有可显示的结果链接。"
        else:
            message = f"Found {count} search result{'s' if count != 1 else ''}" if count else "No search results found"
            empty_message = "No result links were returned."
        self.on_event({
            "type": "tool_results",
            "tool": "web_search",
            "message": message,
            "empty_message": empty_message,
            "results": results,
        })

    def _tool_summary_status(self) -> str:
        last_tool = next(
            (message.get("name") for message in reversed(self.messages) if message.get("role") == "tool"),
            "",
        )
        if self._request_is_chinese():
            if last_tool == "web_search":
                return "搜索已完成，正在整理结果并生成回复…"
            if last_tool == "web_fetch":
                return "网页已读取，正在整理内容并生成回复…"
            return "工具已完成，正在整理结果并生成回复…"
        if last_tool == "web_search":
            return "Search finished. The model is preparing a reply…"
        if last_tool == "web_fetch":
            return "Page loaded. The model is preparing a reply…"
        return "Tool finished. The model is preparing a reply…"

    def _request_is_chinese(self) -> bool:
        user_text = next(
            (
                str(message.get("content") or "")
                for message in reversed(self.messages)
                if message.get("role") == "user"
            ),
            "",
        )
        return any("\u4e00" <= char <= "\u9fff" for char in user_text)

    def _tool_result_fallback(self) -> str:
        """Show useful search/page output if SGLang cannot summarize it."""
        tool_message = next(
            (
                message
                for message in reversed(self.messages)
                if message.get("role") == "tool"
                and message.get("name") in {"web_search", "web_fetch"}
            ),
            None,
        )
        if tool_message is None:
            return ""
        try:
            payload = json.loads(str(tool_message.get("content") or ""))
        except (TypeError, ValueError):
            return ""
        if not isinstance(payload, dict):
            return ""

        is_chinese = self._request_is_chinese()
        prefix = (
            "搜索已完成；模型暂时无法整理完整摘要，以下是检索到的相关摘录："
            if is_chinese
            else "The model could not synthesize a full summary. Relevant search excerpts:"
        )
        name = str(tool_message.get("name") or "")
        if name == "web_search":
            results = payload.get("results")
            if not isinstance(results, list):
                return ""
            rows = []
            for item in results[:2]:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or "来源").strip()[:160]
                title = title.replace("[", "").replace("]", "")
                url = str(item.get("url") or "").strip()[:1000]
                snippet = " ".join(str(item.get("snippet") or "").split())[:500]
                if snippet and _is_safe_citation_url(url):
                    row = f"{snippet} [{title}]({url})"
                elif _is_safe_citation_url(url):
                    row = f"[{title}]({url})"
                elif snippet:
                    row = snippet
                else:
                    continue
                rows.append(row)
            if not rows:
                return "搜索已完成，但没有找到结果。" if is_chinese else "The search completed without results."
            return prefix + " " + " ".join(rows)

        text = str(payload.get("text") or "").strip()
        if not text:
            return ""
        source_url = str(payload.get("url") or "").strip()[:1000]
        citation = ""
        if _is_safe_citation_url(source_url):
            source_host = urlsplit(source_url).hostname or "网页来源"
            source_title = source_host.replace("[", "").replace("]", "")
            citation = f" [{source_title}]({source_url})"
        page_prefix = (
            "网页已读取，但模型暂时无法总结。页面内容摘录："
            if is_chinese
            else "The page was read, but the model could not summarize it. Excerpt:"
        )
        return f"{page_prefix} {text[:1600]}{citation}"

    def _execute_state_tool(self, name: str, arguments: dict[str, Any]) -> str:
        action = str(arguments.get("action") or "list").strip().lower()
        if name == "session_search":
            rows = self.state_store.search_conversations(str(arguments.get("query") or ""), int(arguments.get("limit", 8)))
            return json.dumps(rows, ensure_ascii=False, default=str) if rows else "No matching indexed conversation messages were found."
        if name == "memory":
            item_id = str(arguments.get("id") or "").strip()
            if action in {"search", "list"}:
                rows = self.state_store.list_memories(str(arguments.get("query") or ""))
                return json.dumps(rows, ensure_ascii=False)
            if action == "save":
                item = self.state_store.save_memory(
                    str(arguments.get("title") or ""), str(arguments.get("body") or ""),
                    [str(tag) for tag in arguments.get("tags", []) if isinstance(tag, str)],
                    memory_id=item_id or None,
                )
                return json.dumps({"saved": True, "id": item["id"], "revision": item["revision"]}, ensure_ascii=False)
            if action == "revisions":
                return json.dumps(self.state_store.memory_revisions(item_id), ensure_ascii=False)
            if action == "rollback":
                item = self.state_store.rollback_memory(item_id, int(arguments.get("revision", 0)))
                return json.dumps({"rolled_back": True, "id": item["id"], "revision": item["revision"]}, ensure_ascii=False)
            if action == "delete":
                self.state_store.delete_memory(item_id)
                return json.dumps({"deleted": True, "id": item_id}, ensure_ascii=False)
            raise ValueError("Unsupported memory action.")
        if name == "skills":
            skill_id = str(arguments.get("id") or "").strip()
            if action in {"search", "list"}:
                rows = self.state_store.list_skills(str(arguments.get("query") or ""), include_disabled=True)
                return json.dumps(rows, ensure_ascii=False)
            if action == "save":
                item = self.state_store.save_skill(
                    str(arguments.get("name") or ""), str(arguments.get("description") or ""),
                    str(arguments.get("body") or ""), skill_id=skill_id or None,
                    enabled=bool(arguments.get("enabled", True)),
                )
                return json.dumps({"saved": True, "id": item["id"], "revision": item["revision"]}, ensure_ascii=False)
            if action == "revisions":
                return json.dumps(self.state_store.skill_revisions(skill_id), ensure_ascii=False)
            if action == "rollback":
                item = self.state_store.rollback_skill(skill_id, int(arguments.get("revision", 0)))
                return json.dumps({"rolled_back": True, "id": item["id"], "revision": item["revision"]}, ensure_ascii=False)
            if action == "enable":
                item = self.state_store.set_skill_enabled(skill_id, bool(arguments.get("enabled", True)))
                return json.dumps({"updated": True, "id": item["id"], "enabled": bool(item["enabled"])}, ensure_ascii=False)
            if action == "delete":
                self.state_store.delete_skill(skill_id)
                return json.dumps({"deleted": True, "id": skill_id}, ensure_ascii=False)
            raise ValueError("Unsupported Skill action.")
        raise ValueError("Unsupported state tool.")

    def _delegate(self, arguments: dict[str, Any]) -> str:
        if self.delegate_depth >= 1:
            raise ValueError("Nested delegation is disabled.")
        tasks = arguments.get("tasks")
        if not isinstance(tasks, list) or not tasks or len(tasks) > 2:
            raise ValueError("Delegate accepts one or two parallel tasks.")
        parent_names = {str(item.get("function", {}).get("name") or "") for item in self._tool_definitions}
        parent_names.discard("delegate")

        def execute_one(item: Any, index: int) -> dict[str, Any]:
            if self.cancel_event.is_set():
                raise AgentCancelledError("Stopped by user.")
            if not isinstance(item, dict) or not str(item.get("prompt") or "").strip():
                return {"task": index + 1, "error": "A non-empty task prompt is required."}
            requested = item.get("tools")
            scope = parent_names if not isinstance(requested, list) or not requested else parent_names.intersection(map(str, requested))
            scope.discard("delegate")
            if not scope:
                return {"task": index + 1, "error": "The child has no tools in its inherited scope."}
            child = SGLangAgent(
                server_url=self.server_url, model=self.model,
                messages=[{"role": "user", "content": str(item["prompt"])[:8000]}],
                max_tokens=min(self.effective_max_tokens, 4096), temperature=self.temperature,
                app_data_dir=self.app_data_dir, run_id=f"{self.run_id}-child-{index}",
                tool_parser=self.tool_parser, cancel_event=self.cancel_event,
                reasoning_parser=self.reasoning_parser, stream_visible_answer=False,
                answer_sanitizer=self.answer_sanitizer, allowed_tools=scope,
                mcp_runtime=self.mcp_runtime, delegate_depth=1,
                authorization_scope=self.authorization_scope,
            )
            result = child.run()
            return {"task": index + 1, "reply": result["reply"], "metrics": result["metrics"]}

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(2, len(tasks)), thread_name_prefix="nextcore-agent-child") as pool:
            futures = [pool.submit(execute_one, item, index) for index, item in enumerate(tasks[:2])]
            results = []
            for future in futures:
                if self.cancel_event.is_set():
                    raise AgentCancelledError("Stopped by user.")
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append({"error": str(exc)[:500]})
        return json.dumps({"delegated_tasks": results}, ensure_ascii=False)

    def _enforce_task_scope(self, name: str, arguments: dict[str, Any]) -> None:
        if self.allowed_tools is None:
            return
        if name not in self.allowed_tools:
            raise PermissionError(f"The task did not authorize the {name} tool.")
        if name in {"computer_use", "browser"} and not bool(self.authorization_scope.get("computer_control")):
            raise PermissionError("This scheduled task does not include computer-control authorization.")
        if name == "file":
            roots = self.authorization_scope.get("allowed_paths")
            if not isinstance(roots, list) or not roots:
                raise PermissionError("This scheduled task has no authorized file paths.")
            candidate = Path(str(arguments.get("path") or "")).expanduser().resolve()
            permitted = False
            for raw_root in roots:
                try:
                    root = Path(str(raw_root)).expanduser().resolve()
                    if candidate == root or root in candidate.parents:
                        permitted = True
                        break
                except (OSError, RuntimeError):
                    continue
            if not permitted:
                raise PermissionError("The file path is outside this scheduled task's authorized directories.")
        if name == "browser":
            if str(arguments.get("action") or "") == "prepare_existing":
                allowed_profiles = self.authorization_scope.get("browser_profile_ids")
                profile_id = str(arguments.get("profile_id") or "")
                if not isinstance(allowed_profiles, list) or profile_id not in allowed_profiles:
                    raise PermissionError("This scheduled task does not authorize an existing signed-in browser profile.")
            allowed_domains = self.authorization_scope.get("allowed_domains")
            url = str(arguments.get("url") or "")
            if url and isinstance(allowed_domains, list) and allowed_domains:
                host = (urlsplit(url).hostname or "").lower()
                if not any(host == domain or host.endswith("." + domain) for domain in allowed_domains):
                    raise PermissionError("The browser URL is outside this scheduled task's authorized domains.")

    def _execute_tool_cancellable(self, name: str, arguments: dict[str, Any]) -> str:
        """Run blocking tools off-thread so cancellation reaches the UI promptly."""
        result_queue: queue.Queue[tuple[str | None, BaseException | None]] = queue.Queue(maxsize=1)

        def run_tool() -> None:
            try:
                self._enforce_task_scope(name, arguments)
                if name in {"memory", "skills", "session_search"}:
                    result = self._execute_state_tool(name, arguments)
                elif name == "code_execution":
                    result = execute_python(str(arguments.get("code") or ""), timeout_seconds=int(arguments.get("timeout_seconds", 10)), cancel_event=self.cancel_event)
                elif name == "delegate":
                    result = self._delegate(arguments)
                elif "__" in name and self.mcp_runtime is not None:
                    result = self.mcp_runtime.execute(name, arguments)
                else:
                    result = execute_tool(
                        name,
                        arguments,
                        app_data_dir=self.app_data_dir,
                        cancel_event=self.cancel_event,
                        authorization_scope=self.authorization_scope if self.allowed_tools is not None else None,
                    )
                result_queue.put((str(result), None))
            except BaseException as exc:
                with suppress(queue.Full):
                    result_queue.put((None, exc))

        worker = threading.Thread(
            target=run_tool,
            name=f"nextcore-agent-tool-{name[:24]}",
            daemon=True,
        )
        worker.start()
        while True:
            if self.cancel_event.is_set():
                if name in {"computer_use", "browser"}:
                    threading.Thread(
                        target=interrupt_computer_use,
                        name="nextcore-agent-stop-computer-use",
                        daemon=True,
                    ).start()
                elif name in {"web_search", "web_fetch"}:
                    threading.Thread(
                        target=interrupt_web_requests,
                        args=(self.cancel_event,),
                        name="nextcore-agent-stop-web-request",
                        daemon=True,
                    ).start()
                raise AgentCancelledError("Stopped by user.")
            try:
                result, error = result_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if self.cancel_event.is_set():
                if name in {"computer_use", "browser"}:
                    threading.Thread(
                        target=interrupt_computer_use,
                        name="nextcore-agent-stop-computer-use",
                        daemon=True,
                    ).start()
                elif name in {"web_search", "web_fetch"}:
                    threading.Thread(
                        target=interrupt_web_requests,
                        args=(self.cancel_event,),
                        name="nextcore-agent-stop-web-request",
                        daemon=True,
                    ).start()
                raise AgentCancelledError("Stopped by user.")
            if error is not None:
                raise error
            return str(result or "")

    def _complete(self, *, force_final_answer: bool = False) -> dict[str, Any]:
        self._check_cancelled()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self.messages,
            "max_tokens": (
                min(self.effective_max_tokens, MAX_POST_TOOL_SUMMARY_TOKENS)
                if self.tool_calls or force_final_answer
                else self.effective_max_tokens
            ),
            "temperature": self.temperature,
            "stream": True,
            "separate_reasoning": True,
        }
        if self.tools_available and not self._single_tool_run and not force_final_answer:
            payload["tools"] = self._tool_definitions
            payload["tool_choice"] = "auto"
        if (self.tool_calls or force_final_answer) and self.reasoning_parser == "qwen3":
            # Qwen3 can spend the full generation budget reasoning after a
            # tool result. During the recovery turn, prioritize visible text.
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        started = time.perf_counter()
        try:
            response = self.session.post(
                f"{self.server_url}/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=(
                    5,
                    POST_TOOL_SUMMARY_TIMEOUT_SECONDS if self.tool_calls else 600,
                ),
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Could not connect to SGLang ({type(exc).__name__}).") from exc
        try:
            response.raise_for_status()
            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            visible_filter = _ThinkTagFilter() if self.stream_visible_answer else None
            tool_calls: dict[int, dict[str, Any]] = {}
            usage: dict[str, Any] = {}
            reasoning_seen = False
            finish_reason: Any = None
            lines: queue.Queue[tuple[str | None, BaseException | None]] = queue.Queue()

            def read_stream() -> None:
                try:
                    for raw_line in response.iter_lines(decode_unicode=True, chunk_size=1):
                        line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
                        lines.put((line, None))
                except BaseException as exc:
                    lines.put((None, exc))
                finally:
                    lines.put((None, None))

            reader = threading.Thread(
                target=read_stream,
                name=f"nextcore-agent-sglang-{self.run_id[:24]}",
                daemon=True,
            )
            reader.start()
            stream_finished = False
            try:
                while not stream_finished:
                    self._check_cancelled()
                    try:
                        raw_line, stream_error = lines.get(timeout=0.15)
                    except queue.Empty:
                        continue
                    if stream_error is not None:
                        if isinstance(stream_error, requests.RequestException):
                            raise RuntimeError(
                                f"SGLang response failed ({type(stream_error).__name__})."
                            ) from stream_error
                        raise RuntimeError(
                            f"SGLang stream reader failed ({type(stream_error).__name__})."
                        ) from stream_error
                    if raw_line is None:
                        stream_finished = True
                        continue
                    if not raw_line:
                        continue
                    line = raw_line
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        stream_finished = True
                        continue
                    try:
                        chunk = json.loads(data)
                    except ValueError:
                        continue
                    if isinstance(chunk.get("usage"), dict):
                        usage.update(chunk["usage"])
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice.get("finish_reason")
                    # Keep reasoning in memory only so Hermes-style recovery
                    # and tool continuations can replay it. Never display it
                    # or write it to the audit record.
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    if isinstance(reasoning, str) and reasoning.strip():
                        reasoning_seen = True
                        reasoning_parts.append(reasoning)
                    elif isinstance(reasoning, list) and reasoning:
                        reasoning_seen = True
                        reasoning_parts.append("".join(str(part) for part in reasoning))
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        if self.first_token_ms is None:
                            self.first_token_ms = (time.perf_counter() - started) * 1000.0
                        content_parts.append(content)
                        if visible_filter is not None:
                            visible = visible_filter.feed(content)
                            if visible:
                                self.streamed_answer.append(visible)
                                self.on_event({"type": "answer_delta", "text": visible})
                    for item in delta.get("tool_calls") or []:
                        if not isinstance(item, dict):
                            continue
                        index = int(item.get("index", 0))
                        acc = tool_calls.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                        if item.get("id"):
                            acc["id"] = str(item["id"])
                        function = item.get("function") or {}
                        if function.get("name"):
                            acc["function"]["name"] += str(function["name"])
                        if function.get("arguments"):
                            acc["function"]["arguments"] += str(function["arguments"])
            finally:
                if self.cancel_event.is_set():
                    response.close()
                reader.join(timeout=0.5)
            if visible_filter is not None:
                visible = visible_filter.feed("", final=True)
                if visible:
                    self.streamed_answer.append(visible)
                    self.on_event({"type": "answer_delta", "text": visible})
            return {
                "content": "".join(content_parts),
                "reasoning_content": "".join(reasoning_parts),
                "tool_calls": [tool_calls[index] for index in sorted(tool_calls)],
                "usage": usage,
                "reasoning_seen": reasoning_seen,
                "finish_reason": finish_reason,
            }
        except requests.HTTPError as exc:
            detail = _response_error_text(response)
            if _looks_like_context_window_error(detail):
                raise SGLangContextWindowError(detail or "SGLang context limit exceeded.") from exc
            raise RuntimeError(f"SGLang response failed ({type(exc).__name__}).") from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"SGLang response failed ({type(exc).__name__}).") from exc
        finally:
            response.close()

    def _audit(self, tool: str, arguments: dict[str, Any], outcome: str, started_at: float) -> None:
        audit_tool_call(
            app_data_dir=self.app_data_dir,
            run_id=self.run_id,
            tool=tool,
            arguments=arguments,
            outcome=outcome,
            duration_ms=(time.perf_counter() - started_at) * 1000.0,
        )

    def _check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise AgentCancelledError("Stopped by user.")


def tool_intent(message: str) -> bool:
    """Small fallback disclosure heuristic; it never authorizes a tool call."""
    text = str(message or "").lower()
    markers = (
        "search the web", "search online", "look up online", "browse the web", "open this website", "visit this website",
        "use the internet", "on the internet", "open safari", "control my mac", "click on", "run this command",
        "terminal", "file on my", "find the file", "read the file", "write a file", "delete the file",
        "computer_use", "computer use", "screenshot", "screen capture", "capture the screen",
        "capture current screen", "look at my screen", "what is on my screen", "what's on my screen",
        "window visible", "window is visible", "is the window visible",
        "上网搜索", "联网搜索", "网上查", "搜索网页", "打开网页", "访问网站", "浏览网站", "打开浏览器", "操作电脑",
        "控制电脑", "运行命令", "执行命令", "终端", "读一下文件", "读取文件", "写入文件", "查找文件", "删除文件",
        "截屏", "截图", "屏幕截图", "截取屏幕", "截取当前屏幕", "捕获屏幕", "查看屏幕", "查看当前屏幕",
        "看看屏幕", "屏幕上有什么", "窗口是否可见", "是否可见",
    )
    return any(marker in text for marker in markers)


def _explicit_tool_call(message: str) -> tuple[str, dict[str, Any]] | None:
    """Route only clearly named, parameterized read or execution requests."""
    original = str(message or "").strip()
    text = original.casefold()
    capture_markers = (
        "screenshot", "screen capture", "capture the screen", "capture current screen",
        "take a screenshot", "inspect the screen", "look at my screen", "what is on my screen",
        "what's on my screen", "截屏", "截图", "屏幕截图", "截取屏幕", "截取当前屏幕",
        "捕获屏幕", "查看屏幕", "查看当前屏幕", "看看屏幕", "屏幕上有什么",
    )
    if any(marker in text for marker in capture_markers):
        arguments: dict[str, Any] = {"action": "capture", "mode": "vision"}
        if "nextcore" in text:
            arguments["app"] = "token-workshed NextCore"
        return "computer_use", arguments
    if ("window" in text and "visible" in text) or ("窗口" in text and "可见" in text):
        arguments = {"action": "capture", "mode": "vision"}
        if "nextcore" in text:
            arguments["app"] = "token-workshed NextCore"
        return "computer_use", arguments

    # A command is routed directly only when the user explicitly names the
    # terminal/command action and marks the exact command in backticks.
    command_match = re.search(r"`([^`\n]+)`", original)
    terminal_intent = any(
        marker in text
        for marker in (
            "terminal", "run this command", "run the command", "execute this command",
            "execute the command", "运行命令", "执行命令", "用终端", "通过终端",
        )
    )
    if command_match and terminal_intent:
        command = command_match.group(1).strip()
        if command:
            return "terminal", {"command": command}

    # Read only an explicitly identified path. Write/delete requests remain
    # model-mediated and continue through the normal Agent policy.
    destructive_file_intent = any(
        marker in text
        for marker in ("write", "overwrite", "delete", "remove", "rename", "写入", "覆盖", "删除", "移除", "重命名")
    )
    read_file_intent = any(
        marker in text
        for marker in ("read", "show me the contents", "open the file", "cat the file", "读取", "读一下", "查看文件内容")
    )
    names_file = "file" in text or "文件" in text
    if command_match and read_file_intent and names_file and not destructive_file_intent:
        path = command_match.group(1).strip()
        if path:
            return "file", {"action": "read", "path": path}

    # Explicit web searches and URL reads do not depend on the model deciding
    # whether to call a tool. Result text remains untrusted Agent input.
    search_patterns = (
        r"\b(?:search(?: the)? web|search online|look up online)\s+for\s+(.+)",
        r"(?:上网搜索|联网搜索|搜索网页|网上搜索|网上查找|搜索一下)\s*(.+)",
    )
    for pattern in search_patterns:
        match = re.search(pattern, original, flags=re.IGNORECASE)
        if match:
            query = match.group(1).strip().strip(" \t\r\n.,!?;，。；！？")
            if query and len(query) <= 500:
                return "web_search", {"query": query, "max_results": 5}

    url_match = re.search(r"https?://[^\s<>\"'`]+", original, flags=re.IGNORECASE)
    web_intent = any(
        marker in text
        for marker in (
            "open ", "visit ", "fetch ", "read this website", "browse to ",
            "打开网页", "打开网站", "访问网站", "读取网页", "浏览这个网址",
        )
    )
    if url_match and web_intent:
        url = url_match.group(0).rstrip(".,!?;:，。；！？")
        if url:
            return "web_fetch", {"url": url}
    return None


def required_tool_for_message(message: str) -> str | None:
    """Return a tool for an explicit deterministic in-app request."""
    call = _explicit_tool_call(message)
    return call[0] if call else None


def required_tool_arguments_for_message(message: str) -> dict[str, Any] | None:
    """Return pre-bound arguments only for clear, exact user intent."""
    call = _explicit_tool_call(message)
    return dict(call[1]) if call else None


def _tool_result_error(name: str, result: str) -> str:
    """Return a concise failure explanation from a structured tool result."""
    if name not in {"terminal", "computer_use", "browser"}:
        return ""
    try:
        payload = json.loads(result)
    except (TypeError, ValueError):
        return ""
    if not isinstance(payload, dict):
        return ""
    if name == "terminal":
        code = payload.get("exit_code")
        return f"Command exited with code {code}." if isinstance(code, int) and code != 0 else ""
    if name == "computer_use":
        if payload.get("ok") is False or payload.get("error"):
            return str(payload.get("message") or payload.get("error") or "Computer use failed.")[:1000]
        return ""
    if str(payload.get("status") or "").casefold() in {"failed", "error", "refused"}:
        return str(payload.get("message") or payload.get("error") or "Browser action failed.")[:1000]
    return ""


def _extract_tool_attachment(
    name: str, arguments: dict[str, Any], result: str
) -> tuple[str, dict[str, str] | None]:
    """Keep local screen pixels out of model context and stream them to NextCore UI."""
    if name != "computer_use" or arguments.get("action") != "capture":
        return result, None
    try:
        payload = json.loads(result)
    except (TypeError, ValueError):
        return result, None
    if not isinstance(payload, dict):
        return result, None
    screenshot = payload.pop("screenshot_b64", None)
    mime_type = str(payload.pop("screenshot_mime_type", "") or "").strip().lower()
    is_vision_capture = str(arguments.get("mode") or "ax").strip().lower() == "vision"
    if not is_vision_capture or not isinstance(screenshot, str) or not screenshot:
        return json.dumps(payload, ensure_ascii=False), None
    if mime_type not in {"image/jpeg", "image/png", "image/webp"}:
        mime_type = "image/png"
    try:
        screenshot, mime_type = _prepare_chat_capture(screenshot, mime_type)
    except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError):
        payload["screenshot_display_error"] = (
            "The screen was captured, but NextCore could not prepare the image for chat display."
        )
        return json.dumps(payload, ensure_ascii=False), None
    attachment = {
        "mime_type": mime_type,
        "base64": screenshot,
        "app": str(payload.get("app") or "")[:200],
        "window_title": str(payload.get("window_title") or "")[:300],
    }
    return json.dumps(payload, ensure_ascii=False), attachment


def _prepare_chat_capture(encoded: str, mime_type: str) -> tuple[str, str]:
    """Keep screenshot attachments compact enough for reliable chat display."""
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("The screenshot is not valid base64.") from exc
    if not raw or len(raw) > 64 * 1024 * 1024:
        raise ValueError("The screenshot is empty or exceeds the processing limit.")

    with Image.open(io.BytesIO(raw)) as image:
        width, height = image.size
        if width <= 0 or height <= 0 or width * height > MAX_CHAT_CAPTURE_PIXELS:
            raise ValueError("The screenshot dimensions exceed the display limit.")
        image.load()
        source_format = str(image.format or "").upper()
        normalized_mime = {
            "JPEG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp",
        }.get(source_format)
        if normalized_mime is None:
            raise ValueError("The screenshot format is not supported.")

        if (
            len(raw) <= MAX_CHAT_CAPTURE_BYTES
            and width <= MAX_CHAT_CAPTURE_DIMENSIONS[0]
            and height <= MAX_CHAT_CAPTURE_DIMENSIONS[1]
        ):
            return encoded, normalized_mime

        frame = image.convert("RGB")
        frame.thumbnail(MAX_CHAT_CAPTURE_DIMENSIONS, Image.Resampling.LANCZOS)
        for quality in (84, 76, 68):
            output = io.BytesIO()
            frame.save(output, format="JPEG", quality=quality, optimize=True)
            compressed = output.getvalue()
            if len(compressed) <= MAX_CHAT_CAPTURE_BYTES:
                return base64.b64encode(compressed).decode("ascii"), "image/jpeg"
    raise ValueError("The screenshot could not be reduced to a chat-safe size.")
