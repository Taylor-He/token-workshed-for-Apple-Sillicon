from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import os
import queue
import sys
import tarfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from token_workshed_nextcore import css_svg_ui, desktop_ui, sglang_runtime
from token_workshed_nextcore.agent import (
    AgentCancelledError,
    SGLangAgent,
    driver_manager,
    macos_permissions,
)
from token_workshed_nextcore.agent import (
    runtime as agent_runtime,
)
from token_workshed_nextcore.agent import (
    tools as agent_tools,
)
from token_workshed_nextcore.agent.audit import audit_tool_call
from token_workshed_nextcore.agent.computer_use.backend import (
    ActionResult,
    CaptureResult,
    UIElement,
)
from token_workshed_nextcore.agent.computer_use.cua_backend import (
    CuaDriverBackend,
    _AsyncBridge,
    _resolve_mcp_invocation,
    _safe_subprocess_env,
    cua_driver_child_env,
)
from token_workshed_nextcore.ide_bridge import IDEBridge, register_ide_bridge_routes


def _sse(*chunks: dict[str, Any]) -> list[str]:
    return [f"data: {json.dumps(chunk)}" for chunk in chunks] + ["data: [DONE]"]


_ONE_PIXEL_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg=="


class _Response:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.closed = threading.Event()

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self, **_: Any):
        for line in self.lines:
            if self.closed.is_set():
                return
            yield line

    def close(self) -> None:
        self.closed.set()


class _Session:
    def __init__(self, responses: list[_Response]) -> None:
        self.responses = responses
        self.payloads: list[dict[str, Any]] = []
        self.timeouts: list[Any] = []

    def post(self, _url: str, *, json: dict[str, Any], **kwargs: Any) -> _Response:
        self.payloads.append(json)
        self.timeouts.append(kwargs.get("timeout"))
        return self.responses.pop(0)


def _agent(
    tmp_path: Path,
    *,
    parser: str | None = None,
    reasoning_parser: str | None = None,
    stream_visible_answer: bool = False,
    required_tool: str | None = None,
    required_tool_arguments: dict[str, Any] | None = None,
) -> SGLangAgent:
    return SGLangAgent(
        server_url="http://127.0.0.1:8000",
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=64,
        temperature=0.2,
        app_data_dir=tmp_path,
        run_id="test-run",
        tool_parser=parser,
        cancel_event=threading.Event(),
        reasoning_parser=reasoning_parser,
        required_tool=required_tool,
        required_tool_arguments=required_tool_arguments,
        stream_visible_answer=stream_visible_answer,
    )


@pytest.fixture(autouse=True)
def _isolate_app_agent_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep app-factory tests away from the developer's persistent Agent data."""
    monkeypatch.setenv("TOKEN_WORKSHED_AGENT_DATA_DIR", str(tmp_path / "agent-data"))


@pytest.mark.parametrize(
    "message",
    [
        "只截取当前屏幕一次，并只回答 NextCore 窗口是否可见。",
        "请截屏并告诉我窗口是否可见。",
        "Use computer_use to capture the current screen once.",
        "Check whether the NextCore window is visible.",
    ],
)
def test_tool_intent_recognizes_screen_capture_requests(message: str) -> None:
    assert agent_runtime.tool_intent(message)


@pytest.mark.parametrize(
    "message",
    [
        "只截取当前屏幕一次，并只回答 NextCore 窗口是否可见。",
        "Check whether the NextCore window is visible.",
        "Take a screenshot of the current screen.",
    ],
)
def test_screen_inspection_requests_require_computer_use(message: str) -> None:
    assert agent_runtime.required_tool_for_message(message) == "computer_use"
    assert agent_runtime.required_tool_arguments_for_message(message) is not None


@pytest.mark.parametrize("message", ["hi", "Search the web if needed", "窗口里有哪些按钮？"])
def test_other_requests_do_not_force_a_tool(message: str) -> None:
    assert agent_runtime.required_tool_for_message(message) is None
    assert agent_runtime.required_tool_arguments_for_message(message) is None


@pytest.mark.parametrize(
    ("message", "tool", "arguments"),
    [
        (
            "Please call the NextCore terminal tool with read-only `pwd`.",
            "terminal",
            {"command": "pwd"},
        ),
        (
            "Read the file `/tmp/nextcore-agent-check.txt`.",
            "file",
            {"action": "read", "path": "/tmp/nextcore-agent-check.txt"},
        ),
        (
            "Search the web for SGLang Runtime.",
            "web_search",
            {"query": "SGLang Runtime", "max_results": 5},
        ),
        (
            "Open https://example.com/ and report its title.",
            "web_fetch",
            {"url": "https://example.com/"},
        ),
    ],
)
def test_explicit_read_and_execution_requests_are_routed_deterministically(
    message: str, tool: str, arguments: dict[str, Any]
) -> None:
    assert agent_runtime.required_tool_for_message(message) == tool
    assert agent_runtime.required_tool_arguments_for_message(message) == arguments


def test_explicit_terminal_tool_runs_without_model_tool_call_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = _Response(_sse({"choices": [{"delta": {"content": "The current folder is /tmp."}}]}))
    agent = _agent(
        tmp_path,
        reasoning_parser="qwen3",
        required_tool="terminal",
        required_tool_arguments={"command": "pwd"},
    )
    session = _Session([response])
    agent.session = session  # type: ignore[assignment]
    executed: list[tuple[str, dict[str, Any]]] = []

    def execute(name: str, arguments: dict[str, Any], **_: Any) -> str:
        executed.append((name, arguments))
        return json.dumps({"exit_code": 0, "output": "/tmp"})

    monkeypatch.setattr(agent_runtime, "execute_tool", execute)
    result = agent.run()

    assert executed == [("terminal", {"command": "pwd"})]
    assert "tools" not in session.payloads[0]
    assert session.payloads[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert result["reply"] == "The current folder is /tmp."
    assert agent.tool_calls == 1


def test_agent_does_not_send_tools_without_a_supported_parser(tmp_path: Path) -> None:
    response = _Response(_sse({"choices": [{"delta": {"content": "hello"}}]}))
    agent = _agent(tmp_path)
    session = _Session([response])
    agent.session = session  # type: ignore[assignment]

    result = agent.run()

    assert result["reply"] == "hello"
    assert "tools" not in session.payloads[0]
    assert "tool_choice" not in session.payloads[0]


def test_agent_dispatches_one_read_only_capture_and_prevents_more_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = _Response(_sse({"choices": [{"delta": {"content": "The NextCore window is visible."}}]}))
    agent = _agent(
        tmp_path,
        parser="hermes",
        required_tool="computer_use",
        required_tool_arguments={
            "action": "capture",
            "mode": "vision",
            "app": "token-workshed NextCore",
        },
    )
    session = _Session([response])
    agent.session = session  # type: ignore[assignment]
    executed: list[tuple[str, dict[str, Any]]] = []

    def execute(name: str, arguments: dict[str, Any], **_: Any) -> str:
        executed.append((name, arguments))
        return json.dumps({"app": "token-workshed NextCore", "window_title": "Chat", "elements": []})

    monkeypatch.setattr(agent_runtime, "execute_tool", execute)

    result = agent.run()

    assert executed == [
        (
            "computer_use",
            {"action": "capture", "mode": "vision", "app": "token-workshed NextCore"},
        )
    ]
    assert "tools" not in session.payloads[0]
    assert session.payloads[0]["messages"][-1]["role"] == "tool"
    assert result["reply"] == "The NextCore window is visible."
    assert agent.tool_calls == 1


def test_capture_screenshot_is_streamed_to_chat_and_removed_from_model_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(
        tmp_path,
        parser="hermes",
        required_tool="computer_use",
        required_tool_arguments={"action": "capture", "mode": "vision"},
    )
    session = _Session(
        [_Response(_sse({"choices": [{"delta": {"content": "The window is visible."}}]}))]
    )
    agent.session = session  # type: ignore[assignment]
    events: list[dict[str, Any]] = []
    agent.on_event = events.append

    def capture(*_args: Any, **_kwargs: Any) -> str:
        return json.dumps(
            {
                "app": "token-workshed NextCore",
                "window_title": "Chat",
                "width": 2,
                "height": 2,
                "screenshot_b64": _ONE_PIXEL_PNG,
                "screenshot_mime_type": "image/png",
            }
        )

    monkeypatch.setattr(agent_runtime, "execute_tool", capture)

    result = agent.run()

    attachment_event = next(event for event in events if event["type"] == "tool_attachment")
    assert attachment_event["attachment"] == {
        "mime_type": "image/png",
        "base64": _ONE_PIXEL_PNG,
        "app": "token-workshed NextCore",
        "window_title": "Chat",
    }
    tool_result = session.payloads[0]["messages"][-1]["content"]
    assert "screenshot_b64" not in tool_result
    assert "c2NyZWVu" not in tool_result
    assert result["reply"] == "The window is visible."


def test_large_screen_capture_is_downscaled_before_chat_attachment() -> None:
    from PIL import Image

    source = Image.new("RGB", (3840, 2160), (22, 34, 48))
    raw = io.BytesIO()
    source.save(raw, format="PNG")
    encoded = base64.b64encode(raw.getvalue()).decode("ascii")

    prepared, mime_type = agent_runtime._prepare_chat_capture(encoded, "image/png")
    prepared_bytes = base64.b64decode(prepared)
    with Image.open(io.BytesIO(prepared_bytes)) as image:
        assert image.size == (2560, 1440)
    assert mime_type == "image/jpeg"
    assert len(prepared_bytes) <= agent_runtime.MAX_CHAT_CAPTURE_BYTES


def test_capture_failure_is_reported_without_an_empty_model_followup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(
        tmp_path,
        parser="hermes",
        required_tool="computer_use",
        required_tool_arguments={"action": "capture", "mode": "vision"},
    )
    session = _Session([])
    agent.session = session  # type: ignore[assignment]
    events: list[dict[str, Any]] = []
    agent.on_event = events.append

    def fail_capture(*_args: Any, **_kwargs: Any) -> str:
        raise RuntimeError("Accessibility permission is not granted.")

    monkeypatch.setattr(agent_runtime, "execute_tool", fail_capture)

    with pytest.raises(
        RuntimeError,
        match=r"Screen capture failed: Accessibility permission is not granted\.",
    ):
        agent.run()

    assert session.payloads == []
    statuses = [event for event in events if event.get("type") == "tool_status"]
    assert [event["status"] for event in statuses] == ["running", "failed"]
    audit = (tmp_path / "agent" / "audit.jsonl").read_text(encoding="utf-8")
    assert json.loads(audit.splitlines()[-1])["outcome"] == "error"


def test_empty_final_answer_is_not_rendered_as_an_internal_placeholder(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    agent.session = _Session(
        [_Response(_sse({"choices": [{"delta": {"content": ""}}]}))]
    )  # type: ignore[assignment]

    result = agent.run()

    assert result["reply"] == "SGLang returned no visible answer."
    assert "empty response" not in result["reply"]


def test_screen_capture_arguments_target_nextcore_when_named() -> None:
    assert agent_runtime.required_tool_arguments_for_message(
        "Capture once and check if the NextCore window is visible."
    ) == {
        "action": "capture",
        "mode": "vision",
        "app": "token-workshed NextCore",
    }


def test_agent_runs_tool_call_loop_and_drops_reasoning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool_chunk = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": json.dumps(
                                    json.dumps({"query": "SGLang"})
                                ),
                            },
                        }
                    ]
                }
            }
        ]
    }
    final_chunk = {
        "choices": [
            {
                "delta": {
                    "reasoning_content": "private chain of thought",
                    "content": "Done.",
                }
            }
        ]
    }
    session = _Session([_Response(_sse(tool_chunk)), _Response(_sse(final_chunk))])
    agent = _agent(tmp_path, parser="hermes")
    agent.session = session  # type: ignore[assignment]
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_execute(name: str, args: dict[str, Any], **_: Any) -> str:
        calls.append((name, args))
        return '{"results":["public result"]}'

    monkeypatch.setattr(agent_runtime, "execute_tool", fake_execute)
    events: list[dict[str, Any]] = []
    agent.on_event = events.append

    result = agent.run()

    assert result["reply"] == "Done."
    assert calls == [("web_search", {"query": "SGLang"})]
    supplied_tools = session.payloads[0]["tools"]
    supplied_names = [item["function"]["name"] for item in supplied_tools]
    required_names = {
        item["function"]["name"] for item in agent_tools.TOOL_DEFINITIONS
    }
    assert required_names.issubset(supplied_names)
    assert len(supplied_names) == len(set(supplied_names))
    assert "private chain of thought" not in json.dumps(events)
    assert [event["status"] for event in events if event["type"] == "tool_status"] == [
        "running",
        "completed",
        "running",
    ]
    assert any(
        event.get("type") == "tool_status"
        and "preparing a reply" in event.get("message", "").lower()
        for event in events
    )


@pytest.mark.parametrize("failure_mode", ["exception", "empty"])
def test_search_results_are_shown_when_model_followup_fails_or_is_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    agent = _agent(
        tmp_path,
        required_tool="web_search",
        required_tool_arguments={"query": "SGLang"},
    )
    agent.messages[-1]["content"] = "联网搜索 SGLang"
    agent.on_event = (events := []).append
    monkeypatch.setattr(
        agent_runtime,
        "execute_tool",
        lambda *_args, **_kwargs: json.dumps(
            {
                "query": "SGLang",
                "results": [
                    {
                        "title": "SGLang 官方仓库",
                        "url": "https://github.com/sgl-project/sglang",
                    }
                ],
            },
            ensure_ascii=False,
        ),
    )
    if failure_mode == "exception":
        def fail_summary() -> dict[str, Any]:
            raise RuntimeError("summary request timed out")

        monkeypatch.setattr(agent, "_complete", fail_summary)
    else:
        monkeypatch.setattr(
            agent,
            "_complete",
            lambda: {"content": "", "tool_calls": [], "usage": {}},
        )

    result = agent.run()

    assert "搜索已完成" in result["reply"]
    assert "SGLang 官方仓库" in result["reply"]
    assert "https://github.com/sgl-project/sglang" in result["reply"]
    assert events[-1]["type"] == "answer_delta"


def test_post_tool_model_summary_is_short_and_has_a_visible_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_chunk = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-brief-summary",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": '{"query":"SGLang"}',
                            },
                        }
                    ]
                }
            }
        ]
    }
    final_chunk = {"choices": [{"delta": {"content": "Found it."}}]}
    session = _Session([_Response(_sse(tool_chunk)), _Response(_sse(final_chunk))])
    agent = _agent(tmp_path, parser="hermes", reasoning_parser="qwen3")
    agent.max_tokens = 1024
    agent.effective_max_tokens = 1024
    agent.session = session  # type: ignore[assignment]
    events: list[dict[str, Any]] = []
    agent.on_event = events.append
    monkeypatch.setattr(
        agent_runtime,
        "execute_tool",
        lambda *_args, **_kwargs: '{"query":"SGLang","results":[]}',
    )

    result = agent.run()

    assert result["reply"] == "Found it."
    assert session.payloads[1]["max_tokens"] == agent_runtime.MAX_POST_TOOL_SUMMARY_TOKENS
    assert session.payloads[1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert session.timeouts[1] == (5, agent_runtime.POST_TOOL_SUMMARY_TIMEOUT_SECONDS)
    assert any("preparing a reply" in event.get("message", "").lower() for event in events)


def test_reasoning_tags_are_removed_from_the_final_answer(tmp_path: Path) -> None:
    response = _Response(
        _sse(
            {
                "choices": [
                    {
                        "delta": {
                            "content": "<think>private reasoning</think>\nFinal Answer: public answer"
                        }
                    }
                ]
            }
        )
    )
    agent = _agent(tmp_path)
    agent.session = _Session([response])  # type: ignore[assignment]
    agent.answer_sanitizer = lambda raw: css_svg_ui._sanitize_assistant_output_text(
        css_svg_ui._split_reasoning_and_answer(raw)[1]
    )
    events: list[dict[str, Any]] = []
    agent.on_event = events.append

    result = agent.run()

    assert result["reply"] == "public answer"
    assert "private reasoning" not in json.dumps(events)


def test_think_tag_filter_handles_split_tags_and_unicode() -> None:
    answer_filter = agent_runtime._ThinkTagFilter()

    assert answer_filter.feed("Straße stays. <thi") == "Straße stays. "
    assert answer_filter.feed("nk>private thought</thi") == ""
    assert answer_filter.feed("nk> Visible ending.") == " Visible ending."


def test_visible_answer_streams_before_completion_without_reasoning(
    tmp_path: Path,
) -> None:
    first_chunk_sent = threading.Event()
    allow_completion = threading.Event()

    class _PausedResponse(_Response):
        def iter_lines(self, **_: Any):
            yield _sse(
                {"choices": [{"delta": {"content": "Public answer. <think>hidden"}}]}
            )[0]
            first_chunk_sent.set()
            allow_completion.wait(timeout=3)
            yield _sse(
                {
                    "choices": [
                        {
                            "delta": {
                                "reasoning_content": "separate hidden reasoning",
                                "content": " thought</think> More text.",
                            }
                        }
                    ]
                }
            )[0]
            yield "data: [DONE]"

    response = _PausedResponse([])
    agent = _agent(tmp_path, stream_visible_answer=True)
    agent.session = _Session([response])  # type: ignore[assignment]
    agent.answer_sanitizer = lambda raw: css_svg_ui._sanitize_assistant_output_text(
        css_svg_ui._split_reasoning_and_answer(raw)[1]
    )
    events: queue.Queue[dict[str, Any]] = queue.Queue()
    agent.on_event = events.put
    worker_error: list[BaseException] = []

    def run() -> None:
        try:
            agent.run()
        except BaseException as exc:
            worker_error.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        assert first_chunk_sent.wait(timeout=2)
        first_event = events.get(timeout=2)
        assert first_event == {"type": "answer_delta", "text": "Public answer. "}
        assert worker.is_alive()
        assert allow_completion.is_set() is False
    finally:
        allow_completion.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert not worker_error
    remaining_events: list[dict[str, Any]] = []
    while not events.empty():
        remaining_events.append(events.get_nowait())
    assert "hidden" not in json.dumps([first_event, *remaining_events])
    assert "More text." in json.dumps(remaining_events)


def test_streamed_answer_can_be_replaced_by_sanitized_final(tmp_path: Path) -> None:
    response = _Response(_sse({"choices": [{"delta": {"content": "draft answer"}}]}))
    agent = _agent(tmp_path, stream_visible_answer=True)
    agent.session = _Session([response])  # type: ignore[assignment]
    agent.answer_sanitizer = lambda _raw: "corrected final answer"
    events: list[dict[str, Any]] = []
    agent.on_event = events.append

    result = agent.run()

    assert result["reply"] == "corrected final answer"
    assert events == [
        {"type": "answer_delta", "text": "draft answer"},
        {"type": "answer_replace", "text": "corrected final answer"},
    ]


def test_stream_cancellation_closes_sglang_response(tmp_path: Path) -> None:
    entered = threading.Event()
    response_closed = threading.Event()

    class _BlockingResponse(_Response):
        def iter_lines(self, **_: Any):
            entered.set()
            response_closed.wait(timeout=5)
            return
            yield ""

        def close(self) -> None:
            super().close()
            response_closed.set()

    response = _BlockingResponse([])
    agent = _agent(tmp_path)
    agent.session = _Session([response])  # type: ignore[assignment]
    worker_error: list[BaseException] = []

    def run() -> None:
        try:
            agent.run()
        except BaseException as exc:
            worker_error.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    assert entered.wait(timeout=2)
    agent.cancel_event.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert worker_error and isinstance(worker_error[0], AgentCancelledError)
    assert response.closed.is_set()


def test_cancel_during_tool_returns_without_waiting_for_blocking_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(tmp_path)
    started = threading.Event()

    def blocking_tool(*_: Any, **kwargs: Any) -> str:
        started.set()
        kwargs["cancel_event"].wait(timeout=5)
        raise InterruptedError("stopped")

    monkeypatch.setattr(agent_runtime, "execute_tool", blocking_tool)
    errors: list[BaseException] = []

    def run_tool() -> None:
        try:
            agent._execute_tool_cancellable("terminal", {"command": "sleep 5"})
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=run_tool, daemon=True)
    worker.start()
    assert started.wait(timeout=2)
    started_at = time.monotonic()
    agent.cancel_event.set()
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert time.monotonic() - started_at < 1
    assert errors and isinstance(errors[0], AgentCancelledError)


def test_cancel_during_browser_action_closes_the_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(tmp_path)
    started = threading.Event()
    stopped = threading.Event()

    def blocking_browser(*_: Any, **__: Any) -> str:
        started.set()
        stopped.wait(timeout=5)
        return "{}"

    monkeypatch.setattr(agent_runtime, "execute_tool", blocking_browser)
    monkeypatch.setattr(agent_runtime, "interrupt_computer_use", stopped.set)
    errors: list[BaseException] = []

    def run_tool() -> None:
        try:
            agent._execute_tool_cancellable("browser", {"action": "inspect"})
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=run_tool, daemon=True)
    worker.start()
    assert started.wait(timeout=2)
    agent.cancel_event.set()
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert stopped.wait(timeout=1)
    assert errors and isinstance(errors[0], AgentCancelledError)


def test_cancel_during_web_fetch_closes_only_that_run_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancel_event = threading.Event()
    entered = threading.Event()
    closed = threading.Event()

    class _BlockingWebResponse:
        encoding = "utf-8"
        headers: dict[str, str] = {}
        is_redirect = False
        is_permanent_redirect = False
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def iter_content(self, _chunk_size: int):
            entered.set()
            closed.wait(timeout=5)
            return
            yield b""

        def close(self) -> None:
            closed.set()

    response = _BlockingWebResponse()
    monkeypatch.setattr(agent_tools.requests, "get", lambda *_args, **_kwargs: response)
    errors: list[BaseException] = []

    def fetch() -> None:
        try:
            agent_tools._web_fetch("https://example.com", cancel_event=cancel_event)  # type: ignore[arg-type]
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=fetch, daemon=True)
    worker.start()
    assert entered.wait(timeout=2)
    cancel_event.set()
    agent_tools.interrupt_web_requests(cancel_event)
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert closed.is_set()
    assert errors and isinstance(errors[0], InterruptedError)


def test_audit_stores_only_argument_digest_and_outcome(tmp_path: Path) -> None:
    audit_tool_call(
        app_data_dir=tmp_path,
        run_id="run",
        tool="file",
        arguments={"path": "/Users/me/secret.txt", "content": "private body"},
        outcome="ok",
        duration_ms=1,
    )

    audit = (tmp_path / "agent" / "audit.jsonl").read_text(encoding="utf-8")
    assert "/Users/me/secret.txt" not in audit
    assert "private body" not in audit
    assert "arguments_sha256" in audit


def test_file_search_checks_cancellation_while_walking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cancel_event = threading.Event()

    def walk(path: Path, **_kwargs: Any):
        yield str(path), ["subdir"], ["first.txt"]
        cancel_event.set()
        yield str(path / "subdir"), [], ["second.txt"]

    monkeypatch.setattr(agent_tools.os, "walk", walk)

    with pytest.raises(InterruptedError, match="File search stopped"):
        agent_tools._file(
            {"action": "search", "path": str(tmp_path), "pattern": "*.md"},
            cancel_event=cancel_event,
        )


def test_sglang_runtime_fails_clearly_without_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sglang_runtime.sys, "platform", "darwin")
    monkeypatch.setattr(sglang_runtime.platform, "machine", lambda: "arm64")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
        ),
    )

    with pytest.raises(SystemExit, match="no accessible Metal/MPS device"):
        sglang_runtime._require_metal_device("mlx")


def test_browser_tool_uses_exact_window_and_isolated_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class _Backend:
        def call_tool(
            self, name: str, args: dict[str, Any], *, timeout: float
        ) -> dict[str, Any]:
            calls.append((name, args))
            if name == "browser_prepare":
                return {"structuredContent": {"status": "prepared", "prepared_pid": 42}}
            if name == "get_browser_state":
                return {
                    "structuredContent": {
                        "target_id": "opaque-target",
                        "tabs": [
                            {
                                "tab_id": "opaque-tab",
                                "title": "Example",
                                "url": "https://example.com",
                            }
                        ],
                        "cookies": ["secret"],
                    }
                }
            return {"structuredContent": {"status": "ok"}}

    monkeypatch.setattr(
        agent_tools, "_get_computer_backend", lambda _path, **_kwargs: _Backend()
    )
    prepared = agent_tools.execute_tool(
        "browser",
        {"action": "prepare_isolated", "pid": 42},
        app_data_dir=tmp_path,
        cancel_event=threading.Event(),
    )
    state = agent_tools.execute_tool(
        "browser",
        {"action": "inspect", "pid": 42, "window_id": 99},
        app_data_dir=tmp_path,
        cancel_event=threading.Event(),
    )

    assert json.loads(prepared)["prepared_pid"] == 42
    assert "cookies" not in state
    assert calls[0] == (
        "browser_prepare",
        {
            "pid": 42,
            "allow_launch": True,
            "profile": {"mode": "isolated_new"},
            "session": "",
        },
    )
    assert calls[1][1]["pid"] == 42 and calls[1][1]["window_id"] == 99


def test_computer_tool_returns_accessibility_data_and_runs_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class _Backend:
        def capture(self, *, mode: str, app: str | None = None) -> CaptureResult:
            calls.append(("capture", {"mode": mode, "app": app}))
            return CaptureResult(
                mode=mode,
                width=100,
                height=80,
                png_b64="private screenshot bytes",
                elements=[UIElement(index=1, role="AXButton", label="Continue")],
            )

        def click(self, **kwargs: Any) -> ActionResult:
            calls.append(("click", kwargs))
            return ActionResult(ok=True, action="click", message="Clicked Continue")

    monkeypatch.setattr(
        agent_tools, "_get_computer_backend", lambda *_args, **_kwargs: _Backend()
    )
    cancel_event = threading.Event()
    captured = agent_tools.execute_tool(
        "computer_use",
        {"action": "capture", "mode": "ax", "app": "Example"},
        app_data_dir=tmp_path,
        cancel_event=cancel_event,
    )
    clicked = agent_tools.execute_tool(
        "computer_use",
        {"action": "click", "element": 1},
        app_data_dir=tmp_path,
        cancel_event=cancel_event,
    )

    assert "Continue" in captured
    assert "private screenshot bytes" not in captured
    assert json.loads(clicked)["ok"] is True
    assert calls == [
        ("capture", {"mode": "ax", "app": "Example"}),
        ("click", {"element": 1, "x": None, "y": None}),
    ]


def test_vision_capture_fails_when_driver_returns_no_image_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Backend:
        def capture(self, **_: Any) -> CaptureResult:
            return CaptureResult(mode="vision", width=1600, height=900)

    monkeypatch.setattr(
        agent_tools, "_get_computer_backend", lambda *_args, **_kwargs: _Backend()
    )

    with pytest.raises(RuntimeError, match="returned no image pixels"):
        agent_tools._computer_use(
            {"action": "capture", "mode": "vision"},
            tmp_path,
            threading.Event(),
        )


def test_computer_use_async_bridge_does_not_depend_on_hermes_runtime() -> None:
    bridge = _AsyncBridge()
    bridge.start()

    async def current_loop_is_running() -> bool:
        return asyncio.get_running_loop().is_running()

    try:
        assert bridge.run(current_loop_is_running()) is True
    finally:
        bridge.stop()


def test_computer_backend_uses_its_app_managed_driver_path(tmp_path: Path) -> None:
    managed_driver = tmp_path / "agent" / "bin" / "cua-driver"
    backend = CuaDriverBackend(driver_command=str(managed_driver))

    assert backend._driver_command == str(managed_driver)
    assert backend._session._driver_command == str(managed_driver)


def test_macos_cua_invocation_uses_direct_embedded_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from token_workshed_nextcore.agent.computer_use import cua_backend

    monkeypatch.setattr(cua_backend.sys, "platform", "darwin")
    monkeypatch.setenv("TOKEN_WORKSHED_APP_BUNDLE_ID", "com.example.nextcore")
    monkeypatch.setattr(
        cua_backend.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {"mcp_invocation": {"command": "/managed/cua-driver", "args": ["mcp"]}}
            ),
        ),
    )

    command, args = _resolve_mcp_invocation("/managed/cua-driver")

    assert command == "/managed/cua-driver"
    assert args == [
        "mcp",
        "--direct",
        "--embedded",
        "--host-bundle-id",
        "com.example.nextcore",
    ]
    env = cua_driver_child_env({"TOKEN_WORKSHED_APP_BUNDLE_ID": "com.example.nextcore"})
    assert env["CUA_DRIVER_EMBEDDED"] == "1"
    assert env["CUA_DRIVER_HOST_BUNDLE_ID"] == "com.example.nextcore"


def test_macos_cua_invocation_fallback_still_uses_direct_embedded_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from token_workshed_nextcore.agent.computer_use import cua_backend

    monkeypatch.setattr(cua_backend.sys, "platform", "darwin")
    monkeypatch.setenv("TOKEN_WORKSHED_APP_BUNDLE_ID", "com.example.nextcore")
    monkeypatch.setattr(
        cua_backend.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError()),
    )

    assert _resolve_mcp_invocation("/managed/cua-driver") == (
        "/managed/cua-driver",
        ["mcp", "--direct", "--embedded", "--host-bundle-id", "com.example.nextcore"],
    )


def test_macos_permissions_request_both_grants_without_system_ui_in_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(macos_permissions.sys, "platform", "darwin")
    monkeypatch.setattr(
        macos_permissions,
        "_request_accessibility_permission",
        lambda: calls.append("accessibility") or True,
    )
    monkeypatch.setattr(
        macos_permissions,
        "_request_screen_recording_permission",
        lambda: calls.append("screen recording") or True,
    )

    macos_permissions.request_computer_use_permissions()

    assert calls == ["accessibility", "screen recording"]


def test_macos_permissions_explain_missing_grants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(macos_permissions.sys, "platform", "darwin")
    monkeypatch.setattr(
        macos_permissions, "_request_accessibility_permission", lambda: False
    )
    monkeypatch.setattr(
        macos_permissions, "_request_screen_recording_permission", lambda: True
    )

    with pytest.raises(
        macos_permissions.MacOSPermissionRequiredError, match="Accessibility permission"
    ):
        macos_permissions.request_computer_use_permissions()


def test_read_only_capture_can_start_driver_to_register_missing_permissions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from token_workshed_nextcore.agent.computer_use import cua_backend

    monkeypatch.setattr(agent_tools, "_CUA_BACKEND", None)
    monkeypatch.setattr(agent_tools, "_CUA_PERMISSION_ERROR", None)
    monkeypatch.setattr(
        agent_tools,
        "request_computer_use_permissions",
        lambda: (_ for _ in ()).throw(
            macos_permissions.MacOSPermissionRequiredError("grant required")
        ),
    )
    calls: list[str] = []
    monkeypatch.setattr(
        agent_tools,
        "ensure_driver",
        lambda *_args, **_kwargs: calls.append("download") or Path("/managed/cua-driver"),
    )

    class _Backend:
        def __init__(
            self, *, driver_command: str, allow_existing_profile: bool = False
        ) -> None:
            _ = allow_existing_profile
            calls.append(f"backend:{driver_command}")

        def start(self) -> None:
            calls.append("start")

        def stop(self) -> None:
            calls.append("stop")

    monkeypatch.setattr(cua_backend, "CuaDriverBackend", _Backend)

    backend = agent_tools._get_computer_backend(
        tmp_path,
        cancel_event=threading.Event(),
        allow_missing_permissions=True,
    )

    assert isinstance(backend, _Backend)
    assert agent_tools._CUA_PERMISSION_ERROR == "grant required"
    assert calls == ["download", "backend:/managed/cua-driver", "start"]


def test_missing_permissions_allow_one_capture_but_block_computer_actions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Backend:
        def capture(self, **_: Any) -> CaptureResult:
            return CaptureResult(mode="ax", width=800, height=600, elements=[])

        def click(self, **_: Any) -> ActionResult:
            pytest.fail("computer actions must stay blocked without permissions")

    monkeypatch.setattr(agent_tools, "_CUA_BACKEND", _Backend())
    monkeypatch.setattr(agent_tools, "_CUA_PERMISSION_ERROR", "grant required")
    monkeypatch.setattr(agent_tools, "_CUA_PERMISSION_CAPTURE_PROBE_USED", False)

    result = agent_tools._computer_use(
        {"action": "capture"}, tmp_path, threading.Event()
    )
    assert json.loads(result)["width"] == 800
    assert agent_tools._CUA_PERMISSION_CAPTURE_PROBE_USED is True

    with pytest.raises(macos_permissions.MacOSPermissionRequiredError):
        agent_tools._computer_use({"action": "click"}, tmp_path, threading.Event())


def test_computer_tool_reports_missing_macos_permissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Backend:
        def capture(self, **_: Any) -> CaptureResult:
            return CaptureResult(mode="ax", width=0, height=0)

    monkeypatch.setattr(
        agent_tools, "_get_computer_backend", lambda *_args, **_kwargs: _Backend()
    )

    with pytest.raises(
        RuntimeError, match="System Settings > Privacy & Security"
    ) as error:
        agent_tools.execute_tool(
            "computer_use",
            {"action": "capture"},
            app_data_dir=tmp_path,
            cancel_event=threading.Event(),
        )
    assert "Accessibility and Screen Recording" in str(error.value)


def test_managed_driver_download_verifies_pinned_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary_contents = b"test-only CUA driver executable"
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
        member = tarfile.TarInfo("release/cua-driver")
        member.size = len(binary_contents)
        archive.addfile(member, io.BytesIO(binary_contents))
    archive_contents = archive_buffer.getvalue()
    expected_digest = hashlib.sha256(archive_contents).hexdigest()
    opened_requests: list[Any] = []

    class _Response:
        def __init__(self) -> None:
            self.stream = io.BytesIO(archive_contents)

        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def read(self, size: int = -1) -> bytes:
            return self.stream.read(size)

    def fake_urlopen(request: Any, **_: Any) -> _Response:
        opened_requests.append(request)
        return _Response()

    monkeypatch.setattr(driver_manager.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(driver_manager.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(driver_manager, "CUA_DRIVER_SHA256", expected_digest)
    monkeypatch.setattr(driver_manager.urllib.request, "urlopen", fake_urlopen)

    assert driver_manager.CUA_DRIVER_VERSION == "0.28.1"
    path = driver_manager.ensure_driver(tmp_path)
    cached_path = driver_manager.ensure_driver(tmp_path)

    assert path == cached_path == driver_manager.managed_driver_path(tmp_path)
    assert path.read_bytes() == binary_contents
    assert os.access(path, os.X_OK)
    assert len(opened_requests) == 1
    assert driver_manager.CUA_DRIVER_URL in opened_requests[0].full_url

    path.write_bytes(b"modified managed executable")
    repaired_path = driver_manager.ensure_driver(tmp_path)

    manifest_path = path.with_name("cua-driver.manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert repaired_path == path
    assert path.read_bytes() == binary_contents
    assert manifest["version"] == driver_manager.CUA_DRIVER_VERSION
    assert manifest["archive_sha256"] == expected_digest
    assert manifest["binary_sha256"] == hashlib.sha256(binary_contents).hexdigest()
    assert len(opened_requests) == 2


def test_managed_driver_rejects_checksum_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_contents = b"not the pinned driver archive"

    class _Response:
        def __init__(self) -> None:
            self.stream = io.BytesIO(archive_contents)

        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def read(self, size: int = -1) -> bytes:
            return self.stream.read(size)

    monkeypatch.setattr(driver_manager.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(driver_manager.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(driver_manager, "CUA_DRIVER_SHA256", "0" * 64)
    monkeypatch.setattr(
        driver_manager.urllib.request, "urlopen", lambda *_args, **_kwargs: _Response()
    )

    with pytest.raises(
        driver_manager.DriverUnavailableError, match="checksum did not match"
    ):
        driver_manager.ensure_driver(tmp_path)
    assert not driver_manager.managed_driver_path(tmp_path).exists()


def test_granite_models_use_sglang_hermes_parser_for_native_tool_format() -> None:
    assert (
        desktop_ui._detect_tool_call_parser("ibm-granite/granite-4.0-1b")
        == "hermes"
    )


def test_manager_selects_parser_from_model_family_without_runtime_probe() -> None:
    manager = desktop_ui.LocalModelServerManager.__new__(
        desktop_ui.LocalModelServerManager
    )
    assert manager._effective_tool_call_parser("ibm-granite/granite-4.0-1b") == "hermes"
    assert manager._effective_tool_call_parser("openai-community/gpt2") == "auto"


@pytest.mark.parametrize(
    "command",
    [
        "env sudo rm -rf /tmp/example",
        "osascript -e 'do shell script \\\"whoami\\\" with administrator privileges'",
    ],
)
def test_terminal_rejects_privilege_escalation(command: str) -> None:
    with pytest.raises(ValueError, match="Privilege escalation is disabled"):
        agent_tools._terminal({"command": command}, threading.Event())


def test_terminal_child_environment_filters_credentials_and_keeps_shell_basics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = {
        "HOME": "/Users/example",
        "PATH": "/opt/homebrew/bin:/usr/bin:/bin",
        "LANG": "en_US.UTF-8",
        "OPENAI_API_KEY": "secret",
        "AWS_ACCESS_KEY_ID": "secret",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "AWS_SESSION_TOKEN": "secret",
        "GITHUB_PAT": "secret",
        "TOKEN_WORKSHED_AGENT_UI_TOKEN": "secret",
        "SSH_AUTH_SOCK": "/tmp/ssh-agent.sock",
        "DATABASE_URL": "postgres://user:secret@db.example.test/app",
        "HTTPS_PROXY": "http://user:secret@proxy.example.test:8080",
        "PIP_INDEX_URL": "https://packages.example.test/simple?access_token=secret",
        "DYLD_INSERT_LIBRARIES": "/tmp/injected.dylib",
        "BASH_ENV": "/Users/example/.bash_env",
        "ZDOTDIR": "/Users/example/.zsh",
    }
    monkeypatch.setattr(agent_tools.os, "environ", environment)

    child_env = agent_tools._safe_child_env()

    assert child_env["HOME"] == "/Users/example"
    assert child_env["PATH"] == "/opt/homebrew/bin:/usr/bin:/bin"
    assert child_env["LANG"] == "en_US.UTF-8"
    assert not set(environment).intersection(child_env) - {
        "HOME",
        "PATH",
        "LANG",
    }

    driver_env = _safe_subprocess_env(cua_driver_child_env(environment))
    assert driver_env["PATH"] == environment["PATH"]
    assert driver_env["CUA_DRIVER_RS_TELEMETRY_ENABLED"] == "0"
    assert driver_env["CUA_DRIVER_HOST_BUNDLE_ID"] == "com.taylorhe.tokenworkshed.nextcore"
    assert not set(environment).intersection(driver_env) - {
        "HOME",
        "PATH",
        "LANG",
    }


def test_terminal_cancel_stops_the_running_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancel_event = threading.Event()
    process_holder: list[Any] = []
    real_popen = agent_tools.subprocess.Popen

    def tracked_popen(*args: Any, **kwargs: Any):
        process = real_popen(*args, **kwargs)
        process_holder.append(process)
        return process

    monkeypatch.setattr(agent_tools.subprocess, "Popen", tracked_popen)
    original_monotonic = time.monotonic()
    timer = threading.Timer(0.3, cancel_event.set)
    timer.start()
    try:
        with pytest.raises(InterruptedError, match="Command stopped"):
            agent_tools._terminal({"command": "sleep 30"}, cancel_event)
    finally:
        timer.cancel()

    assert process_holder
    assert process_holder[0].args[:3] == ["/bin/zsh", "-f", "-c"]
    assert process_holder[0].poll() is not None
    assert time.monotonic() - original_monotonic < 3


def test_app_stream_is_the_only_chat_route_that_can_enable_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen_parsers: list[str | None] = []
    seen_messages: list[list[dict[str, Any]]] = []
    seen_streaming_flags: list[bool] = []
    seen_required_tools: list[str | None] = []
    seen_required_arguments: list[dict[str, Any] | None] = []

    class _Manager:
        def running_models(self) -> list[dict[str, str]]:
            return [{"model": "test-model", "tool_call_parser": "hermes"}]

        def reasoning_parser_for_model(self, _model: str) -> str:
            return "qwen3"

    class _Agent:
        def __init__(self, **kwargs: Any) -> None:
            seen_parsers.append(kwargs["tool_parser"])
            seen_messages.append(kwargs["messages"])
            seen_streaming_flags.append(bool(kwargs.get("stream_visible_answer")))
            seen_required_tools.append(kwargs.get("required_tool"))
            seen_required_arguments.append(kwargs.get("required_tool_arguments"))
            self.kwargs = kwargs

        def run(self) -> dict[str, Any]:
            callback = self.kwargs.get("on_event") or (lambda _event: None)
            callback({"type": "answer_delta", "text": "ok"})
            return {"reply": "ok", "model": "test-model", "metrics": {}}

    monkeypatch.setattr(css_svg_ui, "SGLangAgent", _Agent)
    app = css_svg_ui.create_app("http://127.0.0.1:8000", 64, 0.2)
    app.state.agent_ui_token = "native-window-secret"
    app.state.agent_app_data_dir = tmp_path
    app.state.model_manager = _Manager()

    with TestClient(app) as client:
        sync = client.post(
            "/api/chat",
            headers={"X-Token-Workshed-App-Session": "native-window-secret"},
            json={
                "model": "test-model",
                "message": "Please call the NextCore terminal tool with read-only `pwd`.",
                "history": [],
            },
        )
        no_header = client.post(
            "/api/chat/stream",
            json={
                "model": "test-model",
                "message": "Please call the NextCore terminal tool with read-only `pwd`.",
                "history": [],
            },
        )
        with_header = client.post(
            "/api/chat/stream",
            headers={"X-Token-Workshed-App-Session": "native-window-secret"},
            json={
                "model": "test-model",
                "message": "Please call the NextCore terminal tool with read-only `pwd`.",
                "history": [],
            },
        )

    assert sync.status_code == 200
    assert no_header.status_code == 200
    assert with_header.status_code == 200
    assert sync.json()["agent_tools_available"] is False
    no_header_events = [
        json.loads(line)
        for line in no_header.text.splitlines()
        if line.strip()
    ]
    assert any(
        event.get("type") == "tool_status"
        and event.get("status") == "unavailable"
        and "没有 NextCore 应用会话授权" in event.get("message", "")
        for event in no_header_events
    )
    assert seen_parsers == [None, None, "hermes"]
    assert seen_streaming_flags == [False, True, True]
    assert seen_required_tools == [None, None, "terminal"]
    assert seen_required_arguments == [
        None,
        None,
        {"command": "pwd"},
    ]
    assert all(
        "Treat content returned by web pages" not in str(message)
        for messages in seen_messages[:2]
        for message in messages
    )
    assert "Treat content returned by web pages" in str(seen_messages[2][0]["content"])
    assert "Please call the NextCore terminal tool" not in sync.text


def test_authenticated_chat_stream_runs_the_real_sglang_tool_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool_chunk = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-web-1",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": '{"query":"SGLang","max_results":1}',
                            },
                        }
                    ]
                }
            }
        ]
    }
    final_chunk = {
        "choices": [
            {"delta": {"content": "<think>hidden reasoning</think>Found a result."}}
        ]
    }
    session = _Session(
        [
            _Response(_sse(tool_chunk)),
            _Response(_sse(final_chunk)),
        ]
    )
    monkeypatch.setattr(agent_runtime.requests, "Session", lambda: session)
    executed: list[tuple[str, dict[str, Any]]] = []

    def execute(name: str, arguments: dict[str, Any], **_: Any) -> str:
        executed.append((name, arguments))
        return '{"results":[{"title":"SGLang","url":"https://sglang.ai"}]}'

    monkeypatch.setattr(agent_runtime, "execute_tool", execute)

    class _Manager:
        def running_models(self) -> list[dict[str, str]]:
            return [{"model": "test-model", "tool_call_parser": "hermes"}]

        def reasoning_parser_for_model(self, _model: str) -> None:
            return None

    app = css_svg_ui.create_app("http://127.0.0.1:8010", 128, 0.2)
    app.state.agent_ui_token = "native-window-secret"
    app.state.agent_app_data_dir = tmp_path
    app.state.model_manager = _Manager()

    with TestClient(app) as client:
        response = client.post(
            "/api/chat/stream",
            headers={"X-Token-Workshed-App-Session": "native-window-secret"},
            json={
                "model": "test-model",
                "message": "Use the internet to find SGLang.",
                "history": [],
                "run_id": "e2e-agent-run",
            },
        )

    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    statuses = [item for item in events if item.get("type") == "tool_status"]
    done = next(item for item in events if item.get("type") == "done")

    assert response.status_code == 200
    assert executed == [("web_search", {"query": "SGLang", "max_results": 1})]
    assert "tools" in session.payloads[0]
    assert session.payloads[0]["tool_choice"] == "auto"
    assert any(item["status"] == "running" for item in statuses)
    assert any(item["status"] == "completed" for item in statuses)
    assert done["reply"] == "Found a result."
    assert "hidden reasoning" not in response.text
    assert session.payloads[1]["messages"][-1]["role"] == "tool"


def test_authenticated_chat_stream_returns_search_results_if_model_summary_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FailingSession:
        def post(self, *_args: Any, **_kwargs: Any) -> None:
            raise agent_runtime.requests.Timeout("read timed out")

    monkeypatch.setattr(agent_runtime.requests, "Session", _FailingSession)
    monkeypatch.setattr(
        agent_runtime,
        "execute_tool",
        lambda *_args, **_kwargs: json.dumps(
            {
                "query": "SGLang",
                "results": [
                    {
                        "title": "SGLang official GitHub",
                        "url": "https://github.com/sgl-project/sglang",
                    }
                ],
            }
        ),
    )
    app = css_svg_ui.create_app("http://127.0.0.1:8010", 128, 0.2)
    app.state.agent_ui_token = "native-window-secret"
    app.state.agent_app_data_dir = tmp_path

    with TestClient(app) as client:
        response = client.post(
            "/api/chat/stream",
            headers={"X-Token-Workshed-App-Session": "native-window-secret"},
            json={
                "model": "test-model",
                "message": "联网搜索 SGLang",
                "history": [],
                "run_id": "summary-fallback-run",
            },
        )

    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    statuses = [event for event in events if event.get("type") == "tool_status"]
    answer = next(event for event in events if event.get("type") == "answer_delta")
    done = next(event for event in events if event.get("type") == "done")

    assert response.status_code == 200
    assert any("正在整理结果并生成回复" in event.get("message", "") for event in statuses)
    assert "搜索已完成" in answer["text"]
    assert "https://github.com/sgl-project/sglang" in done["reply"]
    assert not any(event.get("type") == "error" for event in events)


def test_authenticated_capture_stream_includes_inline_image_attachment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _Session(
        [_Response(_sse({"choices": [{"delta": {"content": "The NextCore window is visible."}}]}))]
    )
    monkeypatch.setattr(agent_runtime.requests, "Session", lambda: session)
    screenshot = _ONE_PIXEL_PNG

    def capture(*_args: Any, **_kwargs: Any) -> str:
        return json.dumps(
            {
                "app": "token-workshed NextCore",
                "window_title": "Chat",
                "elements": [],
                "width": 2,
                "height": 2,
                "screenshot_b64": screenshot,
                "screenshot_mime_type": "image/png",
            }
        )

    monkeypatch.setattr(agent_runtime, "execute_tool", capture)

    class _Manager:
        def running_models(self) -> list[dict[str, str]]:
            return [{"model": "test-model", "tool_call_parser": "hermes"}]

        def reasoning_parser_for_model(self, _model: str) -> str | None:
            return None

    app = css_svg_ui.create_app("http://127.0.0.1:8010", 128, 0.2)
    app.state.agent_ui_token = "native-window-secret"
    app.state.agent_app_data_dir = tmp_path
    app.state.model_manager = _Manager()

    with TestClient(app) as client:
        response = client.post(
            "/api/chat/stream",
            headers={"X-Token-Workshed-App-Session": "native-window-secret"},
            json={
                "model": "test-model",
                "message": "只截取当前屏幕一次，并只回答 NextCore 窗口是否可见。",
                "history": [],
            },
        )

    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    attachment = next(event for event in events if event.get("type") == "tool_attachment")
    tool_result = session.payloads[0]["messages"][-1]["content"]
    assert response.status_code == 200
    assert attachment["attachment"] == {
        "mime_type": "image/png",
        "base64": screenshot,
        "app": "token-workshed NextCore",
        "window_title": "Chat",
    }
    assert "screenshot_b64" not in tool_result
    assert screenshot not in tool_result
    assert any(event.get("type") == "done" for event in events)


def test_capture_only_run_reports_unrenderable_image_instead_of_claiming_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(
        tmp_path,
        parser="hermes",
        required_tool="computer_use",
        required_tool_arguments={"action": "capture", "mode": "vision"},
    )
    events: list[dict[str, Any]] = []
    agent.on_event = events.append

    def capture(*_args: Any, **_kwargs: Any) -> str:
        return json.dumps(
            {
                "app": "token-workshed NextCore",
                "width": 1600,
                "height": 900,
                "screenshot_b64": "not-an-image",
                "screenshot_mime_type": "image/png",
            }
        )

    monkeypatch.setattr(agent_runtime, "execute_tool", capture)

    with pytest.raises(RuntimeError, match="could not prepare the image for chat display"):
        agent.run()

    assert not any(event.get("type") == "tool_attachment" for event in events)
    assert any(
        event.get("type") == "tool_status" and event.get("status") == "failed"
        for event in events
    )


def test_authenticated_capture_failure_is_streamed_as_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Manager:
        def running_models(self) -> list[dict[str, str]]:
            return [{"model": "test-model", "tool_call_parser": "hermes"}]

        def reasoning_parser_for_model(self, _model: str) -> None:
            return None

    def fail_capture(*_args: Any, **_kwargs: Any) -> str:
        raise RuntimeError("Accessibility permission is not granted.")

    monkeypatch.setattr(agent_runtime, "execute_tool", fail_capture)
    app = css_svg_ui.create_app("http://127.0.0.1:8010", 128, 0.2)
    app.state.agent_ui_token = "native-window-secret"
    app.state.agent_app_data_dir = tmp_path
    app.state.model_manager = _Manager()

    with TestClient(app) as client:
        response = client.post(
            "/api/chat/stream",
            headers={"X-Token-Workshed-App-Session": "native-window-secret"},
            json={
                "model": "test-model",
                "message": "只截取当前屏幕一次，并只回答 NextCore 窗口是否可见。",
                "history": [],
            },
        )

    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    statuses = [item for item in events if item.get("type") == "tool_status"]
    error = next(item for item in events if item.get("type") == "error")
    assert response.status_code == 200
    assert [item["status"] for item in statuses] == ["running", "failed"]
    assert error["message"] == "Screen capture failed: Accessibility permission is not granted."
    assert not any(item.get("type") == "done" for item in events)


def test_chat_cancel_endpoint_requires_app_token_and_stops_active_run(
    tmp_path: Path,
) -> None:
    app = css_svg_ui.create_app("http://127.0.0.1:8010", 128, 0.2)
    app.state.agent_ui_token = "native-window-secret"
    event = app.state.agent_runs.register("cancel-test-run")

    with TestClient(app) as client:
        denied = client.post("/api/chat/cancel", json={"run_id": "cancel-test-run"})
        stopped = client.post(
            "/api/chat/cancel",
            headers={"X-Token-Workshed-App-Session": "native-window-secret"},
            json={"run_id": "cancel-test-run"},
        )

    assert denied.status_code == 401
    assert stopped.status_code == 200
    assert stopped.json()["cancelled"] is True
    assert event.is_set()


def test_external_ide_agent_surface_is_always_disabled() -> None:
    bridge = IDEBridge(
        get_models=lambda: ["test-model"],
        get_active_model=lambda: "test-model",
        switch_model=lambda _model: (True, "switched"),
        get_server_url=lambda: "http://127.0.0.1:8000",
        default_server_url="http://127.0.0.1:8000",
        default_max_tokens=64,
        default_temperature=0.2,
    )
    app = css_svg_ui.FastAPI()
    register_ide_bridge_routes(app, bridge)

    with TestClient(app) as client:
        status = client.get("/api/ide/v2/status")
        session = client.post(
            "/api/ide/v2/sessions",
            json={"title": "external", "model": "test-model", "cwd": "/tmp"},
        )

    assert status.status_code == 200
    assert status.json()["agent_access"] is False
    assert session.status_code == 403
    assert "only from the Token Workshed NextCore chat" in session.json()["detail"]
