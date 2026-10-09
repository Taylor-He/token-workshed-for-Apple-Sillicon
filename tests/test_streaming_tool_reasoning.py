# SPDX-License-Identifier: Apache-2.0
"""Regression tests for reasoning-aware streaming tool calls."""

import asyncio
import json
from types import SimpleNamespace


def test_qwen_reasoning_stream_still_emits_structured_tool_calls(monkeypatch):
    from vllm_mlx import server
    from vllm_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

    outputs = [
        SimpleNamespace(
            new_text="<think>",
            finished=False,
            finish_reason=None,
            prompt_tokens=12,
            completion_tokens=1,
        ),
        SimpleNamespace(
            new_text="Use the directory tool.",
            finished=False,
            finish_reason=None,
            prompt_tokens=12,
            completion_tokens=6,
        ),
        SimpleNamespace(
            new_text="</think>",
            finished=False,
            finish_reason=None,
            prompt_tokens=12,
            completion_tokens=7,
        ),
        SimpleNamespace(
            new_text="<tool_call>\n<function=pwd>\n",
            finished=False,
            finish_reason=None,
            prompt_tokens=12,
            completion_tokens=8,
        ),
        SimpleNamespace(
            new_text="</function>\n",
            finished=False,
            finish_reason=None,
            prompt_tokens=12,
            completion_tokens=14,
        ),
        SimpleNamespace(
            new_text="</tool_call>",
            finished=True,
            finish_reason="stop",
            prompt_tokens=12,
            completion_tokens=15,
        ),
    ]

    class FakeEngine:
        async def stream_chat(self, messages, **kwargs):
            del messages, kwargs
            for output in outputs:
                yield output

    monkeypatch.setattr(server, "_reasoning_parser", Qwen3ReasoningParser())
    monkeypatch.setattr(server, "_enable_auto_tool_choice", True)
    monkeypatch.setattr(server, "_tool_call_parser", "qwen")
    monkeypatch.setattr(server, "_tool_parser_instance", None)
    monkeypatch.setattr(server, "_engine", None)

    request = server.ChatCompletionRequest(
        model="Qwen/Qwen3.5-2B",
        messages=[server.Message(role="user", content="Run pwd")],
        stream=True,
    )

    async def collect():
        return [
            chunk
            async for chunk in server.stream_chat_completion(
                FakeEngine(),
                [{"role": "user", "content": "Run pwd"}],
                request,
            )
        ]

    chunks = asyncio.run(collect())
    payloads = [
        json.loads(chunk.removeprefix("data: ").strip())
        for chunk in chunks
        if "[DONE]" not in chunk
    ]

    reasoning = [
        payload["choices"][0]["delta"].get("reasoning")
        for payload in payloads
        if payload.get("choices")
        and payload["choices"][0]["delta"].get("reasoning")
    ]
    tool_chunks = [
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices")
        and payload["choices"][0]["delta"].get("tool_calls")
    ]

    assert reasoning == ["Use the directory tool."]
    assert len(tool_chunks) == 1
    assert tool_chunks[0]["finish_reason"] == "tool_calls"
    function = tool_chunks[0]["delta"]["tool_calls"][0]["function"]
    assert function == {"name": "pwd", "arguments": "{}"}
