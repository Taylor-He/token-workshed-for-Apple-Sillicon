# SPDX-License-Identifier: Apache-2.0
"""The SGLang CLI receives the parser settings for OpenAI streaming output."""

from token_workshed_nextcore.sglang_runtime import server_command


def test_qwen_streaming_tool_and_reasoning_parsers_are_routed_to_sglang() -> None:
    command = server_command(
        "/venv/bin/python",
        "Qwen/Qwen3.5-2B",
        "127.0.0.1",
        8000,
        tool_parser="qwen3",
        reasoning_parser="qwen3",
    )

    assert command[command.index("--tool-call-parser") + 1] == "qwen"
    assert command[command.index("--reasoning-parser") + 1] == "qwen3"
    assert "token_workshed_nextcore.sglang_runtime" in command[2]
