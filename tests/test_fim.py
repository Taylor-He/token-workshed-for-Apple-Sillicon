from __future__ import annotations

from vllm_mlx.fim import (
    build_fim_prompt,
    detect_fim_tokens,
    record_completion_diagnostic,
)


class NativeFIMTokenizer:
    special_tokens_map = {
        "additional_special_tokens": [
            "<|fim_prefix|>",
            "<|fim_suffix|>",
            "<|fim_middle|>",
        ]
    }


def test_native_fim_prompt_uses_complete_token_triplet():
    tokens = detect_fim_tokens(NativeFIMTokenizer())
    assert tokens == {
        "prefix": "<|fim_prefix|>",
        "suffix": "<|fim_suffix|>",
        "middle": "<|fim_middle|>",
    }
    rendered = build_fim_prompt("before", "after", NativeFIMTokenizer())
    assert rendered.native is True
    assert rendered.prompt == "<|fim_prefix|>before<|fim_suffix|>after<|fim_middle|>"


def test_missing_fim_tokens_forces_generic_fallback_and_records_diagnostic():
    rendered = build_fim_prompt("before", "after")
    assert rendered.native is False
    assert "PREFIX:\nbefore" in rendered.prompt
    assert "SUFFIX:\nafter" in rendered.prompt
    record = record_completion_diagnostic(
        "test-model", suffix_requested=True, fim_prompt=rendered
    )
    assert record["mode"] == "experimental_generic_fallback"
    assert record["native_fim"] is False
