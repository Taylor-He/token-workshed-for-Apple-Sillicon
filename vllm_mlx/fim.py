# SPDX-License-Identifier: Apache-2.0
"""Fill-in-the-middle helpers for the JetBrains completion bridge.

The desktop App intentionally keeps this layer small. A model may expose FIM
tokens through a Hugging Face tokenizer, or it may not. In the latter case we
still make the request (the product requirement is to force the attempt), but
record an explicit compatibility diagnostic instead of pretending that the
model has native FIM support.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

_TOKEN_CANDIDATES = {
    "prefix": ("<|fim_prefix|>", "<fim_prefix>"),
    "suffix": ("<|fim_suffix|>", "<fim_suffix>"),
    "middle": ("<|fim_middle|>", "<fim_middle>"),
}
_DIAGNOSTIC_LIMIT = 128
_DIAGNOSTICS: OrderedDict[str, dict[str, Any]] = OrderedDict()
_LOCK = threading.RLock()


@dataclass(frozen=True)
class FIMPrompt:
    """Rendered completion prompt and its compatibility mode."""

    prompt: str
    native: bool
    reason: str
    tokens: dict[str, str | None]


def _actual_tokenizer(tokenizer: Any) -> Any:
    current = tokenizer
    for _ in range(2):
        wrapped = getattr(current, "tokenizer", None)
        if wrapped is None or wrapped is current:
            break
        current = wrapped
    return current


def _tokenizer_names(tokenizer: Any) -> set[str]:
    names: set[str] = set()
    if tokenizer is None:
        return names
    for attr in ("special_tokens_map", "added_tokens_encoder"):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, list):
                    names.update(str(part) for part in item)
                else:
                    names.add(str(item))
                names.add(str(key))
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        with suppress(Exception):
            names.update(str(item) for item in get_vocab())
    return names


def detect_fim_tokens(tokenizer: Any) -> dict[str, str | None]:
    """Find a complete FIM token triplet without loading model weights."""
    actual = _actual_tokenizer(tokenizer)
    names = _tokenizer_names(actual)
    found: dict[str, str | None] = {key: None for key in _TOKEN_CANDIDATES}
    for key, candidates in _TOKEN_CANDIDATES.items():
        for attr in (f"fim_{key}_token", f"{key}_token"):
            value = getattr(actual, attr, None)
            if isinstance(value, str) and value.strip():
                found[key] = value.strip()
                break
        if found[key] is None:
            for candidate in candidates:
                if candidate in names:
                    found[key] = candidate
                    break
    return found


def build_fim_prompt(prefix: str, suffix: str, tokenizer: Any = None) -> FIMPrompt:
    """Build a native FIM prompt, or a clearly marked generic fallback."""
    prefix = str(prefix or "")
    suffix = str(suffix or "")
    tokens = detect_fim_tokens(tokenizer)
    if all(tokens.values()):
        rendered = (
            f"{tokens['prefix']}{prefix}{tokens['suffix']}{suffix}"
            f"{tokens['middle']}"
        )
        return FIMPrompt(rendered, True, "native_fim_tokens", tokens)
    rendered = (
        "Complete the code between PREFIX and SUFFIX. Return only inserted code.\n"
        "PREFIX:\n"
        f"{prefix}\n"
        "SUFFIX:\n"
        f"{suffix}\n"
        "INSERTED:\n"
    )
    return FIMPrompt(rendered, False, "generic_completion_fallback", tokens)


def record_completion_diagnostic(
    model: str,
    *,
    suffix_requested: bool,
    fim_prompt: FIMPrompt,
) -> dict[str, Any]:
    """Store a bounded, non-secret completion compatibility record."""
    record = {
        "model": str(model or "").strip(),
        "suffix_requested": bool(suffix_requested),
        "mode": "native_fim" if fim_prompt.native else "experimental_generic_fallback",
        "native_fim": fim_prompt.native,
        "reason": fim_prompt.reason,
        "tokens": dict(fim_prompt.tokens),
        "recorded_at": int(time.time()),
    }
    key = record["model"] or "unknown"
    with _LOCK:
        _DIAGNOSTICS[key] = record
        _DIAGNOSTICS.move_to_end(key)
        while len(_DIAGNOSTICS) > _DIAGNOSTIC_LIMIT:
            _DIAGNOSTICS.popitem(last=False)
    return dict(record)


def completion_diagnostics() -> list[dict[str, Any]]:
    """Return the latest bounded diagnostics for App UI/contract tests."""
    with _LOCK:
        return [dict(item) for item in reversed(_DIAGNOSTICS.values())]
