"""Model family detection for parser/adapter routing."""

from __future__ import annotations

import re
from typing import Any


def detect_model_family(model_id: str, metadata: dict[str, Any]) -> str:
    parts = [str(model_id or "")]
    model_type = metadata.get("model_type")
    if isinstance(model_type, str):
        parts.append(model_type)
    architecture = metadata.get("architecture")
    if isinstance(architecture, list):
        parts.extend(str(item) for item in architecture if isinstance(item, str))

    text = " ".join(parts).strip().lower()
    if not text:
        return "unknown"

    if "hermes" in text:
        return "hermes"
    if "qwen3-coder" in text or ("qwen3" in text and "coder" in text):
        return "qwen3_coder"
    if "qwen3" in text or "qwq" in text:
        return "qwen3"
    if "qwen" in text:
        return "qwen"
    if re.search(r"llama[\s\-]?3\.1", text):
        return "llama31"
    if re.search(r"llama[\s\-]?3\.2", text):
        return "llama32"
    if re.search(r"llama[\s\-]?4", text):
        return "llama4"
    if "mistral" in text:
        return "mistral"
    if "granite" in text:
        return "granite"
    if "deepseek" in text:
        return "deepseek"
    if "glm" in text:
        return "glm"
    if "gpt-oss" in text or "harmony" in text:
        return "gpt_oss"
    return "unknown"
