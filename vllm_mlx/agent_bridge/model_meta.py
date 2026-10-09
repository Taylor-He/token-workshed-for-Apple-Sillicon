"""Model metadata loading for adapter selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _extract_context_length(payloads: list[dict[str, Any]]) -> int | None:
    keys = (
        "max_position_embeddings",
        "max_sequence_length",
        "max_seq_len",
        "seq_length",
        "n_positions",
        "context_length",
        "context_window",
        "max_model_len",
        "model_max_length",
        "max_tokens",
    )
    candidate_payloads: list[dict[str, Any]] = []
    for payload in payloads:
        candidate_payloads.append(payload)
        for nested_key in ("text_config", "language_config", "llm_config"):
            nested = payload.get(nested_key)
            if isinstance(nested, dict):
                candidate_payloads.append(nested)

    # Tokenizer configs commonly use enormous integers as an "unbounded"
    # sentinel. They are not usable model context windows.
    max_reasonable_context_length = 16_777_216
    for payload in candidate_payloads:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                parsed = int(value)
                if 0 < parsed <= max_reasonable_context_length:
                    return parsed
            if isinstance(value, str) and value.strip().isdigit():
                parsed = int(value.strip())
                if 0 < parsed <= max_reasonable_context_length:
                    return parsed
        rope_scaling = payload.get("rope_scaling")
        if isinstance(rope_scaling, dict):
            original = rope_scaling.get("original_max_position_embeddings")
            if isinstance(original, (int, float)):
                parsed = int(original)
                if 0 < parsed <= max_reasonable_context_length:
                    return parsed
    return None


def _extract_special_tokens(tokenizer_cfg: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "bos_token",
        "eos_token",
        "pad_token",
        "unk_token",
        "sep_token",
        "cls_token",
        "mask_token",
        "additional_special_tokens",
    )
    output: dict[str, Any] = {}
    for field in fields:
        value = tokenizer_cfg.get(field)
        if value is None:
            continue
        output[field] = value
    return output


def _resolve_model_dir(model: str, allow_download: bool) -> tuple[Path | None, list[str]]:
    warnings: list[str] = []
    as_path = Path(model).expanduser()
    if as_path.exists() and as_path.is_dir():
        return as_path, warnings

    try:
        from huggingface_hub import snapshot_download

        model_dir = Path(
            snapshot_download(
                repo_id=model,
                local_files_only=not allow_download,
                allow_patterns=[
                    "config.json",
                    "tokenizer_config.json",
                    "generation_config.json",
                ],
            )
        )
        return model_dir, warnings
    except Exception as exc:
        warnings.append(f"Unable to resolve Hugging Face snapshot for '{model}': {exc}")
        return None, warnings


def read_model_metadata(model: str, allow_download: bool = False) -> dict[str, Any]:
    model_dir, warnings = _resolve_model_dir(model=model, allow_download=allow_download)
    if model_dir is None:
        return {
            "model_ref": model,
            "model_dir": None,
            "config": {},
            "tokenizer_config": {},
            "generation_config": {},
            "architecture": [],
            "model_type": "",
            "chat_template": None,
            "max_context_length": None,
            "special_tokens": {},
            "warnings": warnings,
        }

    config = _read_json(model_dir / "config.json")
    tokenizer_cfg = _read_json(model_dir / "tokenizer_config.json")
    generation_cfg = _read_json(model_dir / "generation_config.json")

    architecture = config.get("architectures")
    if not isinstance(architecture, list):
        architecture = []
    architecture = [str(item) for item in architecture if isinstance(item, str)]

    model_type = config.get("model_type")
    if not isinstance(model_type, str):
        model_type = ""

    chat_template = tokenizer_cfg.get("chat_template")
    if chat_template is not None and not isinstance(chat_template, str):
        chat_template = None

    max_context_length = _extract_context_length([config, tokenizer_cfg, generation_cfg])
    special_tokens = _extract_special_tokens(tokenizer_cfg)

    return {
        "model_ref": model,
        "model_dir": str(model_dir),
        "config": config,
        "tokenizer_config": tokenizer_cfg,
        "generation_config": generation_cfg,
        "architecture": architecture,
        "model_type": model_type,
        "chat_template": chat_template,
        "max_context_length": max_context_length,
        "special_tokens": special_tokens,
        "warnings": warnings,
    }
