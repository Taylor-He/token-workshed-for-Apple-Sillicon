# SPDX-License-Identifier: Apache-2.0
"""Standalone optional Workshed data and model task runners.

This module is invoked with the explicitly consented Workshed toolchain
interpreter. It deliberately avoids importing the desktop app package.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# This worker is launched by file path. Python then puts ``token_workshed_nextcore`` at
# ``sys.path[0]``, where this package's ``platform.py`` shadows the standard
# library module imported by Transformers and PyTorch.
_WORKER_DIRECTORY = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0] or ".").resolve() == _WORKER_DIRECTORY:
    sys.path.pop(0)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_rows(path: Path, fmt: str) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    fmt = (
        fmt
        if fmt != "auto"
        else {".jsonl": "jsonl", ".json": "json", ".txt": "text", ".csv": "csv"}.get(
            suffix, "auto"
        )
    )
    if fmt in {"jsonl", "json"}:
        if fmt == "jsonl":
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
            rows = (
                value
                if isinstance(value, list)
                else value.get("data", []) if isinstance(value, dict) else []
            )
        return [row if isinstance(row, dict) else {"text": str(row)} for row in rows]
    if fmt == "text":
        return [
            {"text": line}
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if fmt == "csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if fmt in {"parquet", "arrow"}:
        from datasets import Dataset

        dataset = (
            Dataset.from_parquet(str(path))
            if fmt == "parquet"
            else Dataset.from_file(str(path))
        )
        return [dict(row) for row in dataset]
    raise ValueError(f"Cannot read local dataset file format: {path.name}")


def _source_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    params = config["params"]
    kind = str(params["source_kind"])
    ref = str(params["ref"])
    if kind == "huggingface":
        from datasets import load_dataset

        kwargs: dict[str, Any] = {"split": str(params.get("split") or "train")}
        if params.get("revision"):
            kwargs["revision"] = str(params["revision"])
        if params.get("subset"):
            kwargs["name"] = str(params["subset"])
        dataset = load_dataset(ref, **kwargs)
        return [dict(row) for row in dataset]

    path = Path(ref).expanduser().resolve()
    if path.is_dir():
        if (path / "manifest.json").is_file():
            rows: list[dict[str, Any]] = []
            for name in ("train.jsonl", "valid.jsonl", "test.jsonl"):
                file_path = path / name
                if file_path.is_file():
                    rows.extend(_read_rows(file_path, "jsonl"))
            if rows:
                return rows
        split_name = str(params.get("split") or "train")
        candidates = [
            path / f"{split_name}.jsonl",
            path / f"{split_name}.json",
            path / f"{split_name}.parquet",
        ]
        files = [item for item in candidates if item.is_file()]
        if not files:
            files = sorted(
                item
                for item in path.iterdir()
                if item.is_file()
                and item.suffix.lower()
                in {".jsonl", ".json", ".txt", ".csv", ".parquet", ".arrow"}
            )
        if not files:
            raise ValueError(f"No supported dataset files were found in {path}.")
        rows = []
        for item in files:
            rows.extend(_read_rows(item, str(params.get("format", "auto"))))
        return rows
    if not path.is_file():
        raise ValueError(f"Dataset source does not exist: {path}")
    return _read_rows(path, str(params.get("format", "auto")))


def _field(row: dict[str, Any], key: str, label: str) -> Any:
    if key not in row:
        raise ValueError(
            f"Dataset row is missing the configured {label} field '{key}'."
        )
    return row[key]


def _message_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    return "\n".join(
        f"{item.get('role', 'user')}: {item.get('content', '')}"
        for item in messages
        if isinstance(item, dict)
    )


def _render_messages(
    tokenizer: Any, messages: Any, *, add_generation_prompt: bool = False
) -> str:
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_template) and getattr(tokenizer, "chat_template", None):
        return str(
            apply_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        )
    return _message_text(messages)


def _normalize_row(
    row: dict[str, Any], params: dict[str, Any], logical_format: str
) -> dict[str, Any]:
    prompt_key = str(params.get("prompt_field", "prompt"))
    response_key = str(params.get("response_field", "response"))
    messages_key = str(params.get("messages_field", "messages"))
    text_key = str(params.get("text_field", "text"))
    chosen_key = str(params.get("chosen_field", "chosen"))
    rejected_key = str(params.get("rejected_field", "rejected"))
    if logical_format == "auto":
        if chosen_key in row and rejected_key in row:
            logical_format = "preference"
        elif messages_key in row:
            logical_format = "chat"
        elif text_key in row:
            logical_format = "text"
        else:
            logical_format = "completions"
    if logical_format == "chat":
        messages = _field(row, messages_key, "messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("Chat records must contain a non-empty messages array.")
        clean = [
            {"role": str(item.get("role", "user")), "content": item.get("content", "")}
            for item in messages
            if isinstance(item, dict)
        ]
        last = clean[-1]
        result: dict[str, Any] = {
            "messages": clean,
            "prompt": _message_text(clean[:-1]),
            "completion": str(last.get("content", "")),
        }
        return result
    if logical_format == "completions":
        prompt = _field(row, prompt_key, "prompt")
        response_name = response_key if response_key in row else "completion"
        completion = _field(row, response_name, "response")
        return {
            "prompt": str(prompt),
            "completion": str(completion),
            "messages": [
                {"role": "user", "content": str(prompt)},
                {"role": "assistant", "content": str(completion)},
            ],
        }
    if logical_format == "text":
        return {"text": str(_field(row, text_key, "text"))}
    if logical_format == "preference":
        return {
            "prompt": str(_field(row, prompt_key, "prompt")),
            "chosen": str(_field(row, chosen_key, "chosen")),
            "rejected": str(_field(row, rejected_key, "rejected")),
        }
    raise ValueError(f"Unsupported logical dataset format: {logical_format}")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_dataset(config: dict[str, Any]) -> dict[str, Any]:
    params = config["params"]
    rows = _source_rows(config)
    if not rows:
        raise ValueError("The selected dataset split contains no records.")
    logical_format = str(params.get("record_format", "auto"))
    normalized = [_normalize_row(row, params, logical_format) for row in rows]
    logical_format = str(params.get("record_format", "auto"))
    if logical_format == "auto":
        first = rows[0]
        logical_format = (
            "preference"
            if "chosen" in first and "rejected" in first
            else (
                "chat"
                if "messages" in first
                else "text" if "text" in first else "completions"
            )
        )
    indices = list(range(len(normalized)))
    seed = int(params.get("seed") if params.get("seed") is not None else 0)
    if bool(params.get("shuffle", True)):
        random.Random(seed).shuffle(indices)
    validation_fraction = float(params.get("validation_fraction", 0.05))
    validation_count = (
        min(len(indices) - 1, max(1, int(round(len(indices) * validation_fraction))))
        if len(indices) > 1 and validation_fraction > 0
        else 0
    )
    valid = [normalized[index] for index in indices[:validation_count]]
    train = [normalized[index] for index in indices[validation_count:]]
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "train.jsonl", train)
    if valid:
        _write_jsonl(output / "valid.jsonl", valid)
    manifest = {
        "schema_version": 1,
        "source": {
            "kind": params["source_kind"],
            "ref": params["ref"],
            "revision": params.get("revision", ""),
            "subset": params.get("subset", ""),
            "split": params.get("split", "train"),
        },
        "format": logical_format,
        "fields": {
            key: params.get(key)
            for key in (
                "prompt_field",
                "response_field",
                "messages_field",
                "text_field",
                "chosen_field",
                "rejected_field",
            )
        },
        "seed": seed,
        "shuffle": bool(params.get("shuffle", True)),
        "counts": {"source": len(rows), "train": len(train), "valid": len(valid)},
        "files": ["train.jsonl", *(["valid.jsonl"] if valid else [])],
    }
    write_json(output / "manifest.json", manifest)
    return {"path": str(output), "manifest": manifest}


def _sglang_request(server_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        server_url.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"SGLang returned HTTP {exc.code}: {detail}") from exc
    if not isinstance(result, dict):
        raise RuntimeError("SGLang returned an invalid response.")
    return result


def generate_teacher(config: dict[str, Any]) -> dict[str, Any]:
    model_path = str(config["inputs"]["teacher"])
    source_dir = Path(str(config["inputs"]["dataset"]))
    manifest = read_json(source_dir / "manifest.json", {})
    params = config["params"]
    server_url = str(config["server_url"])
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    split_results: dict[str, list[dict[str, Any]]] = {}
    errors: list[str] = []
    for split_name in ("train", "valid", "test"):
        split_path = source_dir / f"{split_name}.jsonl"
        if not split_path.is_file():
            continue
        result_rows = []
        for index, row in enumerate(_read_rows(split_path, "jsonl")):
            try:
                messages = row.get("messages")
                if isinstance(messages, list) and messages:
                    prompt_messages = list(messages[:-1] if messages[-1].get("role") == "assistant" else messages)
                    prompt = str(row.get("prompt") or _message_text(prompt_messages))
                else:
                    prompt = str(row.get("prompt", row.get("text", "")))
                    prompt_messages = [{"role": "user", "content": prompt}]
                request = {"model": model_path, "messages": prompt_messages,
                           "max_tokens": int(params["max_new_tokens"]),
                           "temperature": float(params["temperature"]),
                           "top_p": float(params["top_p"]), "stream": False}
                if params.get("seed") is not None:
                    request["seed"] = int(params["seed"])
                response = _sglang_request(server_url, "/v1/chat/completions", request)
                answer = str(response["choices"][0]["message"].get("content") or "").strip()
                if not answer:
                    raise ValueError("SGLang returned an empty teacher response.")
                enriched = dict(row)
                enriched[str(params.get("response_field", "teacher_response"))] = answer
                enriched["prompt"] = prompt
                enriched["completion"] = answer
                enriched["messages"] = [*prompt_messages, {"role": "assistant", "content": answer}]
                result_rows.append(enriched)
            except Exception as exc:
                if params.get("on_error", "fail") != "skip":
                    raise RuntimeError(f"Teacher generation failed in {split_name} row {index}: {exc}") from exc
                errors.append(f"{split_name} row {index}: {exc}")
        if result_rows:
            split_results[split_name] = result_rows
    if not split_results.get("train"):
        raise ValueError("Teacher generation produced no usable responses.")
    for split_name, result_rows in split_results.items():
        _write_jsonl(output / f"{split_name}.jsonl", result_rows)
    if manifest.get("format") == "preference":
        manifest["format"] = "chat"
    manifest["counts"] = {**(manifest.get("counts") if isinstance(manifest.get("counts"), dict) else {}),
                          **{name: len(rows) for name, rows in split_results.items()}}
    manifest["files"] = [f"{name}.jsonl" for name in split_results]
    manifest["teacher_generation"] = {"model": model_path, "engine": "sglang",
        "generated": sum(len(rows) for rows in split_results.values()), "skipped": len(errors),
        "max_new_tokens": int(params["max_new_tokens"])}
    write_json(output / "manifest.json", manifest)
    return {"path": str(output), "summary": manifest["teacher_generation"]}


def evaluate_loss(config: dict[str, Any]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    model_ref = str(config["inputs"]["model"])
    dataset_path = Path(str(config["inputs"]["dataset"]))
    params = config["params"]
    split = str(params.get("split", "validation"))
    candidate = dataset_path / ("valid.jsonl" if split in {"validation", "valid"} else "test.jsonl" if split == "test" else "train.jsonl")
    if not candidate.is_file():
        raise ValueError(f"Dataset artifact has no '{split}' split required for evaluation.")
    tokenizer = AutoTokenizer.from_pretrained(model_ref)
    max_examples = int(params.get("max_batches", 25)) * int(params.get("batch_size", 1))
    max_length = int(params.get("max_sequence_length", 2048))
    losses: list[float] = []
    tokens = 0
    for row in _read_rows(candidate, "jsonl"):
        text = str(row.get("text", "")) or str(row.get("prompt", "")) + str(row.get("completion", ""))
        if isinstance(row.get("messages"), list):
            text = _render_messages(tokenizer, row["messages"])
        if not text.strip():
            continue
        ids = tokenizer(text, truncation=True, max_length=max_length)["input_ids"]
        if len(ids) < 2:
            continue
        response = _sglang_request(str(config["server_url"]), "/generate", {
            "input_ids": ids, "return_logprob": True, "logprob_start_len": 0,
            "sampling_params": {"max_new_tokens": 0, "temperature": 0},
        })
        entries = response.get("meta_info", {}).get("input_token_logprobs")
        if not isinstance(entries, list):
            raise RuntimeError("SGLang Torch/MPS did not return prompt logprobs for this model.")
        values = [float(entry[0]) for entry in entries if isinstance(entry, (list, tuple)) and entry and entry[0] is not None]
        if len(values) != len(ids) - 1 or not all(math.isfinite(x) for x in values):
            raise RuntimeError("SGLang returned incomplete or non-finite prompt logprobs; loss is unavailable for this model.")
        losses.append(-sum(values) / len(values))
        tokens += len(values)
        if len(losses) >= max_examples:
            break
    if not losses:
        raise ValueError("The selected dataset split has no evaluable text records.")
    mean_loss = sum(losses) / len(losses)
    metrics = {"loss": mean_loss, "perplexity": math.exp(min(mean_loss, 80)),
               "examples": len(losses), "tokens": tokens, "split": split,
               "batch_size": int(params.get("batch_size", 1)), "device": "mps",
               "engine": "sglang", "notes": []}
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metrics.json", metrics)
    return {"path": str(output_dir / "metrics.json"), "metrics": metrics}


def evaluate_lm(config: dict[str, Any]) -> dict[str, Any]:
    from lm_eval import simple_evaluate

    params = config["params"]
    started = time.monotonic()
    model_ref = str(config["inputs"]["model"])
    results = simple_evaluate(
        model="local-completions",
        model_args={"model": model_ref, "base_url": str(config["server_url"]).rstrip("/") + "/v1/completions",
                    "tokenizer": model_ref, "tokenized_requests": False},
        tasks=list(params["tasks"]), num_fewshot=int(params["fewshot"]),
        limit=params.get("limit"), batch_size=params.get("batch_size", "auto"),
        random_seed=params.get("seed"),
    )
    report = {"suite": "lm-eval", "tasks": list(params["tasks"]),
              "fewshot": params["fewshot"], "limit": params.get("limit"),
              "results": results.get("results", {}), "versions": results.get("versions", {}),
              "device": "SGLang MLX backend", "engine": "sglang", "notes": [],
              "elapsed_seconds": round(time.monotonic() - started, 3)}
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "report.json", report)
    return {"path": str(output_dir / "report.json"), "report": report}


def run_action(action: str, config_path: Path) -> dict[str, Any]:
    if action in {"train-sft", "train-lora", "train-qlora", "train-dora", "train-full", "train-dpo", "train-logits-distill", "fuse-adapter", "quantize-mlx"}:
        raise ValueError(
            f"Workshed action '{action}' was removed from SGLang-only NextCore."
        )
    config = read_json(config_path, {})
    if not isinstance(config, dict):
        raise ValueError("Workshed execution config is not a JSON object.")
    if action == "prepare-dataset":
        return prepare_dataset(config)
    if action == "generate-teacher":
        return generate_teacher(config)
    if action == "evaluate-loss":
        return evaluate_loss(config)
    if action == "evaluate-lm":
        return evaluate_lm(config)
    raise ValueError(f"Unknown Workshed execution action: {action}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=[
            "prepare-dataset",
            "generate-teacher",
            "evaluate-loss",
            "evaluate-lm",
        ],
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    try:
        result = run_action(args.action, args.config)
        write_json(args.result, result)
        return 0
    except Exception as exc:
        print(f"Workshed {args.action} failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
