# SPDX-License-Identifier: Apache-2.0
"""Text generation benchmark client for a running SGLang OpenAI API."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests
from tabulate import tabulate


DEFAULT_PROMPTS = (
    "Hello, how are you?",
    "What is 2+2?",
    "Say hello in Spanish.",
    "What is the capital of France? Give one historical fact.",
    "Explain why leaves appear green in one sentence.",
)


@dataclass
class BenchmarkResult:
    prompt: str
    prompt_tokens: int
    generated_tokens: int
    ttft: float
    total_time: float

    @property
    def tpot(self) -> float:
        return max(0.0, self.total_time - self.ttft) / max(1, self.generated_tokens - 1)

    @property
    def generation_tps(self) -> float:
        return self.generated_tokens / max(self.total_time - self.ttft, 1e-9)


@dataclass
class BenchmarkSummary:
    model: str
    base_url: str
    results: list[BenchmarkResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime": "SGLang",
            "model": self.model,
            "base_url": self.base_url,
            "runs": [
                {
                    **asdict(result),
                    "tpot": result.tpot,
                    "generation_tps": result.generation_tps,
                }
                for result in self.results
            ],
            "mean_ttft_seconds": statistics.mean(row.ttft for row in self.results)
            if self.results
            else 0.0,
            "mean_generation_tokens_per_second": statistics.mean(
                row.generation_tps for row in self.results
            )
            if self.results
            else 0.0,
        }


def benchmark_single_prompt(
    base_url: str,
    model: str,
    prompt: str,
    *,
    max_tokens: int = 256,
    temperature: float = 0.7,
    timeout: float = 180.0,
) -> BenchmarkResult:
    """Measure streamed text generation through SGLang's OpenAI endpoint."""

    started = time.perf_counter()
    first_token_at: float | None = None
    generated_text: list[str] = []
    usage: dict[str, Any] = {}
    response = requests.post(
        f"{base_url.rstrip('/')}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        stream=True,
        timeout=(10, timeout),
    )
    response.raise_for_status()
    with response:
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            choices = event.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            text = str(choices[0].get("text", "") or "")
            if text:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                generated_text.append(text)

    finished = time.perf_counter()
    full_text = "".join(generated_text)
    ttft = max(0.0, (first_token_at or finished) - started)
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    generated_tokens = int(usage.get("completion_tokens", 0) or 0)
    if generated_tokens <= 0 and full_text:
        generated_tokens = max(1, len(full_text.split()))
    return BenchmarkResult(
        prompt=prompt,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        ttft=ttft,
        total_time=max(0.0, finished - started),
    )


def run_benchmark(
    model_name: str,
    *,
    base_url: str = "http://127.0.0.1:8010",
    num_prompts: int = 5,
    max_tokens: int = 256,
    temperature: float = 0.7,
    warmup_runs: int = 1,
) -> BenchmarkSummary:
    prompts = list(DEFAULT_PROMPTS[: max(1, min(int(num_prompts), len(DEFAULT_PROMPTS)))])
    for prompt in prompts[: max(0, int(warmup_runs))]:
        benchmark_single_prompt(
            base_url, model_name, prompt, max_tokens=max_tokens, temperature=temperature
        )
    results = [
        benchmark_single_prompt(
            base_url, model_name, prompt, max_tokens=max_tokens, temperature=temperature
        )
        for prompt in prompts
    ]
    return BenchmarkSummary(model=model_name, base_url=base_url, results=results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark streamed text generation from a running SGLang server."
    )
    parser.add_argument("--model", required=True, help="Model name served by SGLang")
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--prompts", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--mllm",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--video", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.mllm or args.video:
        parser.error("Multimodal benchmarking is not supported by this NextCore build.")

    summary = run_benchmark(
        args.model,
        base_url=args.base_url,
        num_prompts=args.prompts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        warmup_runs=args.warmup,
    )
    payload = summary.to_dict()
    table = [
        [
            index + 1,
            row.prompt[:46],
            row.prompt_tokens,
            row.generated_tokens,
            f"{row.ttft:.3f}s",
            f"{row.generation_tps:.2f}",
        ]
        for index, row in enumerate(summary.results)
    ]
    print(tabulate(table, headers=["Run", "Prompt", "Input tokens", "Output tokens", "TTFT", "Output tok/s"]))
    print(f"\nSGLang mean TTFT: {payload['mean_ttft_seconds']:.3f}s")
    print(
        "SGLang mean output throughput: "
        f"{payload['mean_generation_tokens_per_second']:.2f} tokens/s"
    )
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
