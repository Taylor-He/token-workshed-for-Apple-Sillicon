# Token Workshed 0.2.1 · Local development build

This build extends the v0.2.0 Workshed flow while preserving the work-order API
v2 envelope and the `vllm-mlx` runtime version `0.4.0`.

## Workshed changes

- Dataset preparation now materializes local and Hugging Face sources into
  managed train/validation JSONL artifacts. It supports chat, prompt/completion,
  plain text, and preference records with configurable field mappings and a
  deterministic validation split.
- MLX SFT, LoRA, QLoRA, and DoRA consume the normalized dataset artifact.
  Teacher response generation keeps the dataset splits, and response
  distillation trains from those responses.
- DPO and compatible-logit distillation now have local Transformers runners.
  They verify student/teacher vocabulary and model compatibility where needed,
  report the selected MPS or CPU device, and save the resulting model locally.
- Loss evaluation and `lm-eval` write structured metric/report artifacts.
  Benchmark requests use the app's local OpenAI-compatible vllm-mlx server and
  report request latency and throughput data.
- Quality gates consume named metrics. A failed gate finalizes a failed run,
  preserves prior artifacts, and marks remaining blocks skipped. A missing
  metric configured with `missing=skip` is recorded as skipped and execution
  continues.
- The existing run-log dialog now includes compact per-step results.

## Compatibility and packaging

- Work-order schema remains v2. Block definitions and runners advance to
  version 3 so frozen plans detect the new execution contract.
- The app version is `0.2.1`; the Python runtime remains `0.4.0`.
- App and installer bundles are local build artifacts. No GitHub Release, tag,
  or remote push was created for this development build.
- The release `.app` and `token-workshed-0.2.1-installer.pkg` are local build
  artifacts. The app bundle is ad hoc signed; the installer package has no
  signature and is not notarized. The package payload contains the Workshed
  runner and excludes `docs/promotion/`. The package was not installed on this
  Mac.

## Validation scope

- Targeted Python Workshed, API, managed-model, dataset-normalization,
  quality-gate, execution, and terminal tests: 64 passed.
- Native UI Rust tests: 53 passed; `cargo fmt --check` passed.
- Ruff checks and Python byte-compilation passed for the changed Workshed
  modules and tests.
- Apple Silicon M3 runtime acceptance used MLX on Metal. A two-step LoRA run
  completed adapter fusion, model registration, and generation. A Workshed
  text-data preparation → all-layer SFT (`num_layers=-1`) → fuse → register
  run completed one training step; MLX-LM reported 85.054M of 124.440M
  parameters trainable, train loss 5.677, and validation loss 5.698. This is a
  pipeline smoke result, not a convergence or model-quality claim.
- Teacher generation produced five rows, followed by a one-step response
  distillation and registration. One-step DPO and logits distillation each
  completed on MPS with finite saved weights. Loss evaluation, a one-example
  HellaSwag sample, JSON report export, and a named-metric quality gate passed.
- The local OpenAI-compatible server loaded a registered Workshed model and
  returned a chat completion. This validates the local execution path only;
  it is not a broad model-quality benchmark.
- A chat-formatted SFT attempt with GPT-2 was rejected by MLX-LM because that
  tokenizer has no chat template. The successful SFT smoke run used the
  supported plain-text dataset format.
