# token-workshed NextCore support status

token-workshed NextCore keeps the Token Workshed desktop and OpenAI-compatible app API while
replacing the vLLM-MLX inference server with SGLang Runtime.

## Fixed runtime

- SGLang Runtime `v0.5.22`, commit
  `baa9befa381511cd4d56f83980951e5a45fffc6d`.
- Python `3.12`; macOS `14+`; Apple Silicon (`arm64`).
- Text generation and chat use the SGLang MLX backend.
- Workshed loss scoring uses SGLang Torch/MPS, with chat services restored
  after the scoring job exits.
- SGLang-Omni is pinned separately to `v0.2.0`, commit
  `e8c635ecb9c68bcf2fbc32ceb3c40f85f3590657`; it is experimental and does not
  provide a fallback for the main runtime.

## Available routes

The desktop app retains `/v1/models`, `/v1/chat/completions`, and
`/v1/completions`. Text generation, streaming, and simple completions have been
exercised against the pinned local SGLang runtime. Model-specific tool calling
requires a model whose output format is supported by the configured SGLang
parser; the OpenAI-compatible API does not execute local Agent tools.

## App-owned Agent

The native NextCore chat surface owns the SGLang tool loop. It can search and
read public webpages, run terminal commands, access files allowed by macOS,
operate macOS applications, and control an isolated Chromium session. The
computer and browser tools use the pinned CUA Driver `v0.28.1`; NextCore manages
its download and starts it in a background warmup when the desktop app opens.
macOS permissions are checked only when a computer or browser tool is used.
The Driver is not bundled into the source build.

The Agent is enabled only for authenticated requests from the native app's
chat window. `/api/chat`, `/v1/chat/completions`, `/v1/completions`, ACP, MCP,
and IDE clients do not receive the built-in computer, browser, terminal, or
file tools. ACP/IDE Agent session routes return 403; the external MCP bridge
lists no tools and rejects calls. The JetBrains Provider remains inference-only.
If the active model has no supported SGLang tool parser, NextCore continues
ordinary chat and reports that tools are unavailable; it does not send a
substitute JSON tool protocol or switch inference engines.

SGLang reasoning fields are kept only in the active in-memory model exchange
when a recovery or tool continuation needs them. Reasoning is never streamed to
the UI or written to the audit log, and recognized `<think>` sections are
filtered before display. Agent activity is shown as tool status events. Audit
records retain tool names, outcomes, durations, and argument hashes rather
than command text, paths, or file contents. macOS privacy permissions remain
in force; NextCore does not bypass them or elevate itself.

The Agent loop uses bounded Hermes-style recovery for empty, reasoning-only,
and post-tool replies. SGLang's hidden `reasoning_content` is replayed only in
the in-memory model conversation; it is never streamed to the UI or audit log.
NextCore does not set a fixed SGLang context length or shorten chat history
when SGLang reports insufficient output headroom; it retries with a smaller
output-token budget. Persisted chat history retains up to 16,384 messages and
512,000 characters per conversation, enough for a 64K-token text context in
typical use. SGLang and the loaded model configuration remain authoritative.

Only the native NextCore chat UI receives the per-launch token required to
enable Agent tools. See [installation](installation.md) and
[Workshed runtime behavior](workshed.md). For a local server or remote Mac
connection, see [RoboServer and remote mode](roboserver.md).

## Capabilities pending model-level validation

Image, video, ASR, and TTS are reported unavailable until verified for a
specific model on the target macOS runtime. The optional Omni installer has
additional prerequisites and is not enabled by default. SFT, LoRA, QLoRA,
DoRA, DPO, distillation, adapter fusion, and model conversion are not runnable
in NextCore. Legacy work orders remain readable and fail preflight with a
removed-operation message.

The inherited `docs/guides`, `docs/reference`, and `docs/benchmarks` pages
describe the 0.2.1 vLLM-MLX runtime. They are preserved as historical material;
use the NextCore pages for current instructions.
