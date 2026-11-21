# Token Workshed NextCore 0.2.1

First NextCore release: a native local AI workbench and in-app Agent for Apple
Silicon Macs, with optional remote model serving through RoboServer.

## Downloads

- **Full installer** — [token-workshed NextCore.pkg](https://github.com/Taylor-He/token-workshed-nextcore/releases/download/v0.2.1/token-workshed%20NextCore.pkg)
- **Update installer** — [token-workshed NextCore Update.pkg](https://github.com/Taylor-He/token-workshed-nextcore/releases/download/v0.2.1/token-workshed%20NextCore%20Update.pkg)

## What’s included

- SGLang Runtime `v0.5.22`, pinned to commit
  `baa9befa381511cd4d56f83980951e5a45fffc6d`, with Python 3.12 on macOS 14+
  Apple Silicon.
- Native chat, streaming, completions, and the existing OpenAI-compatible
  inference routes.
- An app-owned Agent for web search and reading, files, terminal, browser,
  computer use, MCP connections, memories, and Skills. Agent tools are only
  available from authenticated NextCore in-app chat.
- Optional RoboServer connection to another Apple Silicon Mac. SSH host-key
  verification and a private SSH tunnel protect remote management access.
- SGLang-based teacher generation, benchmarks, and evaluation. Loss scoring
  uses a temporary SGLang Torch/MPS backend.
- Independent NextCore application identity and data directory so it can
  coexist with the earlier Token Workshed app.

## Requirements and first launch

- macOS 14 or later on Apple Silicon (`arm64`).
- Internet access during installation to install the pinned SGLang runtime;
  model weights are downloaded separately and are not bundled.
- The installer can use an existing Python 3.12 installation or install the
  Python 3.12.8 bootstrap from python.org.

## Distribution status

These packages are **not signed with a Developer ID and are not notarized**.
macOS Gatekeeper may block them from opening normally. The app bundle is
locally code-signature verified, but this does not replace Developer ID
signing or notarization.

## Runtime limits

- Text inference uses SGLang. There is no vLLM-MLX inference fallback.
- Image, video, ASR, and TTS are unavailable until verified for each model and
  runtime. SGLang-Omni is separate and experimental.
- The OpenAI-compatible API, ACP, MCP, and IDE integrations provide inference;
  they cannot invoke NextCore's in-app Agent tools.
- Legacy training and conversion work orders remain readable. Unsupported
  operations are rejected before execution.
