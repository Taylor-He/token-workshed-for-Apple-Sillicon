# token-workshed NextCore

token-workshed NextCore is an independent development line of Token Workshed
0.2.1. It uses a pinned SGLang Runtime for every model execution path while
keeping the OpenAI-compatible app API, native desktop manager, and App-owned
chat Agent and tool safety controls.

## Runtime

- SGLang Runtime `v0.5.22`, source commit
  `baa9befa381511cd4d56f83980951e5a45fffc6d`.
- Python `3.12` and macOS `14+` on Apple Silicon.
- Text generation uses SGLang's MLX backend. Workshed loss scoring uses a
  temporary SGLang Torch/MPS server.
- The NextCore chat surface owns the SGLang Agent loop and its tools. OpenAI,
  ACP, MCP, and IDE surfaces remain inference-only and cannot call those tools.

## Install and run from source

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e '.[ide]'
scripts/install_sglang_runtime.sh
token-workshed-nextcore-desktop
```

The desktop app listens on `127.0.0.1:8010`; its local UI/API listens on
`127.0.0.1:7872`. To start a standalone text server:

```bash
token-workshed-nextcore serve MODEL_ID --port 8010
```

## Compatibility

The app keeps its existing OpenAI-compatible routes, including `/v1/models`,
`/v1/chat/completions`, and `/v1/completions`. The JetBrains completion adapter
translates a completion `suffix` into tokenizer-native FIM when the cached
tokenizer exposes a complete FIM token set; otherwise it records that the
generic completion format is experimental.

Workshed teacher generation, language-model evaluation, benchmarks, and loss
scoring use SGLang. Loss evaluation uses the Torch/MPS backend, then restores
the previous chat service. Training, adapter fusion, and model conversion
blocks are removed from the runnable Workbench; saved work orders containing
those blocks remain readable and are rejected during preflight.

Image, video, ASR, and TTS capabilities stay unavailable until validated for
the specific model and macOS runtime. No model falls back to another runtime.
SGLang-Omni `v0.2.0` remains a separately pinned experimental runtime and is
not enabled as a fallback.

## Co-installation

NextCore uses a separate bundle ID (`com.taylorhe.tokenworkshed.nextcore`),
application data directory (`token-workshed-nextcore`), and ports, so it can
be installed beside Token Workshed 0.2.1. Package builds are produced with
`scripts/build_macos_app_bundle.sh` and
`scripts/build_pkg_installer.sh`.

## Documentation

- [NextCore runtime and support status](docs/nextcore/README.md)
- [NextCore installation](docs/nextcore/installation.md)
- [Workshed with SGLang](docs/nextcore/workshed.md)
- [0.2.1 README archive](docs/archive/README_0.2.1.md)

Most inherited `docs/guides`, `docs/reference`, and benchmark pages describe
the 0.2.1 vLLM-MLX runtime. They are retained as historical material and are
not NextCore operating instructions.

The examples directory also preserves 0.2.1 material. See
[examples/README.md](examples/README.md) for the maintained SGLang example and
the status of the inherited scripts.

The macOS app displays as `token-workshed NextCore`. Its stable bundle path is
`nextcore.app`, and the installer is named `token-workshed NextCore.pkg`.
