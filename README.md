# Token Workshed

Token Workshed is a custom desktop-focused fork of `vllm-mlx` for Apple Silicon Macs.
It combines:

- a local `vllm-mlx` runtime
- a native Rust/Iced/libcosmic desktop UI
- a desktop model manager (model switching, community search/deploy, runtime options)
- macOS `.app` / `.pkg` packaging workflow

Current desktop release line in this repo:

- `token-workshed`: `0.2.0`
- `vllm-mlx`: `0.4.0`
- `OpenClaw`: `2026.7.1`
- `Hermes Agent`: `0.18.2` (release tag `v2026.7.7.2`)

## Highlights

- Native Apple Silicon local inference (text + multimodal support from `vllm-mlx`)
- Desktop manager mode (`token-workshed-desktop`) with local model lifecycle control
- Community page with Hugging Face model search and one-click download/deploy
- One-time Hugging Face account binding popup (saved locally for authenticated search/download)
- Startup model integrity check and cleanup for incomplete local caches
- If no model is installed, UI still opens and shows:
  - `To start using Token Workshed, please install a model.`
- **Workshed** is a local, block-based model-development workspace. Configure
  model and data inputs, quantization settings, and delivery actions in the
  native UI; preflight shows the resolved plan before a run starts.
- Quantized Workshed models can be registered in the managed model library and
  launched from Model Settings.
- Optional `Enable openclaw/hermes` toggle in chat composer (left-bottom of input box)
- Server Settings supports concurrent extra models and chat dispatch mode:
  - `Single (active model)` for normal switch-and-chat
  - `Parallel (all running models)` for one-prompt multi-model fan-out

## JetBrains Plugin

Token Workshed `0.2.0` includes a separate thin JetBrains plugin repository at
`../token-workshed-jetbrains`. It adds no Tool Window, chat page, settings
page, or status-bar UI. Its only job is to wake the local App, guide native
AI Assistant setup, and register the App-owned ACP Agent entry.

- Native AI Assistant Chat and the experimental completion path use the
  restricted OpenAI-compatible Provider exposed by the App. The App Settings
  → JetBrains Connections card provides the loopback Base URL, active model,
  Provider key, and MCP command to copy into JetBrains' native settings.
- Code completion sends `suffix` to `/v1/completions`; Token Workshed attempts
  tokenizer-native FIM and records an explicit generic-fallback diagnostic
  when the model has no FIM template. It is intentionally labeled
  **experimental reuse of the chat model** until a real IDE insertion is
  verified.
- The plugin pairs only after the user approves it in the App. It stores only
  an opaque scoped bridge credential in JetBrains PasswordSafe. App manager,
  model-server, Hermes, and tool credentials never cross into the plugin.
- The plugin merges one `Token Workshed` entry into `~/.jetbrains/acp.json`,
  preserving every other Agent entry. `token-workshed-acp` and
  `token-workshed-mcp` are App-owned stdio processes; the App keeps Agent
  sessions, MCP configuration, approvals, and audit records.
- Agent file changes are returned as patch proposals with `apply: false` and
  are never written to the IDE workspace. Read-only policy is automatic;
  terminal, network, external writes, and side-effecting MCP calls wait for an
  App decision. The v2 bridge and security boundary are documented in
  [`docs/IDE_BRIDGE_V2.md`](docs/IDE_BRIDGE_V2.md).

## OpenClaw Integration

- Bundled OpenClaw source version: `2026.7.1` (tag `v2026.7.1`)
- Bundled Hermes Agent source version: `0.18.2` (tag `v2026.7.7.2`)
- The exact pins and runtime requirements are tracked in
  [`integrations/COMPONENT_VERSIONS.md`](integrations/COMPONENT_VERSIONS.md).
- Bundled source copy path:
  - `integrations/openclaw/`
- The macOS installer includes the arm64 Node `24.15.0` runtime at
  `integrations/node-runtime/`. The large executable is not committed to Git;
  source checkouts that use OpenClaw must provide it at that path or set
  `TOKEN_WORKSHED_OPENCLAW_NODE_BIN` to a compatible runtime.
- Hermes source copy path:
  - `integrations/hermes/`
- UI toggle location:
  - Chat page composer, left-bottom, aligned with right-side action icons.
- Chat composer local controls:
  - Local model quick selector + `Enable openclaw/hermes` toggle in one row.
  - Runtime selection is automatic: OpenClaw first, Hermes fallback.
- Effect:
  - When enabled, chat requests include `openclaw_enabled=true` and apply OpenClaw/Hermes tool-use prompt guidance.

### Hermes Runtime Note

- Hermes Agent `0.18.2` requires Python `>=3.11,<3.14`. The desktop bridge
  creates and maintains an isolated venv under
  `integrations/hermes/.token-workshed-state/venv` and installs the local
  `pyproject.toml` package automatically. For a manual source setup:

```bash
python3 -m venv integrations/hermes/.token-workshed-state/venv
integrations/hermes/.token-workshed-state/venv/bin/python -m pip install -U -e integrations/hermes
```

## Requirements

- macOS (Apple Silicon / arm64 recommended)
- Python `>= 3.10`
- Hermes integration: Python `>= 3.11,<3.14`
- OpenClaw integration: Node `24.15+` (recommended), `22.22.3+`, or `25.9+`
- Network access (for Hugging Face search/download)

## Install From Source

```bash
git clone https://github.com/Taylor-He/token-workshed-for-Apple-Sillicon.git
cd token-workshed-for-Apple-Sillicon

python3 -m venv .venv
source .venv/bin/activate

python -m pip install -U pip setuptools wheel
python -m pip install -e .
```

## Run Modes

### 1) Runtime only (OpenAI-compatible server)

```bash
vllm-mlx serve ibm-granite/granite-4.0-h-350m --host 127.0.0.1 --port 8000
```

### 2) Web UI only (connect to an existing backend)

```bash
token-workshed-ui \
  --server-url http://127.0.0.1:8000 \
  --host 127.0.0.1 \
  --port 7862 \
  --open-browser
```

### 3) Desktop all-in-one manager (recommended)

```bash
./script/build_and_run.sh
```

The source-checkout launcher rebuilds the native UI, refreshes
`dist/token-workshed.app`, and starts the current 0.2.0 checkout. This avoids
using a stale console script or an older app bundle from `/Applications`.

Optional explicit model:

```bash
token-workshed-desktop mlx-community/Llama-3.2-3B-Instruct-4bit
```

Useful options:

```bash
token-workshed-desktop --server-port 8000 --ui-port 7862 --browser
```

Remote UI (trusted network only):

```bash
token-workshed-desktop --ui-host 0.0.0.0 --allow-remote-ui
```

- By default, non-loopback UI hosts are rejected.
- In remote mode, manager endpoints require header:
  - `X-Token-Workshed-Manager-Token: <printed-token>`
- For the built-in UI, open once with:
  - `http://<host>:7862/?manager_token=<printed-token>`
- You can provide your own fixed token with:
  - `TOKEN_WORKSHED_MANAGER_TOKEN=...`
- To allow non-local backend targets in UI proxy mode, set:
  - `TOKEN_WORKSHED_ALLOWED_SERVER_HOSTS=host1,host2`

## First Launch Behavior

- On first open, a one-time Hugging Face binding modal is shown.
- Binding info is stored in local desktop state:
  - macOS: `~/Library/Application Support/token-workshed/desktop_state.json`
- If no local model exists, the app still opens in no-active-model mode.

## Build macOS App and Installer

### Build `.app`

```bash
cd native-ui && cargo build --release
cd ..
bash scripts/build_macos_app_bundle.sh
```

Output:

- `dist/token-workshed.app`
- `dist/token-workshed-ide.app` (standalone Rust/libcosmic Companion, if used)

The PyInstaller spec is kept for compatibility, but the native UI build path uses
`scripts/build_macos_app_bundle.sh` to avoid PyInstaller dependency scanning.

### Build `.pkg` installer

```bash
bash scripts/build_pkg_installer.sh 0.2.0 0.4.0 3.12.8 [python_pkg_sha256]
```

Output:

- `dist/token-workshed-0.2.0-installer.pkg`

Installer payload now includes:

- UI app bundle to `/Applications/token-workshed.app`
- Full 0.2.0 source/integration bundle to `/Library/Application Support/token-workshed/token-workshed-0.2.0` (including `integrations/openclaw` + `integrations/hermes`)
- Runtime postinstall bootstrap (Python + `vllm-mlx`) as before

Source bundle excludes packaging/dev caches:

- `.git/`
- `.venv/`
- `.pytest_cache/`
- `.openclaw/`
- `.DS_Store`
- `build/`
- `dist/`
- `.codex/` and local credential directories

This 0.2.0 installer is **unsigned and not notarized**. Gatekeeper may block it
on other Macs; see the release notes for the current distribution limitation.

### Install generated `.pkg`

```bash
sudo installer -pkg "dist/token-workshed-0.2.0-installer.pkg" -target /
open "/Applications/token-workshed.app"
```

## Project Layout

```text
vllm_mlx/
  desktop_ui.py              # desktop manager + manager API endpoints
  css_svg_ui.py              # web UI backend (FastAPI)
  ui_css_svg/                # frontend assets (HTML/CSS/JS)
scripts/
  token_workshed_app_entry.py
  build_pkg_installer.sh
token-workshed.spec          # PyInstaller spec
```

## Troubleshooting

### App closes on startup

Reset desktop state and retry:

```bash
rm -f "$HOME/Library/Application Support/token-workshed/desktop_state.json"
```

Then start again:

```bash
open "/Applications/token-workshed.app"
```

### `ModuleNotFoundError` (for local source run)

Activate venv and install dependencies:

```bash
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e .
```

### Hugging Face 401 / gated model errors

- Bind a valid Hugging Face token in the app popup (or rebind later through app flow)
- Ensure you accepted model terms on the model’s Hugging Face page

## License

This project remains under Apache-2.0, following the upstream license.

## Acknowledgements

- Upstream: [`waybarrios/vllm-mlx`](https://github.com/waybarrios/vllm-mlx)
- Apple ML stack: MLX / mlx-lm / mlx-vlm
