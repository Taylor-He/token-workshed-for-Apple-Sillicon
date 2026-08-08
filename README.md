[README.md](https://github.com/user-attachments/files/26143061/README.md)
# Token Workshed(for Apple Sillicon)

Token Workshed(mlx) is a custom desktop-focused fork of `vllm-mlx` for Apple Silicon Macs.
It combines:

- a local `vllm-mlx` runtime
- a custom CSS+SVG chat UI
- a desktop model manager (model switching, community search/deploy, runtime options)
- macOS `.pkg` packaging workflow

Current desktop release line in this repo:

- `token-workshed`: `0.0.1`
- `vllm-mlx`: `0.2.6`

## Highlights

- Native Apple Silicon local inference (text + multimodal support from `vllm-mlx`)
- Desktop manager mode (`token-workshed-desktop`) with local model lifecycle control
- Community page with Hugging Face model search and one-click download/deploy
- One-time Hugging Face account binding popup (saved locally for authenticated search/download)
- Startup model integrity check and cleanup for incomplete local caches
- If no model is installed, UI still opens and shows:
  - `To start using Token Workshed, please install a model.`

## Requirements

- mac with Apple Silicon(Essential)
- Python `>= 3.10`
- Network access (for Hugging Face search/download)

## Install From Source

```bash
git clone https://github.com/<your-account>/token-workshed.git
cd token-workshed

python3 -m venv .venv
source .venv/bin/activate

python -m pip install -U pip setuptools wheel
python -m pip install -e .
```
### You can also install the `.pkg` file from the releases page.  

## Run Modes

### 1) Runtime only (OpenAI-compatible server)

```bash
vllm-mlx serve your model --host 127.0.0.1 --port 8000
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
token-workshed-desktop
```

Optional explicit model:

```bash
token-workshed-desktop your model
```

Useful options:

```bash
token-workshed-desktop --server-port 8000 --ui-port 7862 --browser
```

## First Launch Behavior

- On first open, a one-time Hugging Face binding modal is shown.
- Binding info is stored in local desktop state:
  - macOS: `~/Library/Application Support/token-workshed/desktop_state.json`
- If no local model exists, the app still opens in no-active-model mode.

## Build macOS App and Installer

### Build `.app` with PyInstaller

```bash
source .venv/bin/activate
pyinstaller --noconfirm token-workshed.spec
```

Output:

- `dist/token-workshed.app`

### Build `.pkg` installer

```bash
bash scripts/build_pkg_installer.sh 0.0.1 0.2.6 3.12.8
```

Output:

- `dist/token-workshed-0.0.1-installer.pkg`

### Install generated `.pkg`

```bash
sudo installer -pkg "dist/token-workshed-0.0.1-installer.pkg" -target /
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
# Latest version v0.1.5 features:
oken-workshed v0.1.5

> macOS Apple Silicon desktop release · Configure AI Agents

v0.1.5 adds model-aware agent setup, a more responsive native desktop UI, and
a safer macOS packaging pipeline.

## Highlights

### Configure AI Agents

- New per-model **Configure AI Agents** flow for OpenClaw and Hermes.
- The app generates and stores a local Agent Profile for each model instead of
  relying on a manually maintained global parameter table.
- Profiles record the detected model family, context window, chat template,
  parser choices, reasoning support, tool-call capability, timeout, and runtime
  probe results.
- Model Settings now shows the current profile status and supports deleting or
  recalibrating a model profile.
- Failed configuration retains useful diagnostics and can be retried; direct
  chat remains available while an agent runtime is unconfigured.

### Better agent and chat experience

- OpenClaw and Hermes use the selected model profile as their runtime default.
- Tools are exposed according to the active request context, allowing the model
  to decide whether to return a tool call instead of forcing tool use from the
  UI.
- Completed reasoning is collapsed into a compact `thought for … seconds`
  summary.
- The Chat composer keeps a stable layout while a model is being configured,
  configured, or recalibrated.
- The Terminal page is now a single terminal-style workspace for entering and
  observing vllm-mlx commands.

### Faster and more stable native UI

- Reduced redundant background polling; inactive pages no longer continuously
  poll backend endpoints.
- Added in-flight request protection and response generations so stale status
  responses cannot overwrite a newer model switch.
- Hardened streaming chat with request IDs, duplicate-send protection, delta
  coalescing, UTF-8-safe chunk decoding, and explicit handling for streams that
  end without a terminal event.
- Desktop state is saved atomically and remains compatible with legacy state.
  Corrupt state is backed up instead of being silently overwritten.
- Unified Python-side state updates so Agent Profiles and desktop preferences
  cannot overwrite one another during concurrent writes.
- Terminal updates avoid replacing unchanged output, and attachment size is
  checked before and after reading.

### Packaging and release hygiene

- macOS installer staging excludes local state, API/token files, browser
  sessions, logs, caches, virtual environments, build output, and debug files.
- The installer fails closed if sensitive or generated payload escapes its
  staging filters.
- Obsolete build products and unused vendor dependencies were cleaned before
  packaging.

## Release assets

- `token-workshed.app`
- `token-workshed-0.1.5-installer.pkg`

The installer places the desktop app in `/Applications`, source and
integrations in `/Library/Application Support/token-workshed/`, and creates a
dedicated vllm-mlx runtime on first install.

## Requirements and known limitations

- macOS 13 or later on Apple Silicon (`arm64`).
- First installation requires internet access to bootstrap Python and the
  vllm-mlx runtime.
- This build is currently **unsigned and not notarized**. It is suitable for
  local testing, but Gatekeeper may block it on another Mac. A public release
  still requires Developer ID signing and Apple notarization.

## Validation

The release candidate was validated locally with:

- `cargo fmt -- --check`
- `cargo check --locked --offline`
- `cargo test --locked --offline` — 16 native UI tests passed
- Focused Python profile, UI, routing, integrity, and terminal tests — 39 tests
  passed
- Installer payload expansion and sensitive-path audit

## Upgrade notes

- Existing local model weights are not removed by this release.
- Agent Profiles are model-specific. Use **Configure AI Agents** after choosing
  a model with no profile, or recalibrate it from Model Settings.
- The new installer intentionally does not carry over local tokens, browser
  sessions, agent state, or desktop caches from a development machine.
