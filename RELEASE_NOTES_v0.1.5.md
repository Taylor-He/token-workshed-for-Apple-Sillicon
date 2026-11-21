# token-workshed v0.1.5

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
