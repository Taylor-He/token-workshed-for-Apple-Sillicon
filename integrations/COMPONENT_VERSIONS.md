# Bundled Component Versions

This file records the versions shipped inside the `token-workshed-0.2.0`
source bundle. It describes project-local integrations only; it does not
change or pin globally installed `openclaw` or `hermes` commands.

| Component | Bundled path | Source pin | Version | Runtime requirement |
| --- | --- | --- | --- | --- |
| OpenClaw | `integrations/openclaw` (`cypherclaw` target) | `v2026.7.1` | `2026.7.1` | Node `24.15+` recommended; `22.22.3+`, `25.9+` supported |
| Hermes Agent | `integrations/hermes` | `v2026.7.7.2` | `0.18.2` | Python `>=3.11,<3.14` |
| Node runtime | `integrations/node-runtime` | `node-v24.15.0-darwin-arm64` | `24.15.0` | macOS arm64 bundle for OpenClaw |

## Integration Notes

- OpenClaw is built from the pinned source with pnpm `11.2.2`; its generated
  `dist/` runtime is included in the project-local checkout.
- The desktop bridge uses `integrations/node-runtime/bin/node` first, so the
  bundled OpenClaw does not depend on the system Node version. Set
  `TOKEN_WORKSHED_OPENCLAW_NODE_BIN` to use another supported runtime.
- Hermes is installed editable into
  `integrations/hermes/.token-workshed-state/venv`. The bridge rebuilds that
  venv when its Python version is outside the supported range and resyncs
  dependencies when the vendored Hermes version changes.
- Both components keep their token-workshed state under
  `.token-workshed-state`; updating source does not intentionally replace
  local configuration, sessions, workspace files, or credentials.
- The Python bridge detects optional OpenClaw agent flags at runtime so older
  checkouts and the current CLI can share the same integration code.
