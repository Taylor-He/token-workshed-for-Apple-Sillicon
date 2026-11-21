# token-workshed v0.1.5

v0.1.5 introduces **Configure AI Agents**: a per-model setup flow that creates
and stores local Agent Profiles for OpenClaw and Hermes. Profiles detect and
reuse the model's runtime defaults, parser settings, tool-call support, and
probe results.

This release also improves the native desktop experience with safer model
switching, more reliable streaming chat, collapsed thinking summaries, a
single terminal-style vllm-mlx workspace, reduced background polling, and
atomic desktop-state saves.

The macOS installer now excludes local state, tokens, browser sessions, caches,
and build artifacts, and fails closed if sensitive files enter the release
payload.

**Requirements:** macOS 13+ on Apple Silicon. First install needs internet to
bootstrap Python and vllm-mlx.

**Important:** This build is currently unsigned and not notarized, so it is
intended for local testing. A public release still needs Developer ID signing
and Apple notarization.

Validated locally: 16 native UI tests and 39 focused Python tests passed.
