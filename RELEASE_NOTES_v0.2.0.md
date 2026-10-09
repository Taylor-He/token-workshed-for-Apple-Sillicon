# Token Workshed 0.2.0

Token Workshed 0.2.0 introduces **Workshed**, a native workspace for developing
and quantizing local models on Apple Silicon.

## What's new

- Renamed the native `Quantize` page to **Workshed** and made it the home for
  local model-development work orders.
- Added editable model, data, quantization, evaluation, and delivery settings,
  with a focused inspector for block parameters and a preflight plan before a
  run begins.
- Added a real local MLX-LM 4-bit quantization flow with managed artifact
  registration. A model produced through the UI was subsequently started from
  the model manager; see the [end-to-end test record](docs/testing/2026-10-08-ui-quantization-e2e.md).
- Added clearer run status, artifact handling, and configurable parameters for
  model-development steps. Step availability is capability-gated by the local
  runner; unsupported operations are not silently substituted.
- Refined the native Chat, Workshed, and Terminal surfaces into a consistent,
  restrained monochrome desktop style.
- Preserved the legacy quantization API and existing managed quantized models.

## Requirements and first launch

- macOS on Apple Silicon (`arm64`).
- An internet connection is required on first install to bootstrap Python and
  the `vllm-mlx` runtime; model downloads require network access as well.
- The installer places the app in `/Applications` and installs its source and
  integrations under `/Library/Application Support/token-workshed/`.

## macOS distribution status

This release's `.pkg` is currently **unsigned and not notarized** because no
Apple Developer ID signing identity or notarization profile is configured on
the build machine. Gatekeeper may block installation on other Macs. A signed,
notarized installer is a separate follow-up once those Apple credentials are
available.

## Validation scope

The small-model UI quantization and launch path was exercised locally. This does
not imply that every training, distillation, evaluation, model, or control-flow
combination has been validated, nor does it replace full visual, accessibility,
or release-signing acceptance.
