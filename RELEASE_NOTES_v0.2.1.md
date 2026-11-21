# token-workshed NextCore 0.2.1

NextCore is the SGLang-based macOS Apple Silicon line of Token Workshed. This
build targets macOS 14 or later and Python 3.12.

## Highlights

- Pins SGLang Runtime v0.5.22 at commit
  `baa9befa381511cd4d56f83980951e5a45fffc6d` for local model serving.
- Keeps the OpenAI-compatible model, chat, and completion routes while
  restricting Agent tools to authenticated NextCore in-app chat.
- Adds the app-managed Agent loop with web search and fetch, terminal and file
  tools, browser and computer-use controls, memories, Skills, MCP connections,
  scheduled tasks, and local activity records.
- Adds separate local and RoboServer runtime targets. Remote management uses
  SSH host-key verification and a tunnel to the server's private API.
- Routes Workshed teacher generation, benchmarks, and evaluation through
  SGLang. Loss scoring uses the temporary Torch/MPS SGLang backend. Retired
  training and non-SGLang conversion steps remain readable in saved work orders
  but are rejected before execution.
- Uses a white app canvas with neutral gray message and history surfaces.

## Supported and unavailable features

- Text serving targets Apple Silicon through the pinned SGLang runtime.
- Image, video, ASR, and TTS capabilities remain unavailable until validated
  for each model and macOS runtime. SGLang-Omni remains a separate experimental
  runtime and is not a fallback.
- No model execution falls back to vLLM-MLX.

## Validation

- CI product-contract set covering NextCore Agent/runtime, RoboServer transfer,
  model management, Workshed execution, APIs, MCP, and security: 506 passed.
- Native UI unit tests: 58 passed; release build and formatting check passed.
- Python undefined-name and unused-variable checks passed for the app and the
  focused test modules.
- The packaged app bundle verifies with `codesign --verify --deep --strict`.
  The app and installer are not Developer ID signed or notarized on this build.
- Package output is a local release artifact. It was built but not installed or
  published to GitHub.
