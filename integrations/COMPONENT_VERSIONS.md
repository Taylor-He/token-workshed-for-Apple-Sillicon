# NextCore Component Versions

Token Workshed NextCore runs the following application-owned inference and
computer-control components:

| Component | Pin | Purpose |
| --- | --- | --- |
| SGLang Runtime | `v0.5.22` (`baa9bef`) | Local model serving and tool-call parsing |
| Python | `3.12` | Application and Agent runtime |
| CUA Driver | `v0.28.1` | macOS desktop and browser control, downloaded on first use |

NextCore does not start OpenClaw or the Hermes CLI. Its Agent loop, tool
execution, permissions, cancellation, and audit are owned by the NextCore
application. The computer-use adapter in `token_workshed_nextcore/agent/` is
adapted from Hermes Agent under MIT; see its retained license and source notice
in `token_workshed_nextcore/agent/third_party/`.
