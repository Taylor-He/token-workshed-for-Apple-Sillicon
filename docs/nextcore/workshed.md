# Workshed with SGLang

Every runnable block that performs model execution uses SGLang Runtime
`v0.5.22`:

| Operation | SGLang backend | Runtime handling |
| --- | --- | --- |
| Teacher generation | MLX | Start a temporary model server, then restore the chat server |
| Language-model evaluation | MLX | Run evaluation requests against a temporary SGLang server |
| Benchmark | MLX | Send generation requests to SGLang |
| Loss/perplexity evaluation | Torch/MPS | Score prompt log probabilities, then restore chat |
| Dataset preparation, report export, registration | No inference | Run as local App-managed data operations |

Every model listed in NextCore can be selected and sent directly to SGLang.
NextCore does not keep per-checkpoint generation or scoring verification
records. If SGLang cannot load a checkpoint or perform an operation, the actual
runtime error is reported; there is no alternate-engine fallback.

SFT, LoRA, QLoRA, DoRA, full fine-tuning, DPO, distillation, adapter fusion,
and non-SGLang model conversion are removed from runnable Workbench
capabilities. Existing work orders remain readable, but preflight rejects any
work order that contains one of those operations before a process is started.

Temporary SGLang servers are stopped and the previous primary/extra model
services and selected active model are restored after success, failure, or
cancellation. A restore failure is reported as a Workshed error.

Image/video evaluation and ASR/TTS remain unavailable until a model-specific
SGLang path has been validated on the target macOS runtime.
