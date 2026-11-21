# Examples

`demo_openai_text.py` is the maintained NextCore example. It discovers the
served model from `/v1/models` and uses the OpenAI-compatible SGLang endpoint
on port `8010` by default.

Most other examples in this directory were carried over from Token Workshed
0.2.1. Their commands and direct MLX engine imports target the retired
vLLM-MLX runtime. Treat them as historical source until they are migrated.
Image, video, ASR, and TTS examples are not evidence that NextCore supports
those capabilities; those remain disabled pending per-model and per-system
validation.
