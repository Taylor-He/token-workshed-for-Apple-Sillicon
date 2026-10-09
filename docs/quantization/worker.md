# CUDA Quant Worker

`vllm_mlx.quant_worker:create_quant_worker_app` is a small FastAPI contract for
the remote half of the Quantize page. Run it behind an HTTPS reverse proxy and
set:

```text
TOKEN_WORKSHED_QUANT_WORKER_TOKEN=<worker-secret>
TOKEN_WORKSHED_QUANT_WORKER_NAME=worker-gpu-01
TOKEN_WORKSHED_QUANT_WORKER_GPU=H100
TOKEN_WORKSHED_QUANT_WORKER_MEMORY=80GB
```

The Worker intentionally refuses to start without a token. It accepts Hub
sources only, advertises its supported presets, reports progress and a vLLM
deployment command, and serves a checksum-verifiable ZIP artifact. A
deployment supplies an LLM Compressor wrapper with
`TOKEN_WORKSHED_QUANT_WORKER_COMMAND`; the wrapper receives `--request` and
`--output` paths and writes the compressed-tensors artifact into the output
workspace. The desktop manager never forwards a Hugging Face token. On
completion the Worker keeps a deployable model directory alongside the
downloadable ZIP and returns a concrete
`vllm serve ... --quantization compressed-tensors` command; the ZIP is only
the transport format for the optional Mac delivery path.
