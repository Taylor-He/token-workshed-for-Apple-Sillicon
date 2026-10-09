# UI quantization end-to-end — 2026-10-08

## Result

Pass for the tested local chain: **UI creation → source selection → 4-bit MLX quantization → registration → UI model switch → real chat generation → cleanup**. This is an inference smoke test, not a model-quality or performance benchmark.

The earlier Workshed label overflow, missing new/add entry points, oversized settings gear, inspector scrollbar overlap, and Community null-job card were corrected. Mouse-based checks confirmed the readable clipped source label, new work-order creation, block insertion/removal, work-order selection, and Community's `No active job` state after polling.

## Runtime boundary

- Checkout: the Token Workshed 0.2.0 source checkout.
- The actual quantization and chat operations used a freshly built debug native executable, staged as the direct entry point of a temporary ad-hoc-signed diagnostic app. It used the real checkout manager and MLX runner; no mock UI, CLI job submission, or API job submission was used.
- The normal Swift wrapper still could not be selected reliably through desktop automation. Its main-thread `waitUntilExit` was removed; child-process lifetime now uses a termination handler.
- After cleanup, the final optimized release binary was rebuilt and packaged with `TOKEN_WORKSHED_INITIAL_PAGE=workshed ./script/build_and_run.sh --verify`. The command exited 0, and a fresh native process was verified at the exact current `dist/token-workshed.app/Contents/MacOS/token-workshed-native-ui` path.
- Both checkout bundles passed strict deep signature verification and plist validation. `/Applications` was not replaced.
- Final release-window automation could not continue because the Mac was locked. Full packaged-window interaction, controlled 1440×1024/980×680 layouts, and VoiceOver acceptance remain pending. Native AX content still exposes window chrome rather than page controls; do not claim complete accessibility acceptance.

## UI workflow and frozen run

1. Clicked **New work order**; entered `ui-quant-granite350m-20261008`.
2. Opened the Load base model chooser, selected **Local** and cached `ibm-granite/granite-4.0-h-350m`, then clicked **Done**.
3. Kept **4-bit / Balanced / MLX · This Mac** and **Test when finished** enabled.
4. Clicked **Run quantization**. The UI showed queued progress, logs, and cancellation controls, then `Last run completed · 100%` with **View result**.
5. Clicked **View result**; selected the new model in Model Settings and clicked **Switch**.
6. Created a separate chat and sent the prompt below with Enter.

| Frozen field | Value |
| --- | --- |
| Work order | `work-order-18dc4bc5d75be860-0`, revision **3** |
| Run | `556ef35bc70e4751bf71453f7c2314fb` |
| Source snapshot | `3b17b717b8f2f5d305b0a92c1491e239aeda19c8` |
| Effective quantization | bits **4**, group size **128**, mode **affine**, preset `mlx-balanced` |
| Content artifact | `workshed/1d5956c034c85fa56d288c57` |
| Model registration | `workshed/ui-quant-granite350m-20261008` |
| Artifact size | **188,604,567 bytes** (about 189 MB) |
| Toolchain | Existing ready `mlx-train` profile; `mlx-lm==0.31.1`, `datasets==4.8.3` |

The manager's frozen runner projection was `mlx_lm.convert` with `-q --q-bits 4 --q-group-size 128 --q-mode affine`, a read-only HF snapshot input, and an owned run staging output. Load, quantize, and register steps all succeeded. The registration smoke test (`mlx_lm.generate`, 16-token limit) also passed. This run reused a ready toolchain; first-time installation, consent, and offline setup failure were not exercised.

## Real serving evidence

- Manager confirmed the new Workshed model was the sole active/running model at `http://127.0.0.1:8000`.
- UI prompt: `Introduce yourself in one short sentence.`
- Visible generated reply: `My name is Granite, a language model developed by IBM for IBM research.`
- Dashboard showed **1 request, ok 1 / err 0**, **100%** success and **3573 ms** average latency. These are observed dashboard values, not independently validated performance statistics.
- The original `hello` conversation was preserved. The separate smoke-test conversation remains as evidence.
- During testing, Model Settings initially displayed the managed model as Hugging Face / Unknown. The metadata builder was subsequently fixed to resolve the managed directory and read quantization bits from `config.json`; 20 focused metadata tests passed. This follow-up metadata rendering was not re-tested with a registered real artifact after cleanup.

## Cleanup and preserved state

1. Through Model Settings, switched back to `ibm-granite/granite-4.0-h-350m`. A manager read confirmed the test model was no longer running.
2. Selected the exact test model and clicked **Delete model**. Model count returned from 7 to **6**. This tombstones only its registered alias, retaining shared content safely.
3. Compared artifacts against the pre-test snapshot: the content artifact above did not exist before testing. Called the scoped manager artifact **trash** action for that exact ID; no other artifact was selected.
4. The manager atomically moved the generated directory to:
   the app-managed Trash location for the exact generated artifact.
   Removal is recoverable from Trash; Trash was not emptied.
5. Verified the original managed-store directory is absent, the Trash destination exists, and the original source snapshot still contains `config.json`. File-content inspection inside Trash was blocked by macOS privacy permissions; no permission bypass was attempted.
6. Restarted the checkout release bundle. Manager still reported **6 models**, the original Granite 350M active/running, no concurrent models, and no error. The test model did not return.
7. Inserted then removed one additional Load base model block in the independent test order to exercise the restored add route and Backspace removal. Its draft advanced to revision 5, while the completed run remains frozen at revision 3. Returned the UI to the original `local-draft`; its entire persisted object is unchanged from the pre-test snapshot (revision 1).

No original weight directory, HF cache, toolchain, old quantization model, or user work order was deleted. The completed run, parameter provenance, and tombstone metadata remain available for audit.

## Automated checks

- Rust: **41 passed**, `cargo test --offline --locked --package token_workshed_native_ui -q`; package-scoped formatting check passed.
- Python: **71 passed** across `test_workshed_managed_model_lifecycle.py`, `test_workshed_api_contract.py`, `test_workshed.py`, `test_desktop_ui_model_integrity.py`, `test_quantization.py`, and `test_desktop_ui_model_details.py`.
- Swift launcher typecheck and shell syntax checks passed.
- Release build, canonical build/package/start verification, both bundle signatures and both plists passed. Existing vendor/compiler warnings remain; this is not a warning-free build.

## Remaining acceptance boundaries

This result covers one cached local Granite 350M quantization and serving path. It does not establish training/DPO/distillation support, first-run toolchain installation, Hub downloading, all control containers, cancellation/recovery under a real active GPU job, every supported model, or full visual/accessibility acceptance.
