# Workshed and Terminal: selected third design

## Target and implementation

- User selected the third paired design, with one change: type Terminal commands inline after the transcript rather than in a bottom composer.
- Original target: `docs/design/workshed-terminal-selected-third.png`.
- Revised target: `docs/design/workshed-terminal-inline-reference.png`. The built-in image generator made only the requested composer-to-prompt change; the generated reference was copied into this checkout.
- Native implementation: `native-ui/src/app/workshed_view.rs`, `native-ui/src/app/terminal_view.rs`, and the existing schema-driven inspector in `native-ui/src/app.rs`.
- Both pages use Chat's sans-serif UI sizing, light Caligo surfaces, 8px corners and compact controls. Only terminal commands/output use monospace.
- The Workshed stack is top-aligned and fills its available track. Source, Quantize and Save controls align to the same column. A plain 240px inspector retains all advanced parameters, typed bindings, reset actions, issues and effective parameter origins.
- Preset and bit selectors use manager definitions. Explicit overrides survive preset changes. The root model-name placeholder remains `input the name of the new model`.
- The running footer shows actual progress with a black progress bar; Cancel is exposed only when the run permits it. Idle/completed states show real state, not the mock's fixed 42%.
- Terminal no longer contains fake traffic lights, fixed `80x24`, or a fictitious shell-switching control. Its context denotes the one actual local session; the current backend does not expose shell switching.

## Automated checks

- Rust: 47 tests passed, including new terminal grouping/input style, monochrome controls, effective quantization defaults, invalid field guards and stale preflight-generation tests.
- Python: 77 tests passed across Workshed, API contract, managed lifecycle, quantization, terminal, model integrity/details and agent profiles.
- Scope: no model quantization, downloads, training or model deletion was performed for this redesign.

## Live native checks

- Canonical release build/run: `TOKEN_WORKSHED_INITIAL_PAGE=workshed ./script/build_and_run.sh --verify`.
- Main bundle startup was verified at the exact checkout `dist/token-workshed.app/Contents/MacOS/token-workshed-native-ui` path. The main wrapper's automation binding timed out, so live screenshots and input checks used direct-entry diagnostic bundles containing the same native program.
- Workshed screenshot: readable source repository, aligned controls, top-aligned root and three blocks, visible pinned Run action, no observed overlap in the inspected state.
- Found two monochrome mismatches during inspection: the native checkbox accent was teal and the Terminal context icon inherited a pale tint. Applied a local monochrome native theme and an explicit black SVG tint respectively.
- Terminal keyboard flow: Tab focused the inline input, typed `printf 'TOKEN_WORKSHED_INLINE_UI_OK\n'`, then Enter. The UI showed the echoed command, actual `TOKEN_WORKSHED_INLINE_UI_OK` output, and a fresh inline prompt. This used the existing local manager terminal endpoint, not a CLI/API submission bypass.
- The source reference and captured Terminal screen were emitted in the same comparison tool input. Screenshot capture and keyboard input worked; coordinate clicks repeatedly returned `noWindowsAvailable` despite the visible window.
- Final follow-up used the optimized executable freshly copied from the project bundle into both existing diagnostic bundles. Joint-input reference comparisons confirmed the black checkbox and black terminal icon. A second real UI command printed `TOKEN_WORKSHED_RELEASE_INLINE_OK` and returned to the inline prompt.
- Final main and companion bundles passed deep strict signature verification and plist lint. Build/run completed successfully. Test windows were closed after inspection; the main current-checkout bundle was left running.

## Acceptance boundary

The code, tests, optimized build, post-fix monochrome captures and inline Terminal execution are verified. Full 1:1 visual acceptance is still blocked: the generated paired reference has no declared pixel density, the observed native viewport differed, coordinate automation could not open the Workshed inspector, and controlled 980x680/1440x1024 plus VoiceOver checks remain pending. Diagnostic-window checks are not a substitute for full main-wrapper acceptance. `/Applications` was not replaced.
