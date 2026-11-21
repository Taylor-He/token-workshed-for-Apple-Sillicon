# Install token-workshed NextCore from source

## Requirements

- Apple Silicon Mac running macOS 14 or later.
- Python 3.12.
- Network access for the pinned SGLang source checkout and Python dependencies.

## Runtime setup

From the NextCore repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[ide]'
scripts/install_sglang_runtime.sh
```

The runtime installer checks out SGLang tag `v0.5.22`, verifies commit
`baa9befa381511cd4d56f83980951e5a45fffc6d`, and installs its `srt_mps` extra
into the NextCore Python environment. It rejects Python versions other than
3.12 and verifies the installed distribution version.

Start the desktop manager:

```bash
token-workshed-nextcore-desktop
```

The default model server and local UI/API ports are `8010` and `7872`. The
standalone text-server command is:

```bash
token-workshed-nextcore serve MODEL_ID --host 127.0.0.1 --port 8010
```

## App bundle and installer

Build the Rust/libcosmic app and bundle locally:

```bash
(cd native-ui && cargo build --release --locked)
bash scripts/build_macos_app_bundle.sh
```

Build a `.pkg` after the app bundle exists:

```bash
bash scripts/build_pkg_installer.sh
```

The outputs are `dist/nextcore.app` (display name `token-workshed NextCore`)
and `dist/token-workshed NextCore.pkg`. The bundle
identifier is `com.taylorhe.tokenworkshed.nextcore`; app state uses the
`token-workshed-nextcore` data directory. The installer stores source at
`/Library/Application Support/token-workshed-nextcore/source` and installs
`nextcore.app`, displayed as token-workshed NextCore, beside the existing
Token Workshed app.

## Optional Omni runtime

SGLang-Omni is separate from the main SGLang environment because its dependency
set differs. The experimental installer currently requires Homebrew `uv` and
`ffmpeg@7`. Only validated models and endpoints should be presented as
available; the main NextCore runtime does not fall back to Omni.
