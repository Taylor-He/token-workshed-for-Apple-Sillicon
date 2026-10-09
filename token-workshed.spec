# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules

_spec_file = globals().get("__file__", "")
ROOT_DIR = Path(_spec_file).resolve().parent if _spec_file else Path.cwd().resolve()
datas = []
hiddenimports = []
datas += collect_data_files('vllm_mlx')
datas += collect_data_files('mlx')
datas += collect_data_files('mlx_vlm')
datas = [
    item for item in datas
    if 'ui_css_svg' not in Path(item[0]).parts
]
hiddenimports += collect_submodules('vllm_mlx')
hiddenimports += collect_submodules('mlx')
hiddenimports += collect_submodules('mlx_lm')
hiddenimports += collect_submodules('mlx_vlm')
native_ui_binary = ROOT_DIR / 'native-ui' / 'target' / 'release' / 'token-workshed-native-ui'
if not native_ui_binary.exists():
    raise SystemExit(
        f'Missing native UI binary: {native_ui_binary}. '
        'Run `cd native-ui && cargo build --release` before building the app bundle.'
    )
binaries = [(str(native_ui_binary), '.')]


a = Analysis(
    ['scripts/token_workshed_app_entry.py'],
    pathex=[str(ROOT_DIR)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='token-workshed',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='token-workshed',
)
app = BUNDLE(
    coll,
    name='token-workshed.app',
    icon=None,
    bundle_identifier=None,
)
