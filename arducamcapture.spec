# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

root = Path(SPECPATH)
resources = root / "src" / "arducam_photo" / "resources"
required_resources = (
    "arducam_108mp.json",
    "icons/nuts-app.svg",
    "icons/nuts-app.png",
    "icons/nuts-app.ico",
    "icons/squirrel.svg",
    "icons/refresh-cw.svg",
    "icons/LICENSE",
    "powered by_white.png",
)
missing = [name for name in required_resources if not (resources / name).is_file()]
if missing:
    raise FileNotFoundError(
        "Missing required original package resources: "
        + ", ".join(f"src/arducam_photo/resources/{name}" for name in missing)
    )

a = Analysis(
    [str(root / "packaging" / "launch_arducam.py")],
    pathex=[str(root / "src")],
    binaries=[],
    datas=[(str(resources), "arducam_photo/resources")],
    hiddenimports=[
        "arducam_photo.resources",
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        "PySide6.QtSvg",
    ],
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
    name='ArducamCapture',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(resources / "icons" / "nuts-app.ico"),
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
    upx=False,
    upx_exclude=[],
    name='ArducamCapture',
)
