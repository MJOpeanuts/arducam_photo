"""Photo folder resolution and unique file names."""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

from .profile import APP_DIR_NAME


def system_pictures_dir() -> str:
    """Real Windows Pictures folder (follows OneDrive redirection via the Known Folder API)."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class GUID(ctypes.Structure):
                _fields_ = [("a", wintypes.DWORD), ("b", wintypes.WORD), ("c", wintypes.WORD),
                            ("d", ctypes.c_ubyte * 8)]

            # FOLDERID_Pictures {33E28130-4E1E-4676-835A-98395C3BC3BB}
            guid = GUID(0x33E28130, 0x4E1E, 0x4676, (ctypes.c_ubyte * 8)(0x83, 0x5A, 0x98, 0x39, 0x5C, 0x3B, 0xC3, 0xBB))
            out = ctypes.c_wchar_p()
            fn = ctypes.windll.shell32.SHGetKnownFolderPath
            fn.argtypes = [ctypes.POINTER(GUID), wintypes.DWORD, wintypes.HANDLE, ctypes.POINTER(ctypes.c_wchar_p)]
            if fn(ctypes.byref(guid), 0, None, ctypes.byref(out)) == 0 and out.value:
                try:
                    return out.value
                finally:
                    ctypes.windll.ole32.CoTaskMemFree(out)
        except Exception:
            pass
    return os.path.join(os.path.expanduser("~"), "Pictures")


def default_photo_dir() -> str:
    return os.path.join(system_pictures_dir(), APP_DIR_NAME)


def resolve_photo_dir(configured: Optional[str]) -> str:
    return os.path.abspath(configured) if configured else default_photo_dir()


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def unique_photo_path(directory: str, now: Optional[float] = None) -> str:
    """Return a path that does not exist yet. Never overwrites; save_png also refuses to."""
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
    base = os.path.join(directory, f"ArducamCapture_{stamp}")
    candidate, n = base + ".png", 1
    while os.path.exists(candidate):
        n += 1
        candidate = f"{base}_{n}.png"
    return candidate


def open_in_system(path: str) -> None:
    """Open a file/folder with the OS handler (Windows: startfile)."""
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606
    else:
        import subprocess
        subprocess.Popen(["xdg-open", path])
