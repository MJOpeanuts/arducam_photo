"""Shooting profile and injectable storage (JSON file by default)."""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Optional, Protocol

from ..config import APIS, NATIVE_108MP, PATHS

FOCUS_MIN, FOCUS_MAX = 0, 1023  # B0494C reference range; the adapter may report otherwise
APP_DIR_NAME = "ArducamCapture"


@dataclass(frozen=True)
class CameraId:
    """Best identification available. With OpenCV alone only the index is known (fragile)."""

    key: str
    label: str
    index: int
    fragile: bool = True

    @staticmethod
    def from_index(index: int) -> "CameraId":
        return CameraId(f"index:{index}", f"Caméra index {index}", index, True)


@dataclass(frozen=True)
class ShootingProfile:
    """Shooting settings only. The RAW transport and the preview video mode are NOT part of it.

    focus_requested: value asked by the user. focus_readback: value read from the driver
    (may differ; not proof of physical effect). focus_verified is None when it cannot be verified.
    """

    camera_key: str = ""
    camera_index: int = 0
    camera_fragile: bool = True
    path: str = NATIVE_108MP
    api: str = "msmf"
    focus_requested: Optional[int] = None
    focus_readback: Optional[float] = None
    focus_verified: Optional[bool] = None
    saved_at: Optional[str] = None

    def problems(self) -> list:
        out = []
        if self.path not in PATHS:
            out.append(f"parcours inconnu {self.path!r}")
        if self.api not in APIS:
            out.append(f"API inconnue {self.api!r}")
        if not isinstance(self.camera_index, int) or self.camera_index < 0 or not self.camera_key:
            out.append("caméra non définie")
        f = self.focus_requested
        if f is None or isinstance(f, bool) or not isinstance(f, int) or not FOCUS_MIN <= f <= FOCUS_MAX:
            out.append("focus non défini ou hors plage")
        return out

    @property
    def is_valid(self) -> bool:
        return not self.problems()

    def same_settings(self, other: Optional["ShootingProfile"]) -> bool:
        """Compare what the user chose; readbacks and timestamps are not user changes."""
        if other is None:
            return False
        keys = ("camera_key", "camera_index", "path", "api", "focus_requested")
        return all(getattr(self, k) == getattr(other, k) for k in keys)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ShootingProfile":
        names = {f.name for f in dataclasses.fields(ShootingProfile)}
        return ShootingProfile(**{k: v for k, v in d.items() if k in names})


class ProfileStore(Protocol):
    """Implement this to let a host application keep profiles elsewhere."""

    def load(self, camera_key: str) -> Optional[ShootingProfile]: ...

    def save(self, profile: ShootingProfile) -> None: ...


class StoreError(Exception):
    pass


def app_data_dir() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, APP_DIR_NAME)


def _atomic_write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _read_json(path: str) -> dict:
    """Return {} when absent. A corrupt file raises StoreError and is never deleted."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        raise StoreError(f"fichier illisible {path}: {e}") from e
    if not isinstance(data, dict):
        raise StoreError(f"fichier invalide {path}: objet JSON attendu")
    return data


def _backup_corrupt(path: str) -> None:
    if os.path.exists(path):
        os.replace(path, f"{path}.corrupt-{time.strftime('%Y%m%d-%H%M%S')}")


class JsonProfileStore:
    """profiles.json in %LOCALAPPDATA%\\ArducamCapture (or the given path)."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or os.path.join(app_data_dir(), "profiles.json")

    def load(self, camera_key: str) -> Optional[ShootingProfile]:
        d = _read_json(self.path).get("profiles", {}).get(camera_key)
        if d is None:
            return None
        try:
            return ShootingProfile.from_dict(d)
        except TypeError as e:
            raise StoreError(f"profil invalide pour {camera_key}: {e}") from e

    def save(self, profile: ShootingProfile) -> None:
        try:
            data = _read_json(self.path)
        except StoreError:
            _backup_corrupt(self.path)  # keep the unreadable file, start a new one
            data = {}
        profiles = data.get("profiles") if isinstance(data.get("profiles"), dict) else {}
        stamped = dataclasses.replace(profile, saved_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        profiles[profile.camera_key] = stamped.to_dict()
        _atomic_write_json(self.path, {"version": 1, "profiles": profiles})


@dataclass(frozen=True)
class AppSettings:
    photo_dir: Optional[str] = None  # None = system Pictures\ArducamCapture
    ccm_path: Optional[str] = None
    camera_index: int = 0
    path: str = NATIVE_108MP
    start_fullscreen: bool = False


class JsonSettingsStore:
    def __init__(self, path: Optional[str] = None):
        self.path = path or os.path.join(app_data_dir(), "settings.json")

    def load(self) -> AppSettings:
        d = _read_json(self.path)
        names = {f.name for f in dataclasses.fields(AppSettings)}
        return AppSettings(**{k: v for k, v in d.items() if k in names})

    def save(self, settings: AppSettings) -> None:
        _atomic_write_json(self.path, dataclasses.asdict(settings))
