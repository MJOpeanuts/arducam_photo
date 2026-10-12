"""Validated, revisioned shooting banks stored independently of legacy profiles."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
from typing import Protocol, runtime_checkable
from uuid import uuid4

from .app.profile import ShootingProfile, StoreError, app_data_dir

SCHEMA_VERSION = 1
_OBSERVATIONS = {"focus_readback", "focus_verified", "saved_at"}
_SHOOTING_FIELDS = {f.name for f in fields(ShootingProfile)} - _OBSERVATIONS


def _text(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise StoreError(f"{field} must be a nonempty string")


def _positive_integer(value: object, field: str) -> None:
    if type(value) is not int or value < 1:
        raise StoreError(f"{field} must be a positive integer")


def _validate_configuration(configuration: object) -> ShootingProfile:
    if not isinstance(configuration, dict) or set(configuration) != {"shooting", "post_processing"}:
        raise StoreError("configuration must contain shooting and post_processing")
    shooting = configuration["shooting"]
    if not isinstance(shooting, dict) or set(shooting) != _SHOOTING_FIELDS:
        raise StoreError("invalid shooting configuration fields")
    if not isinstance(configuration["post_processing"], dict) or configuration["post_processing"]:
        raise StoreError("schema 1 requires an empty post_processing object")
    _text(shooting["camera_key"], "camera_key")
    if type(shooting["camera_index"]) is not int or shooting["camera_index"] < 0:
        raise StoreError("camera_index must be an integer >= 0")
    if type(shooting["camera_fragile"]) is not bool:
        raise StoreError("camera_fragile must be boolean")
    for field in ("path", "api"):
        _text(shooting[field], field)
    for field in ("auto_exposure_mode", "auto_wb_mode"):
        if shooting[field] is not None and not isinstance(shooting[field], str):
            raise StoreError(f"{field} must be a string or None")
    try:
        profile = ShootingProfile(**shooting)
        problems = profile.problems()
    except (TypeError, ValueError, OverflowError) as exc:
        raise StoreError(f"invalid shooting configuration: {exc}") from exc
    if problems:
        raise StoreError("invalid shooting configuration: " + "; ".join(problems))
    return profile


def _unique_object(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise StoreError(f"duplicate configuration field {key!r}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise StoreError(f"nonfinite JSON value {value}")


def _decode(config_json: str) -> dict:
    if not isinstance(config_json, str):
        raise StoreError("config_json must be a JSON string")
    try:
        configuration = json.loads(
            config_json, object_pairs_hook=_unique_object, parse_constant=_invalid_constant
        )
        _validate_configuration(configuration)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise StoreError(f"invalid configuration JSON: {exc}") from exc
    return configuration


def _encode(configuration: dict) -> str:
    try:
        return json.dumps(configuration, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise StoreError(f"configuration cannot be serialized: {exc}") from exc


def _profile_json(profile: ShootingProfile) -> str:
    if not isinstance(profile, ShootingProfile):
        raise StoreError("profile must be a ShootingProfile")
    configuration = {
        "shooting": {name: getattr(profile, name) for name in _SHOOTING_FIELDS},
        "post_processing": {},
    }
    _validate_configuration(configuration)
    return _encode(configuration)


@dataclass(frozen=True)
class Bank:
    """Immutable bank; profile and snapshots are reconstructed rather than shared."""

    id: str
    name: str
    revision: int
    schema_version: int
    config_json: str
    created_at: str
    updated_at: str

    def __post_init__(self) -> None:
        _text(self.id, "id")
        _text(self.name, "name")
        _positive_integer(self.revision, "revision")
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise StoreError(f"unsupported bank schema {self.schema_version!r}")
        for field in ("created_at", "updated_at"):
            value = getattr(self, field)
            _text(value, field)
            try:
                stamp = datetime.fromisoformat(value)
            except ValueError as exc:
                raise StoreError(f"invalid {field}") from exc
            if stamp.tzinfo is None or stamp.utcoffset() is None:
                raise StoreError(f"{field} must include a timezone")
        if datetime.fromisoformat(self.updated_at) < datetime.fromisoformat(self.created_at):
            raise StoreError("updated_at must not precede created_at")
        object.__setattr__(self, "config_json", _encode(_decode(self.config_json)))

    @property
    def profile(self) -> ShootingProfile:
        return _validate_configuration(_decode(self.config_json))

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "revision": self.revision,
            "schema_version": self.schema_version,
            "configuration": _decode(self.config_json),
        }


@runtime_checkable
class BankRepository(Protocol):
    def list(self) -> list[Bank]: ...

    def get(self, id: str) -> Bank: ...

    def create(self, name: str, profile: ShootingProfile) -> Bank: ...

    def save(self, id: str, profile: ShootingProfile, expected_revision: int) -> Bank: ...


class SQLiteBankRepository:
    """SQLite banks in %LOCALAPPDATA%/ArducamCapture/banks.sqlite3 by default."""

    def __init__(self, path: str | os.PathLike | None = None):
        try:
            self.path = Path(path) if path is not None else Path(app_data_dir()) / "banks.sqlite3"
            if str(self.path) == ":memory:":
                raise StoreError("bank repositories require a persistent database path")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connection(initializing=True) as connection:
                connection.execute("BEGIN IMMEDIATE")
                version = connection.execute("PRAGMA user_version").fetchone()[0]
                if version == 0:
                    tables = connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    ).fetchall()
                    if tables:
                        raise StoreError("unversioned existing database is not a bank repository")
                    connection.execute(
                        "CREATE TABLE banks (id TEXT PRIMARY KEY, name TEXT NOT NULL, "
                        "revision INTEGER NOT NULL, schema_version INTEGER NOT NULL, "
                        "config_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
                    )
                    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                connection.execute(
                    "SELECT id, name, revision, schema_version, config_json, created_at, updated_at "
                    "FROM banks LIMIT 0"
                )
        except (OSError, TypeError, ValueError) as exc:
            raise StoreError(f"cannot open bank repository: {exc}") from exc

    @contextmanager
    def _connection(self, *, initializing: bool = False):
        connection = None
        try:
            connection = sqlite3.connect(str(self.path))
            connection.row_factory = sqlite3.Row
            with connection:
                version = connection.execute("PRAGMA user_version").fetchone()[0]
                if version != SCHEMA_VERSION and not (initializing and version == 0):
                    raise StoreError(f"unsupported repository schema {version}")
                yield connection
        except (sqlite3.Error, OSError, UnicodeError) as exc:
            raise StoreError(f"bank repository error: {exc}") from exc
        finally:
            if connection is not None:
                connection.close()

    def list(self) -> list[Bank]:
        with self._connection() as connection:
            return [
                self._bank(row)
                for row in connection.execute("SELECT * FROM banks ORDER BY created_at, id")
            ]

    def get(self, id: str) -> Bank:
        _text(id, "id")
        with self._connection() as connection:
            return self._get(connection, id)

    @staticmethod
    def _get(connection: sqlite3.Connection, id: str) -> Bank:
        row = connection.execute("SELECT * FROM banks WHERE id = ?", (id,)).fetchone()
        if row is None:
            raise StoreError(f"bank {id!r} not found")
        return SQLiteBankRepository._bank(row)

    @staticmethod
    def _bank(row: sqlite3.Row) -> Bank:
        try:
            return Bank(**dict(row))
        except TypeError as exc:
            raise StoreError(f"invalid bank record: {exc}") from exc

    def create(self, name: str, profile: ShootingProfile) -> Bank:
        _text(name, "name")
        config_json = _profile_json(profile)
        now = datetime.now(timezone.utc).isoformat()
        bank = Bank(str(uuid4()), name.strip(), 1, SCHEMA_VERSION, config_json, now, now)
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO banks (id, name, revision, schema_version, config_json, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (bank.id, bank.name, bank.revision, bank.schema_version, bank.config_json,
                 bank.created_at, bank.updated_at),
            )
        return bank

    def save(self, id: str, profile: ShootingProfile, expected_revision: int) -> Bank:
        _text(id, "id")
        _positive_integer(expected_revision, "expected_revision")
        config_json = _profile_json(profile)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._get(connection, id)
            if current.revision != expected_revision:
                raise StoreError(
                    f"revision conflict for bank {id!r}: expected {expected_revision}, "
                    f"found {current.revision}"
                )
            now = datetime.now(timezone.utc).isoformat()
            bank = Bank(id, current.name, current.revision + 1, SCHEMA_VERSION,
                        config_json, current.created_at, now)
            connection.execute(
                "UPDATE banks SET revision = ?, config_json = ?, updated_at = ? "
                "WHERE id = ? AND revision = ?",
                (bank.revision, bank.config_json, bank.updated_at, id, expected_revision),
            )
        return bank
