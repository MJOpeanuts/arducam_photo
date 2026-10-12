from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
import json
import sqlite3
from threading import Barrier

import pytest

from arducam_photo.app.profile import ShootingProfile, StoreError
from arducam_photo.banks import Bank, BankRepository, SQLiteBankRepository


@pytest.fixture
def profile():
    return ShootingProfile(
        camera_key="index:2", camera_index=2, camera_fragile=True,
        focus_requested=310, exposure=-6.5, auto_exposure=0.25,
        auto_exposure_mode="manual", gain=4, auto_wb=0,
        auto_wb_mode="manual", wb_temperature=5000, brightness=1,
        contrast=2, saturation=3, fps=7,
        focus_readback=309, focus_verified=True, saved_at="old observation",
    )


@pytest.fixture
def repository(tmp_path):
    return SQLiteBankRepository(tmp_path / "nested" / "banks.sqlite3")


def update_row(repository, bank, column, value):
    with sqlite3.connect(repository.path) as connection:
        connection.execute(f"UPDATE banks SET {column} = ? WHERE id = ?", (value, bank.id))


def test_default_location_and_no_legacy_persistence(tmp_path, monkeypatch, profile):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    repository = SQLiteBankRepository()
    assert repository.path == tmp_path / "ArducamCapture" / "banks.sqlite3"
    assert isinstance(repository, BankRepository)
    repository.create("Default", profile)
    assert {p.name for p in repository.path.parent.iterdir()} == {"banks.sqlite3"}


def test_persistence_full_configuration_and_revisions(repository, profile):
    assert repository.list() == []
    original = repository.create("  Studio é  ", profile)
    assert original.name == "Studio é"
    assert original.revision == original.schema_version == 1
    assert isinstance(original.config_json, str)
    reopened = SQLiteBankRepository(repository.path)
    assert reopened.get(original.id) == original
    assert reopened.list() == [original]
    shooting = json.loads(original.config_json)["shooting"]
    assert shooting == {
        key: value for key, value in profile.to_dict().items()
        if key not in {"focus_readback", "focus_verified", "saved_at"}
    }
    assert json.loads(original.config_json)["post_processing"] == {}
    assert original.profile == replace(profile, focus_readback=None, focus_verified=None, saved_at=None)
    changed = replace(profile, focus_requested=0, camera_fragile=False)
    saved = reopened.save(original.id, changed, expected_revision=1)
    assert saved.id == original.id and saved.name == original.name
    assert saved.revision == 2 and saved.created_at == original.created_at
    assert saved.updated_at >= original.updated_at
    assert saved.profile.focus_requested == 0
    assert saved.profile.camera_fragile is False
    assert repository.get(saved.id) == saved
    assert original.profile.focus_requested == 310


def test_bank_and_snapshots_are_isolated(repository, profile):
    bank = repository.create("One", profile)
    other = repository.create("One", profile)
    assert bank.id != other.id
    assert len(repository.list()) == 2
    with pytest.raises(FrozenInstanceError):
        bank.name = "Changed"
    first = bank.snapshot()
    assert set(first) == {"id", "name", "revision", "schema_version", "configuration"}
    assert json.loads(json.dumps(first)) == first
    first["configuration"]["shooting"]["focus_requested"] = 999
    first["configuration"]["post_processing"]["unexpected"] = True
    first["revision"] = 999
    assert bank.snapshot()["configuration"]["post_processing"] == {}
    assert bank.snapshot()["revision"] == 1
    assert bank.profile.focus_requested == 310
    assert bank.profile is not bank.profile
    assert repository.get(bank.id).snapshot() == bank.snapshot()
    assert other.profile.focus_requested == 310


def test_stale_save_never_overwrites(repository, profile):
    bank = repository.create("One", profile)
    second = SQLiteBankRepository(repository.path)
    updated = repository.save(bank.id, replace(profile, focus_requested=500), 1)
    with pytest.raises(StoreError, match="revision conflict"):
        second.save(bank.id, replace(profile, focus_requested=600), 1)
    assert second.get(bank.id) == updated
    assert second.save(bank.id, profile, 2).revision == 3


def test_concurrent_writers_allow_exactly_one_revision(repository, profile):
    bank = repository.create("One", profile)
    repositories = [SQLiteBankRepository(repository.path) for _ in range(2)]
    barrier = Barrier(2)

    def write(index):
        barrier.wait()
        try:
            return repositories[index].save(
                bank.id, replace(profile, focus_requested=400 + index), 1
            )
        except StoreError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(write, range(2)))
    winners = [result for result in results if isinstance(result, Bank)]
    assert len(winners) == 1
    assert "revision conflict" in str(next(result for result in results if isinstance(result, StoreError)))
    assert repository.get(bank.id) == winners[0]


@pytest.mark.parametrize("field,value", [
    ("camera_key", ""), ("camera_key", "  "), ("camera_key", 42),
    ("camera_key", []), ("camera_index", True), ("camera_index", 1.0),
    ("camera_index", -1), ("camera_fragile", "false"), ("camera_fragile", 1),
    ("path", "unsupported"), ("path", []), ("api", {}), ("api", "unknown"),
    ("focus_requested", None), ("focus_requested", True),
    ("focus_requested", 1.5), ("focus_requested", -1), ("focus_requested", 1024),
    ("exposure", "1"), ("exposure", True), ("gain", float("nan")),
    ("auto_wb", float("inf")), ("brightness", []), ("contrast", {}),
    ("saturation", float("-inf")), ("wb_temperature", "5000"),
    ("auto_exposure_mode", []), ("auto_exposure_mode", "manual-no"),
    ("auto_wb_mode", True), ("fps", True), ("fps", 0), ("fps", 121),
    ("fps", float("nan")), ("fps", "7"),
])
def test_invalid_profiles_rejected_on_create_and_save(repository, profile, field, value):
    bank = repository.create("One", profile)
    invalid = replace(profile, **{field: value})
    with pytest.raises(StoreError):
        repository.create("Invalid", invalid)
    with pytest.raises(StoreError):
        repository.save(bank.id, invalid, 1)
    assert repository.list() == [bank]


@pytest.mark.parametrize("field", ["auto_exposure", "auto_wb"])
def test_backend_mode_requires_value(repository, profile, field):
    with pytest.raises(StoreError):
        repository.create("Invalid", replace(profile, **{field: None}))


@pytest.mark.parametrize("invalid", [None, {}, "profile", 1])
def test_nonprofile_arguments(repository, profile, invalid):
    bank = repository.create("One", profile)
    with pytest.raises(StoreError):
        repository.create("Invalid", invalid)
    with pytest.raises(StoreError):
        repository.save(bank.id, invalid, 1)
    assert repository.get(bank.id) == bank


@pytest.mark.parametrize("invalid", [None, "", " ", True, 1, [], {}])
def test_invalid_names_and_identifiers(repository, profile, invalid):
    with pytest.raises(StoreError):
        repository.create(invalid, profile)
    with pytest.raises(StoreError):
        repository.get(invalid)
    with pytest.raises(StoreError):
        repository.save(invalid, profile, 1)
    assert repository.list() == []


@pytest.mark.parametrize("revision", [True, False, 1.0, "1", None, 0, -1, [], {}])
def test_malformed_revisions_do_not_match(repository, profile, revision):
    bank = repository.create("One", profile)
    with pytest.raises(StoreError):
        repository.save(bank.id, profile, revision)
    assert repository.get(bank.id) == bank


def test_missing_bank_is_store_error(repository, profile):
    with pytest.raises(StoreError, match="not found"):
        repository.get("missing")
    with pytest.raises(StoreError, match="not found"):
        repository.save("missing", profile, 1)


@pytest.mark.parametrize("mutation", [
    lambda data: None,
    lambda data: [],
    lambda data: {"shooting": data["shooting"]},
    lambda data: {**data, "extra": 1},
    lambda data: {**data, "shooting": []},
    lambda data: {**data, "post_processing": []},
    lambda data: {**data, "post_processing": {"unsupported": 1}},
    lambda data: {**data, "shooting": {**data["shooting"], "unknown": 1}},
    lambda data: {**data, "shooting": {**data["shooting"], "saved_at": None}},
    lambda data: {**data, "shooting": {**data["shooting"], "camera_index": True}},
    lambda data: {**data, "shooting": {**data["shooting"], "camera_fragile": "false"}},
    lambda data: {**data, "shooting": {**data["shooting"], "focus_requested": "310"}},
    lambda data: {**data, "shooting": {**data["shooting"], "gain": float("nan")}},
    lambda data: {**data, "shooting": {
        key: value for key, value in data["shooting"].items() if key != "fps"
    }},
])
def test_corrupt_configuration_rejected_on_reads_and_save(repository, profile, mutation):
    bank = repository.create("One", profile)
    corrupt = json.dumps(mutation(json.loads(bank.config_json)))
    update_row(repository, bank, "config_json", corrupt)
    for read in (repository.list, lambda: repository.get(bank.id),
                 lambda: repository.save(bank.id, profile, 1)):
        with pytest.raises(StoreError):
            read()
    with sqlite3.connect(repository.path) as connection:
        assert connection.execute("SELECT config_json FROM banks").fetchone()[0] == corrupt


@pytest.mark.parametrize("corrupt", ["{", '{"shooting":{},"shooting":{},"post_processing":{}}',
                                   '{"shooting":NaN,"post_processing":{}}', b"not text"])
def test_invalid_json_is_store_error(repository, profile, corrupt):
    bank = repository.create("One", profile)
    update_row(repository, bank, "config_json", corrupt)
    with pytest.raises(StoreError):
        repository.get(bank.id)
    with pytest.raises(StoreError):
        repository.list()


@pytest.mark.parametrize("field,value", [
    ("revision", 0), ("revision", 1.5), ("schema_version", 2),
    ("schema_version", 1.5), ("name", ""), ("name", b"binary"),
    ("created_at", "yesterday"), ("updated_at", "2020-01-01"),
    ("updated_at", "2000-01-01T00:00:00+00:00"),
])
def test_corrupt_bank_metadata_is_rejected(repository, profile, field, value):
    bank = repository.create("One", profile)
    update_row(repository, bank, field, value)
    with pytest.raises(StoreError):
        repository.get(bank.id)
    with pytest.raises(StoreError):
        repository.list()
    with pytest.raises(StoreError):
        repository.save(bank.id, profile, 1)


@pytest.mark.parametrize("field,value", [
    ("revision", True), ("schema_version", True), ("schema_version", 1.0),
    ("config_json", {}), ("id", []), ("created_at", None),
])
def test_bank_constructor_validates_types(repository, profile, field, value):
    bank = repository.create("One", profile)
    with pytest.raises(StoreError):
        replace(bank, **{field: value})


def test_unsupported_repository_version_is_not_modified(repository, profile):
    bank = repository.create("One", profile)
    with sqlite3.connect(repository.path) as connection:
        connection.execute("PRAGMA user_version = 99")
    with pytest.raises(StoreError, match="unsupported repository schema"):
        SQLiteBankRepository(repository.path)
    with pytest.raises(StoreError):
        repository.get(bank.id)
    with pytest.raises(StoreError):
        repository.create("One", profile)
    with sqlite3.connect(repository.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 99
        assert connection.execute("SELECT revision FROM banks").fetchone()[0] == 1


def test_corrupt_database_and_foreign_database_preserved(tmp_path):
    path = tmp_path / "banks.sqlite3"
    path.write_bytes(b"not a SQLite database")
    with pytest.raises(StoreError):
        SQLiteBankRepository(path)
    assert path.read_bytes() == b"not a SQLite database"
    foreign = tmp_path / "foreign.sqlite3"
    with sqlite3.connect(foreign) as connection:
        connection.execute("CREATE TABLE unrelated (id INTEGER)")
    with pytest.raises(StoreError, match="unversioned"):
        SQLiteBankRepository(foreign)
    with sqlite3.connect(foreign) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        assert connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == [
            ("unrelated",)
        ]


def test_filesystem_failures_are_store_errors(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("keep")
    with pytest.raises(StoreError):
        SQLiteBankRepository(blocker / "banks.sqlite3")
    with pytest.raises(StoreError):
        SQLiteBankRepository(tmp_path)
    with pytest.raises(StoreError):
        SQLiteBankRepository(":memory:")
    assert blocker.read_text() == "keep"


def test_sqlite_errors_are_wrapped(repository, profile, monkeypatch):
    bank = repository.create("One", profile)

    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("disk unavailable")

    monkeypatch.setattr(sqlite3, "connect", fail)
    for operation in (repository.list, lambda: repository.get(bank.id),
                      lambda: repository.create("Other", profile),
                      lambda: repository.save(bank.id, profile, 1)):
        with pytest.raises(StoreError, match="disk unavailable"):
            operation()
