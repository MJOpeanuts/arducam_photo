import dataclasses
import json
from types import SimpleNamespace

import pytest

from arducam_photo.app.controller import ModeError
from arducam_photo.app.model import (
    CAPTURE_MODE, DISCARD, PREVIEW_MODE, SAVE, STAY,
    NeedsDecision, ProposalPending, SessionModel,
)
from arducam_photo.app.profile import JsonSettingsStore, ShootingProfile, StoreError
from arducam_photo.banks import SQLiteBankRepository
from arducam_photo.config import COLOR_720P


class Controller:
    legacy_capture = False
    camera_fault = None

    def __init__(self):
        self.state = "preview"
        self.commands = []

    @property
    def can_change_mode(self):
        return self.state in ("preview", "idle")

    def apply_camera_settings(self, values):
        if self.state != "preview":
            raise ModeError("busy")
        self.commands.append(("settings", dict(values)))

    def set_focus(self, value):
        self.apply_camera_settings({"focus": value})

    def auto_adjust(self, before, options):
        self.state = "adjusting"
        self.commands.append(("adjust", before, options))

    def reject_adjustment(self, before):
        self.state = "restoring"
        self.commands.append(("restore", before))

    def close_preview(self):
        self.state = "idle"
        self._listener("preview_closed")

    def capture(self, job):
        self.commands.append(("capture", job))

    def shutdown(self, timeout=None):
        return True


@pytest.fixture
def session(tmp_path):
    repository = SQLiteBankRepository(str(tmp_path / "banks.sqlite3"))
    controller = Controller()
    model = SessionModel(
        None, JsonSettingsStore(str(tmp_path / "settings.json")),
        controller=controller, bank_repository=repository,
    )
    model.path = COLOR_720P
    model.mode = PREVIEW_MODE
    model.draft = ShootingProfile(
        camera_key="index:0", focus_requested=300, path=COLOR_720P,
        exposure=-5, auto_exposure=0.25, auto_exposure_mode="manual", gain=2,
    )
    model.create_bank("Marquage")
    yield model
    model.shutdown()


def proposal(model):
    before = model._camera_control_values(model.draft)
    model.start_auto_adjustment(object())
    model.controller.state = "preview"
    result = SimpleNamespace(before=before, proposed={**before, "exposure": -4, "gain": 3})
    model._on_event("adjustment_proposed", result=result)
    return result


def test_working_settings_and_validation_never_persist(session):
    original = session.banks.get(session.bank.id)
    session.set_focus(400)
    assert session.banks.get(original.id).profile.focus_requested == 300
    proposal(session)
    assert session.proposal is not None
    session.validate_proposal()
    assert session.draft.focus_requested == 400
    assert session.draft.exposure == -4
    assert session.has_unsaved()
    assert session.banks.get(original.id) == original
    with pytest.raises(NeedsDecision):
        session.save_bank(original.id)
    saved = session.save_bank(original.id, confirm_replace=True)
    assert saved.revision == original.revision + 1
    assert saved.profile.exposure == -4
    assert not session.has_unsaved()
    assert not any(command[0] == "capture" for command in session.controller.commands)


def test_proposal_requires_resolution_before_edit_switch_exit_or_save(session):
    other = session.banks.create("Autre", session.draft)
    proposal(session)
    for operation in (
        lambda: session.set_focus(100),
        lambda: session.set_camera_control("gain", 4),
        lambda: session.load_bank(other.id, DISCARD),
        lambda: session.request_capture_mode(DISCARD),
        lambda: session.resolve_preview_exit(DISCARD),
        lambda: session.save_bank(session.bank.id, confirm_replace=True),
    ):
        with pytest.raises(ProposalPending):
            operation()
    assert session.proposal is not None


def test_rejection_waits_for_restoration_and_reports_uncertainty(session):
    before = session.draft
    result = proposal(session)
    session.reject_proposal()
    assert session.proposal is result
    assert session.controller.commands[-1] == ("restore", result.before)
    with pytest.raises(ModeError):
        session.set_focus(100)
    session.controller.state = "preview"
    session._on_event("adjustment_restored", warnings=["gain non vérifié"])
    assert session.proposal is None
    assert session.draft == before
    assert session.restoration_warnings == ["gain non vérifié"]
    assert not session.has_unsaved()


def test_rejection_uses_observed_rollback_not_requested_consignes(session):
    result = proposal(session)
    result.original_settings = {"exposure": -7, "gain": 1, "auto_exposure": 0.75}
    session.reject_proposal()
    assert session.controller.commands[-1] == ("restore", result.original_settings)
    session.controller.state = "preview"
    session._on_event("adjustment_restored", warnings=[])
    assert session.draft.exposure == -5
    assert session.draft.auto_exposure == 0.25
    assert session.banks.get(session.bank.id).profile.exposure == -5


def test_legacy_save_entry_point_cannot_bypass_bank_replacement_confirmation(session):
    session.set_focus(400)
    before = session.bank
    with pytest.raises(NeedsDecision):
        session.save_profile()
    assert session.banks.get(before.id) == before
    session.save_profile(confirm_replace=True)
    assert session.bank.revision == before.revision + 1


def test_adjustment_failure_retains_working_copy_and_never_saves(session):
    original = session.bank
    before = session.draft
    session.start_auto_adjustment(object())
    with pytest.raises(ModeError):
        session.load_bank(original.id, DISCARD)
    with pytest.raises(ModeError):
        session.set_camera_control("gain", 100)
    assert session.draft == before
    session.controller.state = "preview"
    session._on_event("adjustment_failed", error=RuntimeError("cancelled"),
                      restoration_warnings=["exposition non vérifiée"])
    assert session.draft == before and session.proposal is None
    assert session.banks.get(original.id) == original


def test_bank_switch_unsaved_decisions_and_explicit_save(session):
    first = session.bank
    other = session.banks.create("Autre", dataclasses.replace(session.draft, focus_requested=200))
    session.set_focus(400)
    with pytest.raises(NeedsDecision):
        session.load_bank(other.id)
    assert session.load_bank(other.id, STAY) is False
    assert session.bank == first and session.draft.focus_requested == 400
    assert session.load_bank(other.id, SAVE)
    assert session.banks.get(first.id).profile.focus_requested == 400
    assert session.draft.focus_requested == 200
    session.set_focus(500)
    assert session.load_bank(first.id, DISCARD)
    assert session.banks.get(other.id).profile.focus_requested == 200
    assert session.draft.focus_requested == 400


def test_storage_failure_keeps_dirty_copy_and_saved_bank(session, monkeypatch):
    session.set_focus(400)
    before = session.bank

    def fail(*args, **kwargs):
        raise StoreError("disk unavailable")

    monkeypatch.setattr(session.banks, "save", fail)
    with pytest.raises(StoreError):
        session.save_bank(before.id, confirm_replace=True)
    assert session.bank == before
    assert session.draft.focus_requested == 400 and session.has_unsaved()


def test_failed_bank_application_does_not_lose_discardable_work(session, monkeypatch):
    other = session.banks.create("Autre", session.draft)
    session.set_focus(400)
    before = session.bank

    def fail(values):
        raise ModeError("busy")

    monkeypatch.setattr(session.controller, "apply_camera_settings", fail)
    with pytest.raises(ModeError):
        session.load_bank(other.id, DISCARD)
    assert session.bank == before
    assert session.draft.focus_requested == 400 and session.has_unsaved()


def test_photo_snapshot_is_saved_bank_not_draft_or_future_revision(session):
    first = session.bank
    session.set_focus(400)
    session.request_capture_mode(DISCARD)
    assert session.mode == CAPTURE_MODE
    session.trigger()
    job = session.controller.commands[-1][1]
    snapshot = json.loads(job.bank_snapshot)
    session.banks.save(first.id, dataclasses.replace(first.profile, focus_requested=900),
                       expected_revision=first.revision)
    assert job.config.focus == 300
    assert json.loads(job.bank_snapshot) == snapshot
    assert snapshot["id"] == first.id and snapshot["revision"] == first.revision
    assert snapshot["configuration"]["shooting"]["focus_requested"] == 300
    assert snapshot["configuration"]["post_processing"] == {}
    assert not any(command[0] == "adjust" for command in session.controller.commands)


def test_incompatible_camera_bank_cannot_discard_current_work(session):
    other = session.banks.create(
        "Caméra différente",
        dataclasses.replace(session.draft, camera_key="index:1", camera_index=1),
    )
    session.set_focus(400)
    before = session.bank
    with pytest.raises(Exception, match="Contexte caméra"):
        session.load_bank(other.id, DISCARD)
    assert session.bank == before and session.draft.focus_requested == 400
