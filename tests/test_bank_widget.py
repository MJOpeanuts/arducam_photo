import dataclasses
import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QDialog, QDoubleSpinBox, QInputDialog, QLineEdit, QMessageBox

from arducam_photo.app.model import PREVIEW_MODE, STAY, SessionModel
from arducam_photo.app.profile import JsonSettingsStore, ShootingProfile
from arducam_photo.app.widget import CameraWidget
from arducam_photo.banks import SQLiteBankRepository
from arducam_photo.config import COLOR_720P, SettingReport


class Controller:
    camera_fault = None

    def __init__(self):
        self.state = "preview"
        self.calls = []

    @property
    def can_change_mode(self):
        return self.state in ("preview", "idle")

    def apply_camera_settings(self, values):
        self.calls.append(("settings", values))

    def auto_adjust(self, before, options):
        self.calls.append(("adjust", before, options))
        self.state = "adjusting"

    def reject_adjustment(self, before):
        self.calls.append(("reject", before))
        self.state = "restoring"

    def cancel_adjustment(self):
        self.calls.append(("cancel",))

    def probe_controls(self):
        self.calls.append(("probe",))
        self.state = "probing_controls"

    def fps(self):
        return 0, 0

    def shutdown(self, timeout=None):
        return True


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def widget(qapp, tmp_path):
    model = SessionModel(None, JsonSettingsStore(str(tmp_path / "settings.json")),
                         controller=Controller(),
                         bank_repository=SQLiteBankRepository(str(tmp_path / "banks.sqlite3")))
    model.mode = PREVIEW_MODE
    model.path = COLOR_720P
    model.draft = ShootingProfile(camera_key=model.camera.key, camera_index=model.camera.index,
                                  api=model.api, path=model.path, focus_requested=300,
                                  exposure=-5, gain=2, auto_exposure=0.25,
                                  auto_exposure_mode="manual")
    model.create_bank("Initiale")
    widget = CameraWidget(model)
    widget._show_page(PREVIEW_MODE)
    yield widget
    widget._fps_timer.stop()
    widget.deleteLater()
    model.shutdown()
    qapp.processEvents()


def report(value=1, accepted=True):
    return SettingReport(requested=value, attempted=True, accepted=accepted, readback=value)


def enable_controls(widget):
    widget.model.capabilities = {name: report() for name in
                                ("focus", "auto_exposure", "exposure", "gain", "auto_wb", "brightness")}
    widget._on_event("camera_capabilities", {"reports": widget.model.capabilities})


def propose(widget):
    model = widget.model
    model._adjust_before = model.draft
    result = SimpleNamespace(before=model._camera_control_values(model.draft),
                             proposed={"exposure": -4, "gain": 3},
                             metrics={"brightness": 0.5, "saturated_fraction": 0.01,
                                      "local_contrast": 0.1}, iterations=5, elapsed=1.2)
    model.proposal = result
    widget._on_event("adjustment_proposed", {"result": result})
    return result


def test_selecting_bank_never_loads_until_explicit_click(widget):
    model = widget.model
    other = model.banks.create("Autre", dataclasses.replace(model.draft, focus_requested=400))
    widget._refresh_banks(other.id)
    assert model.draft.focus_requested == 300
    widget._load_bank()
    assert model.draft.focus_requested == 400


def test_create_refreshes_selected_bank_and_saves_current_draft(widget, monkeypatch):
    widget.model.draft = dataclasses.replace(widget.model.draft, focus_requested=401)
    monkeypatch.setattr(QInputDialog, "getText", lambda *a: ("Nouvelle", True))
    assert widget._create_bank()
    assert widget.bank_combo.currentData() == widget.model.bank.id
    assert widget.model.bank.name == "Nouvelle"
    assert widget.model.bank.profile.focus_requested == 401
    assert not widget.model.has_unsaved()


def test_save_replaces_only_explicit_target_with_revision_confirmation(widget, monkeypatch):
    model = widget.model
    original = model.bank
    target = model.banks.create("Cible", model.draft)
    widget._refresh_banks(target.id)
    model.draft = dataclasses.replace(model.draft, focus_requested=402)
    prompts = []
    def confirm(*args):
        prompts.append(args[2])
        return QMessageBox.StandardButton.Yes
    monkeypatch.setattr(QMessageBox, "question", confirm)
    widget._save()
    assert "Cible" in prompts[0] and "révision 1" in prompts[0]
    assert model.banks.get(target.id).revision == 2
    assert model.banks.get(target.id).profile.focus_requested == 402
    assert model.banks.get(original.id).revision == 1


def test_cancel_replacement_leaves_bank_and_draft_unchanged(widget, monkeypatch):
    model = widget.model
    widget._refresh_banks(model.bank.id)
    model.draft = dataclasses.replace(model.draft, focus_requested=402)
    monkeypatch.setattr(QMessageBox, "question", lambda *a: QMessageBox.StandardButton.No)
    assert not widget._save_bank_target()
    assert model.bank.revision == 1 and model.has_unsaved()


def test_confirmed_revision_conflict_preserves_concurrent_bank_and_draft(widget, monkeypatch):
    model = widget.model
    target = model.banks.create("Cible concurrente", model.draft)
    widget._refresh_banks(target.id)
    model.draft = dataclasses.replace(model.draft, focus_requested=402)
    draft = model.draft
    saved = model.saved
    other_connection = SQLiteBankRepository(model.banks.path)
    def confirm(*args):
        assert "révision 1" in args[2]
        other_connection.save(target.id, dataclasses.replace(target.profile, focus_requested=450),
                              expected_revision=target.revision)
        return QMessageBox.StandardButton.Yes
    monkeypatch.setattr(QMessageBox, "question", confirm)
    assert not widget._save_bank_target()
    current = other_connection.get(target.id)
    assert current.revision == 2 and current.profile.focus_requested == 450
    assert model.draft == draft and model.saved == saved and model.has_unsaved()
    assert widget.preview_msg.text().startswith("Impossible d’enregistrer la bank.")


def test_unsaved_save_cannot_leave_after_target_save_cancelled(widget, monkeypatch):
    model = widget.model
    model.draft = dataclasses.replace(model.draft, focus_requested=402)
    monkeypatch.setattr(QMessageBox, "exec", lambda self: None)
    monkeypatch.setattr(QMessageBox, "clickedButton",
                        lambda self: next(b for b in self.buttons() if b.text() == "Enregistrer et continuer"))
    monkeypatch.setattr(widget, "_save_bank_target", lambda: False)
    assert widget._ask_unsaved() == STAY
    assert model.has_unsaved()


def test_capabilities_require_explicit_probe_and_finite_accepted_readback(widget):
    assert not widget.focus_spin.isEnabled()
    assert not widget.camera_settings_inputs["exposure"].isEnabled()
    assert widget.btn_probe_controls.isEnabled()
    assert widget.model.controller.calls == []
    widget._probe_controls()
    assert widget.model.controller.calls == [("probe",)]
    widget.model.controller.state = "preview"
    widget.model.capabilities = {"focus": report(float("nan")), "gain": report(2, False),
                                "auto_exposure": report(0.25), "exposure": report(-5)}
    widget._on_event("camera_capabilities", {"reports": widget.model.capabilities})
    assert not widget.focus_spin.isEnabled() and not widget.camera_settings_inputs["gain"].isEnabled()
    assert widget.camera_settings_inputs["exposure"].isEnabled()
    assert widget.camera_mode_inputs["auto_exposure_mode"].isEnabled()
    assert not widget.camera_mode_inputs["auto_wb_mode"].isEnabled()
    assert "Demandé" in widget.focus_spin.toolTip() and "indisponible" in widget.focus_spin.toolTip()
    assert "relu" in widget.preview_details.text()


@pytest.mark.parametrize("state", ["adjusting", "restoring", "probing_controls"])
def test_busy_adjustment_locks_edits_banks_navigation_and_close(widget, monkeypatch, state):
    enable_controls(widget)
    widget.model.controller.state = state
    widget._refresh_preview_state()
    for control in (widget.focus_spin, widget.fps_spin, widget.btn_save, widget.bank_combo,
                    widget.btn_create_bank, widget.btn_load_bank, widget.tabs, widget.btn_trigger,
                    widget.camera_mode_inputs["auto_exposure_mode"]):
        assert not control.isEnabled()
    assert widget.btn_cancel_adjustment.isEnabled() == (state == "adjusting")
    monkeypatch.setattr(widget, "_info", lambda text: None)
    assert not widget.can_close()


def test_restore_warning_never_claims_settings_restored(widget):
    widget._on_event("adjustment_restored", {"warnings": ["gain: restoration impossible"]})
    assert "Restauration non vérifiée/incomplète" in widget.adjustment_summary.text()
    assert "Réglages restaurés" not in widget.adjustment_summary.text()
    assert "restoration impossible" in widget.adjustment_summary.text()


def test_worker_restoration_started_disables_cancel_and_shows_wait_status(widget):
    widget.model.controller.state = "restoring"
    widget._on_event("adjustment_restoration_started", {})
    assert "Restauration des réglages en cours" in widget.adjustment_summary.text()
    assert not widget.btn_cancel_adjustment.isEnabled()
    assert not widget.tabs.isEnabled()


@pytest.mark.parametrize("warnings", [[], ["exposure: restoration failed"]])
def test_adjustment_failure_replaces_running_summary_with_restoration_status(widget, warnings):
    widget.adjustment_summary.setText("Auto-ajustement en cours.")
    widget._on_event("adjustment_failed", {"error": RuntimeError("interrupted"),
                                         "restoration_warnings": warnings})
    text = widget.adjustment_summary.text()
    assert "interrompu" in text and "en cours" not in text
    assert ("Restauration non vérifiée/incomplète" in text) == bool(warnings)


def test_pending_proposal_locks_settings_but_validation_only_updates_draft(widget):
    enable_controls(widget)
    before = widget.model.bank
    propose(widget)
    assert not widget.btn_save.isEnabled() and not widget.tabs.isEnabled()
    assert not widget.btn_probe_controls.isEnabled()
    assert widget.btn_accept_proposal.isEnabled() and widget.btn_reject_proposal.isEnabled()
    assert "brightness" in widget.adjustment_summary.text()
    widget._accept_proposal()
    assert widget.model.draft.exposure == -4 and widget.model.has_unsaved()
    assert widget.model.banks.get(before.id) == before
    assert widget.btn_save.isEnabled()


def test_rejection_disables_cancel_even_if_worker_uses_adjusting_state(widget, monkeypatch):
    propose(widget)
    monkeypatch.setattr(widget.model.controller, "reject_adjustment",
                        lambda before: setattr(widget.model.controller, "state", "adjusting"))
    widget._reject_proposal()
    assert not widget.btn_cancel_adjustment.isEnabled()
    assert not widget.btn_probe_controls.isEnabled()


@pytest.mark.parametrize("converged,reason,label", [
    (True, "converged", "Critères atteints"),
    (False, "time_budget", "Candidat seulement"),
])
def test_proposal_distinguishes_candidate_from_convergence(widget, converged, reason, label):
    result = propose(widget)
    result.converged = converged
    result.stop_reason = reason
    result.metrics.update(converged=converged, stop_reason=reason)
    widget._on_event("adjustment_proposed", {"result": result})
    assert label in widget.adjustment_summary.text()
    assert ("budget temps atteint" in widget.adjustment_summary.text()) == (reason == "time_budget")


def test_rejection_stays_until_async_restore_and_then_can_retry(widget, monkeypatch):
    before = widget.model.draft
    propose(widget)
    monkeypatch.setattr(QMessageBox, "exec", lambda self: None)
    monkeypatch.setattr(QMessageBox, "clickedButton",
                        lambda self: next(b for b in self.buttons() if b.text() == "Rejeter"))
    messages = []
    monkeypatch.setattr(widget, "_info", messages.append)
    assert not widget._resolve_proposal_prompt()
    assert widget.model.controller.state == "restoring" and widget.model.proposal is not None
    assert messages and not widget.btn_reject_proposal.isEnabled()
    widget.model.controller.state = "preview"
    widget.model._on_event("adjustment_restored", warnings=[])
    widget._on_event("adjustment_restored", {"warnings": []})
    assert widget.model.draft == before and widget.model.proposal is None
    assert widget.can_close()


def test_adjustment_requires_fixed_focus_and_explicit_manual_backend_mode(widget, monkeypatch):
    messages = []
    monkeypatch.setattr(widget, "_info", messages.append)
    widget.model.draft = dataclasses.replace(widget.model.draft, auto_exposure_mode=None)
    widget._auto_adjust()
    assert messages and not widget.model.controller.calls


def test_operator_options_normalized_roi_and_backend_bounds(widget, monkeypatch):
    def accept(dialog):
        for name, value in (("exposure_minimum", "-8"), ("exposure_maximum", "-2"),
                            ("gain_minimum", "0"), ("gain_maximum", "12"),
                            ("exposure_step", "0.5"), ("gain_step", "2")):
            dialog.findChild(QLineEdit, name).setText(value)
        dialog.findChild(QDoubleSpinBox, "roi_x").setValue(0.2)
        dialog.findChild(QDoubleSpinBox, "roi_largeur").setValue(0.6)
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(QDialog, "exec", accept)
    options = widget._adjustment_options()
    assert options.roi == (0.2, 0, 0.6, 1)
    assert options.exposure_bounds == (-8, -2) and options.gain_bounds == (0, 12)
    assert options.exposure_step == 0.5 and options.gain_step == 2


def test_startup_bank_list_failure_is_visible_and_not_recreated(widget, monkeypatch):
    def fail():
        raise RuntimeError("database malformed")
    errors = []
    monkeypatch.setattr(widget.model, "list_banks", fail)
    monkeypatch.setattr(widget, "_error", lambda summary, error: errors.append((summary, str(error))))
    widget._refresh_banks()
    assert errors and "database malformed" in errors[0][1]
    assert widget.model.banks.get(widget.model.bank.id).revision == 1
