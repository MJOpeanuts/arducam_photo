"""Qt-free orchestration of modes, bank snapshots, working settings and proposals."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import os
from typing import Callable, Optional

from ..bundled_ccm import bundled_ccm_path
from ..ccm import load_ccm_file
from ..config import COLOR_720P, COLOR_4K, COLOR_12MP, NATIVE_108MP, PATHS, MODES, CaptureConfig
from ..errors import CaptureConfigError
from .controller import CameraController, CaptureJob, ModeError
from .paths import resolve_photo_dir
from .profile import (FOCUS_MAX, FOCUS_MIN, AppSettings, CameraId, ProfileStore, ShootingProfile)

START, PREVIEW_MODE, CAPTURE_MODE = "start", "preview", "capture"
PROCESSING_MODE = "processing"
SAVE, DISCARD, STAY = "save", "discard", "stay"

PATH_LABELS = {COLOR_720P: "720p couleur", COLOR_4K: "4K couleur",
               COLOR_12MP: "12 MP couleur", NATIVE_108MP: "108 MP RAW"}


class NeedsDecision(Exception):
    """Unsaved changes: call again with decision 'save', 'discard' or 'stay'."""


class ProfileMissing(Exception):
    """No valid saved profile: go through Preview / Settings first."""


class ProposalPending(Exception):
    """Validate or reject the proposal before editing or leaving Preview."""


class SessionModel:
    def __init__(self, profile_store: ProfileStore, settings_store, listener: Optional[Callable] = None,
                 *, controller: Optional[CameraController] = None, bank_repository=None,
                 **controller_kwargs):
        self.profiles, self.settings_store = profile_store, settings_store
        self.banks = bank_repository
        self.bank = None
        self.proposal = None
        self._adjust_before = None
        self.restoration_warnings = []
        self.capabilities = {}
        self._ui = listener or (lambda event, **kw: None)
        self.controller = controller or CameraController(self._on_event, **controller_kwargs)
        if controller is not None:
            controller._listener = self._on_event
        self.settings: AppSettings = settings_store.load()
        self.mode = START
        self.pending: Optional[str] = None  # "preview" | "capture" while a transition runs
        self.camera = CameraId.from_index(self.settings.camera_index)
        self.path = self.settings.path
        self.api = "msmf"
        self.saved: Optional[ShootingProfile] = None
        self.draft: Optional[ShootingProfile] = None
        self.last_photo: Optional[str] = None
        self.last_acquisition: Optional[str] = None
        self.processing_source: Optional[str] = None
        self._resources = contextlib.ExitStack()
        self._bundled: Optional[str] = None

    # ---- start screen ----
    def select_camera(self, index: int) -> None:
        self._require_mode(START)
        self.camera = CameraId.from_index(index)
        self._update_settings(camera_index=index)

    def select_path(self, path: str) -> None:
        self._require_mode(START, CAPTURE_MODE)
        if path not in PATHS:
            raise CaptureConfigError(f"parcours inconnu {path!r}")
        self.path = path
        self._update_settings(path=path)

    def set_photo_dir(self, directory: str) -> None:
        self._update_settings(photo_dir=os.path.abspath(directory))

    def set_ccm_path(self, path: str) -> None:
        self._update_settings(ccm_path=path)
        self.controller.verify_ccm(path)

    def set_start_fullscreen(self, enabled: bool) -> None:
        self._update_settings(start_fullscreen=bool(enabled))

    def effective_ccm_path(self) -> str:
        """Explicit user choice wins; otherwise the bundled resource (CaptureConfigError if absent)."""
        if self.settings.ccm_path:
            return self.settings.ccm_path
        if self._bundled is None:
            self._bundled = bundled_ccm_path(self._resources)
        return self._bundled

    def photo_dir(self) -> str:
        return resolve_photo_dir(self.settings.photo_dir)

    def _update_settings(self, **kw) -> None:
        self.settings = dataclasses.replace(self.settings, **kw)
        self.settings_store.save(self.settings)

    def _require_mode(self, *modes) -> None:
        if self.mode not in modes or self.pending or not self.controller.can_change_mode:
            raise ModeError(f"opération impossible en mode {self.mode} (transition: {self.pending})")

    # ---- profile ----
    def has_unsaved(self) -> bool:
        return self.mode == PREVIEW_MODE and self.draft is not None and not self.draft.same_settings(self.saved)

    def _editable(self) -> None:
        self._require_mode(PREVIEW_MODE)
        if self.proposal is not None:
            raise ProposalPending("Validez ou rejetez la proposition avant de continuer.")

    def set_focus(self, value: int) -> None:
        self._editable()
        if self.mode != PREVIEW_MODE or self.pending:
            raise ModeError("réglage du focus uniquement en Preview")
        if not FOCUS_MIN <= value <= FOCUS_MAX:
            raise CaptureConfigError(f"focus hors plage {FOCUS_MIN}..{FOCUS_MAX}")
        self.controller.set_focus(value)  # may raise ModeError: draft untouched then
        self.draft = dataclasses.replace(self.draft, focus_requested=int(value), focus_readback=None)

    def set_camera_control(self, name: str, value: Optional[float]) -> None:
        self._editable()
        names = {"exposure", "auto_exposure", "gain", "auto_wb", "wb_temperature",
                 "brightness", "contrast", "saturation"}
        if name not in names:
            raise CaptureConfigError(f"réglage caméra inconnu : {name}")
        if self.mode != PREVIEW_MODE or self.pending:
            raise ModeError("réglages caméra uniquement en Preview")
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        ):
            raise CaptureConfigError(f"{name} doit être une valeur numérique finie ou vide")
        updates = {name: value}
        if value is None and name == "auto_exposure":
            updates["auto_exposure_mode"] = None
        elif value is None and name == "auto_wb":
            updates["auto_wb_mode"] = None
        candidate = dataclasses.replace(self.draft, **updates)
        self.controller.apply_camera_settings(self._camera_control_values(candidate))
        self.draft = candidate

    def set_camera_mode(self, name: str, mode: Optional[str]) -> None:
        self._editable()
        if name not in ("auto_exposure_mode", "auto_wb_mode") or mode not in (None, "automatic", "manual"):
            raise CaptureConfigError("mode automatique caméra invalide")
        if self.mode != PREVIEW_MODE or self.pending:
            raise ModeError("réglages caméra uniquement en Preview")
        command = "auto_exposure" if name == "auto_exposure_mode" else "auto_wb"
        if mode is not None and getattr(self.draft, command) is None:
            raise CaptureConfigError(f"renseignez la valeur backend {command} avant de choisir un mode")
        candidate = dataclasses.replace(self.draft, **{name: mode})
        self.controller.apply_camera_settings(self._camera_control_values(candidate))
        self.draft = candidate

    @staticmethod
    def _camera_control_values(profile):
        return {
            "focus": profile.focus_requested, "exposure": profile.exposure,
            "auto_exposure": profile.auto_exposure,
            "auto_exposure_mode": profile.auto_exposure_mode,
            "gain": profile.gain, "auto_wb": profile.auto_wb,
            "auto_wb_mode": profile.auto_wb_mode,
            "wb_temperature": profile.wb_temperature, "brightness": profile.brightness,
            "contrast": profile.contrast, "saturation": profile.saturation,
        }

    def save_profile(self, *, confirm_replace: bool = False) -> ShootingProfile:
        self._editable()
        if self.banks is not None:
            if self.bank is None:
                raise ProfileMissing("Créez ou choisissez une bank pour sauvegarder.")
            return self.save_bank(self.bank.id, confirm_replace=confirm_replace).profile
        if self.draft is None:
            raise ProfileMissing("aucun réglage à enregistrer")
        problems = self.draft.problems()
        if problems:
            raise CaptureConfigError("profil incomplet: " + ", ".join(problems))
        self.profiles.save(self.draft)  # may raise; draft stays unsaved then
        self.saved = self.profiles.load(self.draft.camera_key) or self.draft
        return self.saved

    def _load_saved(self) -> None:
        if self.banks is not None:
            self.saved = self.bank.profile if self.bank is not None else None
        else:
            self.saved = self.profiles.load(self.camera.key)

    def list_banks(self):
        return self.banks.list() if self.banks is not None else []

    def create_bank(self, name: str):
        self._editable()
        if self.banks is None:
            raise ProfileMissing("Stockage des banks non configuré.")
        bank = self.banks.create(name, self.draft)
        self.bank, self.saved = bank, bank.profile
        self.draft = dataclasses.replace(self.saved)
        self._ui("bank_changed", bank=bank)
        return bank

    def save_bank(self, bank_id: str, *, confirm_replace: bool = False,
                  expected_revision: Optional[int] = None):
        self._editable()
        if self.banks is None:
            raise ProfileMissing("Stockage des banks non configuré.")
        if not confirm_replace:
            raise NeedsDecision("Confirmez le remplacement de la bank choisie.")
        if expected_revision is None:
            target = self.bank if self.bank is not None and self.bank.id == bank_id else self.banks.get(bank_id)
            expected_revision = target.revision
        bank = self.banks.save(bank_id, self.draft, expected_revision=expected_revision)
        self.bank, self.saved = bank, bank.profile
        self.draft = dataclasses.replace(self.saved)
        self._ui("bank_changed", bank=bank)
        return bank

    def _resolve_unsaved(self, decision, *, discard=True) -> bool:
        if self.proposal is not None:
            raise ProposalPending("Validez ou rejetez la proposition avant de continuer.")
        if not self.has_unsaved():
            return True
        if decision is None:
            raise NeedsDecision()
        if decision == STAY:
            return False
        if decision == SAVE:
            self.save_profile(confirm_replace=True)
        elif decision == DISCARD:
            if discard:
                self.draft = dataclasses.replace(self.saved) if self.saved else None
        else:
            raise ValueError(f"unknown decision {decision!r}")
        return True

    def load_bank(self, bank_id: str, decision=None) -> bool:
        self._editable()
        bank = self.banks.get(bank_id)
        self._compatible_bank_profile(bank)
        if not self._resolve_unsaved(decision, discard=False):
            return False
        bank = self.banks.get(bank_id)
        profile = self._compatible_bank_profile(bank)
        self.controller.apply_camera_settings(self._camera_control_values(profile))
        self.bank, self.saved = bank, profile
        self.draft = dataclasses.replace(profile)
        self.path = profile.path
        self._ui("bank_changed", bank=bank)
        return True

    def _compatible_bank_profile(self, bank):
        profile = bank.profile
        if profile.camera_key != self.camera.key or profile.camera_index != self.camera.index or profile.api != self.api:
            raise CaptureConfigError("Contexte caméra/backend différent ; choisissez ce contexte avant de charger la bank. "
                                     "L’index OpenCV ne garantit pas l’identité matérielle.")
        return profile

    def start_auto_adjustment(self, options) -> None:
        self._editable()
        before = self.draft
        self._adjust_before = before
        self.restoration_warnings = []
        try:
            self.controller.auto_adjust(self._camera_control_values(before), options)
        except Exception:
            self._adjust_before = None
            raise

    def validate_proposal(self) -> None:
        self._require_mode(PREVIEW_MODE)
        if self.proposal is None:
            raise ProposalPending("Aucune proposition à valider.")
        names = {"exposure", "gain", "auto_exposure", "auto_exposure_mode"}
        updates = {k: v for k, v in self.proposal.proposed.items() if k in names}
        self.draft = dataclasses.replace(self._adjust_before, **updates)
        self.proposal = None
        self._adjust_before = None
        self._ui("proposal_resolved")

    def reject_proposal(self) -> None:
        self._require_mode(PREVIEW_MODE)
        if self.proposal is None:
            raise ProposalPending("Aucune proposition à rejeter.")
        original = getattr(self.proposal, "original_settings", None)
        self.controller.reject_adjustment(original if original is not None else self.proposal.before)

    def resolve_preview_exit(self, decision=None) -> bool:
        self._require_mode(PREVIEW_MODE)
        return self._resolve_unsaved(decision)

    # ---- transitions ----
    def enter_preview(self) -> None:
        """Start (or Capture) -> Preview: opens the camera on the worker."""
        self._require_mode(START, CAPTURE_MODE, PROCESSING_MODE)
        leaving_processing = self.mode == PROCESSING_MODE
        self._load_saved()
        self.capabilities = {}
        base = self.saved or ShootingProfile()
        fps = base.fps
        if base.path != self.path or fps is None or any("cadence" in p for p in base.problems()):
            fps = MODES[self.path].fps
        self.draft = dataclasses.replace(
            base, camera_key=self.camera.key, camera_index=self.camera.index,
            camera_fragile=self.camera.fragile, path=self.path, api=self.api,
            fps=fps)
        self.pending = "preview"
        try:
            if leaving_processing:
                self.controller.clear_processed()
                self._ui("processing_cleared")
            self.controller.open_preview(self.camera.index, self.api, self.path)
        except Exception:
            self.pending = None
            raise

    def request_capture_mode(self, decision: Optional[str] = None) -> bool:
        """Go to Capture. Returns False if the user stays in Preview. May raise NeedsDecision."""
        if self.mode == CAPTURE_MODE and not self.pending:
            self._require_mode(CAPTURE_MODE)
            return True
        return self._request_mode(CAPTURE_MODE, decision)

    def request_processing_mode(self, decision: Optional[str] = None) -> bool:
        if self.mode == PROCESSING_MODE and not self.pending:
            self._require_mode(PROCESSING_MODE)
            return True
        return self._request_mode(PROCESSING_MODE, decision)

    def _request_mode(self, target: str, decision: Optional[str]) -> bool:
        self._require_mode(START, PREVIEW_MODE, CAPTURE_MODE, PROCESSING_MODE)
        if self.mode == PREVIEW_MODE:
            if not self._resolve_unsaved(decision):
                return False
            self.pending = target
            self.controller.close_preview()  # worker releases the camera, then preview_closed
        else:
            self._load_saved()
            if self.mode == PROCESSING_MODE:
                self.controller.clear_processed()
                self._ui("processing_cleared")
            self.mode = target
            self._ui("mode_changed", mode=self.mode)
        return True

    def trigger(self) -> None:
        if self.mode != CAPTURE_MODE or self.pending:
            raise ModeError("le déclencheur n'existe qu'en mode Capture")
        if self.proposal is not None:
            raise ProposalPending("Proposition en attente.")
        profile = self.bank.profile if self.banks is not None and self.bank is not None else self.saved
        if self.banks is not None and self.bank is None:
            raise ProfileMissing("Aucune bank sauvegardée sélectionnée.")
        if profile is None or not profile.is_valid:
            raise ProfileMissing("aucun profil valide: passez par Preview / Réglages")
        if profile.path != self.path:
            raise ProfileMissing("mode différent du profil enregistré : enregistrez explicitement un nouveau profil en Preview")
        native = profile.path == NATIVE_108MP
        try:
            ccm = self.effective_ccm_path() if native else None
        except CaptureConfigError:
            if self.controller.legacy_capture:
                raise
            ccm = None
        cfg = CaptureConfig(camera_index=profile.camera_index, api=profile.api, path=profile.path,
                            focus=profile.focus_requested,
                            exposure=profile.exposure, auto_exposure=profile.auto_exposure,
                            auto_exposure_mode=profile.auto_exposure_mode,
                            gain=profile.gain, auto_wb=profile.auto_wb,
                            auto_wb_mode=profile.auto_wb_mode,
                            wb_temperature=profile.wb_temperature, brightness=profile.brightness,
                            contrast=profile.contrast, saturation=profile.saturation,
                            ccm_path=ccm if native else None,
                            apply_ccm=native)
        cfg = dataclasses.replace(cfg, fps=profile.fps)
        cfg.validate()
        snapshot = json.dumps(self.bank.snapshot(), allow_nan=False) if self.bank is not None else None
        self.controller.capture(CaptureJob(cfg, self.photo_dir(), bank_snapshot=snapshot))

    def ccm_status(self) -> str:
        if self.path != NATIVE_108MP:
            return f"Correction couleur : non applicable ({PATH_LABELS.get(self.path, self.path)})"
        try:
            path = self.effective_ccm_path()
            load_ccm_file(path)
        except CaptureConfigError as e:
            return f"Correction couleur : ERREUR, rendu indisponible; original archivable ({e})"
        source = "fichier choisi" if self.settings.ccm_path else "ressource intégrée"
        return f"Correction couleur : activée ({source}: {path})"

    def load_processing(self, path: str) -> None:
        self._require_mode(PROCESSING_MODE)
        self.controller.load_acquisition(path)

    def process(self, recipe) -> None:
        self._require_mode(PROCESSING_MODE)
        if not self.processing_source:
            raise ModeError("ouvrez une acquisition d’abord")
        try:
            ccm = self.effective_ccm_path() if recipe.apply_ccm else None
        except CaptureConfigError:
            ccm = None
        self.controller.process(self.processing_source, recipe, ccm)

    def save_processed(self, options) -> None:
        self._require_mode(PROCESSING_MODE)
        self.controller.save_processed(options)

    def set_fps(self, value: float) -> None:
        self._editable()
        candidate = dataclasses.replace(self.draft, fps=value)
        if any("cadence" in problem for problem in candidate.problems()):
            raise CaptureConfigError("cadence photo hors plage")
        self.draft = candidate

    def shutdown(self, timeout: Optional[float] = None) -> bool:
        done = self.controller.shutdown(timeout)
        if done:
            self._resources.close()
            self._bundled = None
        return done

    # ---- controller events ----
    def _on_event(self, event: str, **kw) -> None:
        if event == "adjustment_proposed":
            self.proposal = kw["result"]
            self._ui(event, **kw)
        elif event == "adjustment_failed":
            self.restoration_warnings = kw.get("restoration_warnings", [])
            self._adjust_before = None
            self._ui(event, **kw)
        elif event == "adjustment_restored":
            self.restoration_warnings = kw.get("warnings", [])
            if self._adjust_before is not None:
                self.draft = self._adjust_before
            self.proposal = None
            self._adjust_before = None
            self._ui(event, **kw)
        elif event == "camera_capabilities":
            self.capabilities = kw["reports"]
            self._ui(event, **kw)
        elif event == "preview_opened":
            self.mode, self.pending = PREVIEW_MODE, None
            if self.draft:
                try:
                    self.controller.apply_camera_settings({
                        "focus": self.draft.focus_requested,
                        "exposure": self.draft.exposure,
                        "auto_exposure": self.draft.auto_exposure,
                        "auto_exposure_mode": self.draft.auto_exposure_mode,
                        "gain": self.draft.gain,
                        "auto_wb": self.draft.auto_wb,
                        "auto_wb_mode": self.draft.auto_wb_mode,
                        "wb_temperature": self.draft.wb_temperature,
                        "brightness": self.draft.brightness,
                        "contrast": self.draft.contrast,
                        "saturation": self.draft.saturation,
                    })
                except ModeError:
                    pass
            self._ui("mode_changed", mode=self.mode)
        elif event == "preview_failed":
            self.pending = None
            self._ui(event, **kw)
        elif event == "preview_closed":
            if self.pending in (CAPTURE_MODE, PROCESSING_MODE):
                self._ui("video_cleared")
                self._load_saved()
                self.mode, self.pending = self.pending, None
                self._ui("mode_changed", mode=self.mode)
        elif event == "focus_applied":
            d = self.draft
            if d is not None and d.focus_requested == kw["requested"]:
                self.draft = dataclasses.replace(d, focus_readback=kw["readback"], focus_verified=None)
            self._ui(event, **kw)
        elif event == "camera_settings_applied":
            self._ui(event, **kw)
        elif event == "camera_settings_failed":
            self._ui(event, **kw)
        elif event == "capture_archived":
            self.last_acquisition = kw["path"]
            self._ui(event, **kw)
        elif event == "processing_loaded":
            self.processing_source = kw["path"]
            self._ui(event, **kw)
        elif event == "capture_done":
            self.last_photo = kw["path"]
            self._ui(event, **kw)
        else:
            self._ui(event, **kw)
