"""Qt-free orchestration of the Start / Preview / Capture modes, profile and settings."""

from __future__ import annotations

import contextlib
import dataclasses
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


class SessionModel:
    def __init__(self, profile_store: ProfileStore, settings_store, listener: Optional[Callable] = None,
                 *, controller: Optional[CameraController] = None, **controller_kwargs):
        self.profiles, self.settings_store = profile_store, settings_store
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

    def set_focus(self, value: int) -> None:
        if self.mode != PREVIEW_MODE or self.pending:
            raise ModeError("réglage du focus uniquement en Preview")
        if not FOCUS_MIN <= value <= FOCUS_MAX:
            raise CaptureConfigError(f"focus hors plage {FOCUS_MIN}..{FOCUS_MAX}")
        self.controller.set_focus(value)  # may raise ModeError: draft untouched then
        self.draft = dataclasses.replace(self.draft, focus_requested=int(value), focus_readback=None)

    def save_profile(self) -> ShootingProfile:
        if self.draft is None:
            raise ProfileMissing("aucun réglage à enregistrer")
        problems = self.draft.problems()
        if problems:
            raise CaptureConfigError("profil incomplet: " + ", ".join(problems))
        self.profiles.save(self.draft)  # may raise; draft stays unsaved then
        self.saved = self.profiles.load(self.draft.camera_key) or self.draft
        return self.saved

    def _load_saved(self) -> None:
        self.saved = self.profiles.load(self.camera.key)

    # ---- transitions ----
    def enter_preview(self) -> None:
        """Start (or Capture) -> Preview: opens the camera on the worker."""
        self._require_mode(START, CAPTURE_MODE, PROCESSING_MODE)
        self._load_saved()
        base = self.saved or ShootingProfile()
        self.draft = dataclasses.replace(
            base, camera_key=self.camera.key, camera_index=self.camera.index,
            camera_fragile=self.camera.fragile, path=self.path, api=self.api,
            fps=base.fps if base.path == self.path and base.fps is not None else MODES[self.path].fps)
        self.pending = "preview"
        try:
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
            if self.has_unsaved():
                if decision is None:
                    raise NeedsDecision()
                if decision == STAY:
                    return False
                if decision == SAVE:
                    self.save_profile()  # on failure we stay in Preview
                elif decision == DISCARD:
                    self.draft = self.saved
                else:
                    raise ValueError(f"unknown decision {decision!r}")
            self.pending = target
            self.controller.close_preview()  # worker releases the camera, then preview_closed
        else:
            if self.mode == PROCESSING_MODE:
                self.controller.clear_processed()
                self._ui("processing_cleared")
            self._load_saved()
            self.mode = target
            self._ui("mode_changed", mode=self.mode)
        return True

    def trigger(self) -> None:
        if self.mode != CAPTURE_MODE or self.pending:
            raise ModeError("le déclencheur n'existe qu'en mode Capture")
        profile = self.saved
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
            ccm = "ressource-CCM-indisponible.json"
        cfg = CaptureConfig(camera_index=profile.camera_index, api=profile.api, path=profile.path,
                            focus=profile.focus_requested, ccm_path=ccm if native else None,
                            apply_ccm=native)
        cfg = dataclasses.replace(cfg, fps=profile.fps)
        cfg.validate()
        self.controller.capture(CaptureJob(cfg, self.photo_dir()))  # frozen copy; may raise busy

    def ccm_status(self) -> str:
        if (self.saved.path if self.saved else self.path) != NATIVE_108MP:
            return "Correction couleur : non applicable (720p)"
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
            ccm = "ressource-CCM-indisponible.json"
        self.controller.process(self.processing_source, recipe, ccm)

    def save_processed(self, options) -> None:
        self._require_mode(PROCESSING_MODE)
        self.controller.save_processed(options)

    def set_fps(self, value: float) -> None:
        self._require_mode(PREVIEW_MODE)
        self.draft = dataclasses.replace(self.draft, fps=float(value))

    def shutdown(self, timeout: Optional[float] = None) -> bool:
        done = self.controller.shutdown(timeout)
        if done:
            self._resources.close()
            self._bundled = None
        return done

    # ---- controller events ----
    def _on_event(self, event: str, **kw) -> None:
        if event == "preview_opened":
            self.mode, self.pending = PREVIEW_MODE, None
            if self.draft and self.draft.focus_requested is not None:
                try:
                    self.controller.set_focus(self.draft.focus_requested)  # reapply saved profile
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
