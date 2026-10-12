"""Bounded exposure/gain search on an existing camera session.

Budgets and cancellation are checked between calls, not inside blocking native
camera operations. Final verification and rollback can outlast the search budget.
Driver readbacks alone never establish physical control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Callable

import numpy as np

from .config import SettingReport


@dataclass(frozen=True)
class AdjustmentOptions:
    roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    exposure_bounds: tuple[float, float] | None = None
    gain_bounds: tuple[float, float] | None = None
    exposure_step: float = 1.0
    gain_step: float = 1.0
    max_iterations: int = 30
    time_budget: float = 10.0
    settle_frames: int = 2
    brightness_target: float = 0.5
    brightness_tolerance: float = 0.05
    saturated_threshold: float = 0.98
    max_saturated_fraction: float = 0.02
    min_local_contrast: float = 0.01
    response_threshold: float = 0.005

    def validate(self) -> None:
        for name in ("exposure_bounds", "gain_bounds"):
            bounds = getattr(self, name)
            if (not isinstance(bounds, (tuple, list)) or len(bounds) != 2
                    or not all(_finite(v) for v in bounds) or bounds[0] >= bounds[1]):
                raise AdjustmentError(f"{name} requires explicit finite increasing backend bounds")
        for name in ("exposure_step", "gain_step", "time_budget", "response_threshold"):
            if not _finite(getattr(self, name)) or getattr(self, name) <= 0:
                raise AdjustmentError(f"{name} must be finite and positive")
        for name in ("brightness_target", "brightness_tolerance", "saturated_threshold",
                     "max_saturated_fraction", "min_local_contrast"):
            if not _finite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise AdjustmentError(f"{name} must be between zero and one")
        if type(self.max_iterations) is not int or self.max_iterations < 3:
            raise AdjustmentError("max_iterations must be an integer >= 3 for capability trials")
        if type(self.settle_frames) is not int or self.settle_frames < 0:
            raise AdjustmentError("settle_frames must be an integer >= 0")
        if (not isinstance(self.roi, (tuple, list)) or len(self.roi) != 4
                or not all(_finite(v) for v in self.roi)):
            raise AdjustmentError("roi must be normalized (x, y, width, height)")
        x, y, w, h = self.roi
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > 1 or y + h > 1:
            raise AdjustmentError("roi must be nonempty and contained in the image")


@dataclass(frozen=True)
class AdjustmentResult:
    before: dict
    proposed: dict
    metrics: dict
    iterations: int
    elapsed: float
    original_settings: dict = field(default_factory=dict)
    restoration_warnings: tuple[str, ...] = ()
    converged: bool = False
    stop_reason: str = ""


class AdjustmentError(RuntimeError):
    def __init__(self, message: str, restoration_warnings=()):
        super().__init__(message)
        self.restoration_warnings = tuple(restoration_warnings)


def _finite(value) -> bool:
    return (not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, float, np.number))
            and math.isfinite(float(value)))


def _observed(snapshot: dict, name: str):
    value = snapshot.get(name)
    if isinstance(value, SettingReport):
        if value.readback_error:
            return None
        value = value.readback
    return value


def _report_problem(reports, name, requested) -> str | None:
    report = reports.get(name) if isinstance(reports, dict) else None
    if not isinstance(report, SettingReport):
        return f"{name}: no verifiable SettingReport"
    if not report.attempted or report.accepted is not True or report.error:
        return f"{name}: backend did not accept the control ({report.error or 'rejected'})"
    if (report.readback_error or not _finite(report.readback)
            or not math.isclose(float(report.readback), float(requested),
                                rel_tol=1e-6, abs_tol=1e-6)):
        return f"{name}: backend readback does not match requested value"
    return None


def restore_settings(session, before: dict) -> list[str]:
    """Best-effort verified rollback; restore modes last to avoid automatic blocking.

    Pass ``result.original_settings``, not a requested bank with unset controls.
    Unknown/unreadable original values cannot be safely guessed.
    """
    warnings = []
    modes = {"auto_exposure", "auto_wb"}
    names = [name for name in before if not name.endswith("_mode") and name not in modes]
    names += [name for name in ("auto_wb", "auto_exposure") if name in before]
    for name in names:
        value = _observed(before, name)
        if not _finite(value):
            warnings.append(f"{name}: original value unavailable; restoration impossible")
            continue
        try:
            reports = session.apply_settings({name: value})
            problem = _report_problem(reports, name, value)
            if problem:
                warnings.append(f"restoration {problem}")
        except Exception as exc:
            warnings.append(f"restoration {name}: {exc}")
    return warnings


def _metrics(frame, options: AdjustmentOptions) -> dict:
    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
            or frame.ndim not in (2, 3) or not frame.size
            or (frame.ndim == 3 and frame.shape[2] != 3)):
        raise AdjustmentError("camera returned no valid uint8 preview image")
    height, width = frame.shape[:2]
    x, y, w, h = options.roi
    left, top = int(x * width), int(y * height)
    right, bottom = min(width, math.ceil((x + w) * width)), min(height, math.ceil((y + h) * height))
    roi = frame[top:bottom, left:right].astype(np.float64) / 255.0
    if not roi.size:
        raise AdjustmentError("roi contains no pixels")
    if roi.ndim == 3:
        gray = roi @ np.array([0.114, 0.587, 0.299])
        saturated = np.any(roi >= options.saturated_threshold, axis=2)
    else:
        gray = roi
        saturated = roi >= options.saturated_threshold
    differences = []
    if gray.shape[0] > 1:
        differences.append(np.abs(np.diff(gray, axis=0)).ravel())
    if gray.shape[1] > 1:
        differences.append(np.abs(np.diff(gray, axis=1)).ravel())
    contrast = float(np.concatenate(differences).mean()) if differences else 0.0
    return {"brightness": float(gray.mean()), "saturated_fraction": float(saturated.mean()),
            "local_contrast": contrast}


def _score(metrics: dict, options: AdjustmentOptions) -> float:
    return (abs(metrics["brightness"] - options.brightness_target)
            + 2 * max(0, metrics["saturated_fraction"] - options.max_saturated_fraction)
            + max(0, options.min_local_contrast - metrics["local_contrast"]))


def _converged(metrics: dict, options: AdjustmentOptions) -> bool:
    return (abs(metrics["brightness"] - options.brightness_target) <= options.brightness_tolerance
            and metrics["saturated_fraction"] <= options.max_saturated_fraction
            and metrics["local_contrast"] >= options.min_local_contrast)


def _responded(first: dict, second: dict, options: AdjustmentOptions) -> bool:
    return any(abs(first[name] - second[name]) >= options.response_threshold for name in first)


class AutoAdjuster:
    def run(self, session, before: dict, options: AdjustmentOptions,
            cancelled: Callable[[], bool] = lambda: False) -> AdjustmentResult:
        """Leave the best candidate active for accept/reject, never persist settings.

        Both exposure and gain must be readable, writable, and visibly effective.
        Manual mode and backend bounds must be supplied explicitly by the caller.
        """
        options.validate()
        requested = dict(before)
        if requested.get("auto_exposure_mode") != "manual" or not _finite(requested.get("auto_exposure")):
            raise AdjustmentError("explicit manual auto_exposure backend value is required")
        started = time.monotonic()
        original = {}
        attempted = False
        iterations = 0

        def check(*, budget=True):
            if cancelled():
                raise AdjustmentError("adjustment cancelled")
            if budget and time.monotonic() - started >= options.time_budget:
                raise AdjustmentError("time budget exhausted before a verified candidate")

        def apply(values):
            nonlocal attempted
            check()
            attempted = True
            reports = session.apply_settings(values)
            for name, value in values.items():
                if name.endswith("_mode"):
                    continue
                problem = _report_problem(reports, name, value)
                if problem:
                    raise AdjustmentError(problem)
            check()

        def acquire(*, budget=True):
            for _ in range(options.settle_frames):
                check(budget=budget)
                frame = session.read()
                if frame is None:
                    raise AdjustmentError("camera read failed during stabilization")
                check(budget=budget)
            check(budget=budget)
            metrics = _metrics(session.read(), options)
            check(budget=budget)
            return metrics

        def evaluate(exposure, gain):
            nonlocal iterations
            apply({"exposure": exposure, "gain": gain, "auto_exposure_mode": "manual"})
            metrics = acquire()
            iterations += 1
            return metrics

        try:
            check()
            snapshot = session.snapshot_settings()
            if not isinstance(snapshot, dict):
                raise AdjustmentError("original camera settings snapshot is unavailable")
            for name in ("exposure", "gain", "auto_exposure"):
                value = _observed(snapshot, name)
                if not _finite(value):
                    raise AdjustmentError(f"original {name} is unreadable; safe adjustment unavailable")
                original[name] = value
            check()
            apply({"auto_exposure": requested["auto_exposure"], "auto_exposure_mode": "manual"})
            e_low, e_high = options.exposure_bounds
            g_low, g_high = options.gain_bounds
            base = (min(e_high, max(e_low, original["exposure"])),
                    min(g_high, max(g_low, original["gain"])))
            baseline = evaluate(*base)
            e_trial = e_high if base[0] - e_low < e_high - base[0] else e_low
            g_trial = g_high if base[1] - g_low < g_high - base[1] else g_low
            exposure_metrics = evaluate(e_trial, base[1])
            if not _responded(baseline, exposure_metrics, options):
                raise AdjustmentError("exposure has no measurable image response")
            gain_metrics = evaluate(base[0], g_trial)
            if not _responded(baseline, gain_metrics, options):
                raise AdjustmentError("gain has no measurable image response")
            candidates = [(base, baseline), ((e_trial, base[1]), exposure_metrics),
                          ((base[0], g_trial), gain_metrics)]
            best, best_metrics = min(candidates, key=lambda item: _score(item[1], options))
            seen = {item[0] for item in candidates}
            stop_reason = "local_optimum"
            while iterations < options.max_iterations and not _converged(best_metrics, options):
                check(budget=False)
                if time.monotonic() - started >= options.time_budget:
                    stop_reason = "time_budget"
                    break
                neighbors = [
                    (min(e_high, max(e_low, best[0] + delta)), best[1])
                    for delta in (-options.exposure_step, options.exposure_step)
                ] + [
                    (best[0], min(g_high, max(g_low, best[1] + delta)))
                    for delta in (-options.gain_step, options.gain_step)
                ]
                improved = False
                for candidate in neighbors:
                    if candidate in seen:
                        continue
                    if iterations >= options.max_iterations or time.monotonic() - started >= options.time_budget:
                        break
                    seen.add(candidate)
                    metrics = evaluate(*candidate)
                    if _score(metrics, options) < _score(best_metrics, options):
                        best, best_metrics = candidate, metrics
                        improved = True
                if not improved:
                    break
            if iterations >= options.max_iterations:
                stop_reason = "max_iterations"
            elif time.monotonic() - started >= options.time_budget:
                stop_reason = "time_budget"
            # The last trial may not be the winner; reapply with verified readbacks.
            check(budget=False)
            attempted = True
            reports = session.apply_settings(
                {"exposure": best[0], "gain": best[1], "auto_exposure_mode": "manual"})
            for name, value in zip(("exposure", "gain"), best):
                problem = _report_problem(reports, name, value)
                if problem:
                    raise AdjustmentError(problem)
            check(budget=False)
            best_metrics = acquire(budget=False)
            converged = _converged(best_metrics, options)
            if converged:
                stop_reason = "converged"
            elif stop_reason == "local_optimum":
                stop_reason = "criteria_not_met"
            proposed = dict(requested)
            proposed.update(exposure=best[0], gain=best[1],
                            auto_exposure=requested["auto_exposure"], auto_exposure_mode="manual")
            best_metrics.update(converged=converged, stop_reason=stop_reason)
            return AdjustmentResult(requested, proposed, dict(best_metrics), iterations,
                                    time.monotonic() - started, dict(original),
                                    converged=converged, stop_reason=stop_reason)
        except Exception as exc:
            warnings = restore_settings(session, original) if attempted else []
            if isinstance(exc, AdjustmentError):
                raise AdjustmentError(str(exc), warnings) from exc
            raise AdjustmentError(f"adjustment failed: {exc}", warnings) from exc
