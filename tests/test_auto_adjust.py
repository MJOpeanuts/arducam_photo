from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from arducam_photo.auto_adjust import (
    AdjustmentError, AdjustmentOptions, AutoAdjuster, _metrics, restore_settings,
)
from arducam_photo.config import SettingReport


class Session:
    def __init__(self, *, exposure_effect=True, gain_effect=True):
        self.values = {"exposure": 1.0, "gain": 1.0, "auto_exposure": 17.0, "focus": 45}
        self.initial = dict(self.values)
        self.calls = []
        self.reads = 0
        self.exposure_effect = exposure_effect
        self.gain_effect = gain_effect
        self.reject = None
        self.mismatch = None
        self.fail_read = False
        self.restore_reject = False

    def snapshot_settings(self):
        return dict(self.values)

    def apply_settings(self, values):
        self.calls.append(dict(values))
        reports = {}
        for name, value in values.items():
            if name.endswith("_mode"):
                continue
            accepted = name != self.reject
            if self.restore_reject and name == "auto_exposure" and value == 17:
                accepted = False
            if accepted:
                self.values[name] = value
            readback = value + 1 if name == self.mismatch else self.values[name]
            reports[name] = SettingReport(value, accepted, readback)
        return reports

    def read(self):
        self.reads += 1
        if self.fail_read:
            return None
        level = 20
        if self.exposure_effect:
            level += 20 * self.values["exposure"]
        if self.gain_effect:
            level += 10 * self.values["gain"]
        texture = np.indices((12, 12)).sum(axis=0) % 2 * 8
        gray = np.clip(level + texture, 0, 255).astype(np.uint8)
        return np.repeat(gray[:, :, None], 3, axis=2)


def options(**changes):
    return replace(AdjustmentOptions(exposure_bounds=(0, 8), gain_bounds=(0, 4),
                                     settle_frames=0, max_iterations=30), **changes)


def before(**changes):
    return dict(focus=45, exposure=None, gain=None, auto_exposure=3.75,
                auto_exposure_mode="manual", **changes)


def test_search_improves_brightness_preserves_focus_and_requested_unset_values():
    session = Session()
    requested = before()
    result = AutoAdjuster().run(session, requested, options())
    assert abs(result.metrics["brightness"] - 0.5) < abs(54 / 255 - 0.5)
    assert result.before == requested and requested["exposure"] is None
    assert result.proposed["focus"] == session.values["focus"] == 45
    assert not any("focus" in call for call in session.calls)
    assert result.original_settings == {"exposure": 1.0, "gain": 1.0, "auto_exposure": 17.0}
    assert 3 <= result.iterations <= 30
    assert result.elapsed >= 0 and not result.restoration_warnings
    assert session.values["exposure"] == result.proposed["exposure"]
    assert session.values["gain"] == result.proposed["gain"]
    with pytest.raises(FrozenInstanceError):
        result.iterations = 0
    with pytest.raises(FrozenInstanceError):
        options().roi = (0, 0, 0.5, 0.5)


def test_reject_restores_observed_values_not_unset_requested_bank():
    session = Session()
    result = AutoAdjuster().run(session, before(), options())
    assert restore_settings(session, result.original_settings) == []
    assert session.values == session.initial
    assert list(session.calls[-1]) == ["auto_exposure"]


@pytest.mark.parametrize("effect", ["exposure", "gain"])
def test_set_get_without_measurable_image_effect_is_not_success(effect):
    session = Session(**{f"{effect}_effect": False})
    with pytest.raises(AdjustmentError, match=f"{effect} has no measurable") as caught:
        AutoAdjuster().run(session, before(), options())
    assert not caught.value.restoration_warnings
    assert session.values == session.initial


@pytest.mark.parametrize("control", ["exposure", "gain", "auto_exposure"])
@pytest.mark.parametrize("failure", ["reject", "mismatch"])
def test_rejected_or_mismatching_controls_refuse_search_and_attempt_all_restoration(control, failure):
    session = Session()
    setattr(session, failure, control)
    with pytest.raises(AdjustmentError) as caught:
        AutoAdjuster().run(session, before(), options())
    assert caught.value.restoration_warnings
    restoration = session.calls[-3:]
    assert [list(call)[0] for call in restoration] == ["exposure", "gain", "auto_exposure"]


@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf")])
def test_unreadable_original_preflight_performs_no_writes(invalid):
    session = Session()
    session.values["gain"] = invalid
    with pytest.raises(AdjustmentError, match="original gain"):
        AutoAdjuster().run(session, before(), options())
    assert session.calls == []


def test_snapshot_reports_use_readback_not_requested():
    session = Session()
    original_snapshot = session.snapshot_settings
    session.snapshot_settings = lambda: {
        name: SettingReport(999, None, value, attempted=False)
        for name, value in original_snapshot().items()
    }
    result = AutoAdjuster().run(session, before(), options())
    assert result.original_settings["exposure"] == 1


def test_snapshot_readback_error_refuses_preflight():
    session = Session()
    session.snapshot_settings = lambda: {
        "exposure": 1, "gain": SettingReport(1, True, 1, readback_error="unsupported"),
        "auto_exposure": 17,
    }
    with pytest.raises(AdjustmentError, match="original gain"):
        AutoAdjuster().run(session, before(), options())
    assert not session.calls


@pytest.mark.parametrize("readback", [float("nan"), float("inf"), None])
def test_nonfinite_or_missing_writable_readback_never_succeeds(readback):
    session = Session()
    original_apply = session.apply_settings
    def apply(values):
        reports = original_apply(values)
        if "gain" in reports:
            reports["gain"] = SettingReport(values["gain"], True, readback)
        return reports
    session.apply_settings = apply
    with pytest.raises(AdjustmentError, match="gain.*readback"):
        AutoAdjuster().run(session, before(), options())
    assert session.values == session.initial


def test_snapshot_exception_is_wrapped_without_writes():
    session = Session()
    def broken():
        raise RuntimeError("device offline")
    session.snapshot_settings = broken
    with pytest.raises(AdjustmentError, match="device offline"):
        AutoAdjuster().run(session, before(), options())
    assert not session.calls


@pytest.mark.parametrize("values", [
    {"auto_exposure": None, "auto_exposure_mode": "manual"},
    {"auto_exposure": 3.75, "auto_exposure_mode": "automatic"},
    {"auto_exposure": 3.75},
])
def test_manual_mode_must_be_explicit_without_backend_conventions(values):
    session = Session()
    with pytest.raises(AdjustmentError, match="explicit manual"):
        AutoAdjuster().run(session, values, options())
    assert not session.calls


def test_failed_frame_rolls_back_modes_and_reports_restore_failure():
    session = Session()
    session.fail_read = True
    session.restore_reject = True
    with pytest.raises(AdjustmentError) as caught:
        AutoAdjuster().run(session, before(), options())
    assert any("auto_exposure" in warning for warning in caught.value.restoration_warnings)
    assert session.values["exposure"] == session.initial["exposure"]
    assert session.values["gain"] == session.initial["gain"]


def test_cancellation_restores_observed_settings():
    session = Session()
    with pytest.raises(AdjustmentError, match="cancelled"):
        AutoAdjuster().run(session, before(), options(), lambda: session.reads >= 2)
    assert session.values == session.initial


def test_cancellation_before_preflight_does_not_write():
    session = Session()
    with pytest.raises(AdjustmentError, match="cancelled"):
        AutoAdjuster().run(session, before(), options(), lambda: True)
    assert not session.calls


def test_settling_frames_are_discarded_without_sleep():
    session = Session()
    result = AutoAdjuster().run(session, before(), options(max_iterations=3, settle_frames=2))
    assert session.reads == (result.iterations + 1) * 3


def test_iteration_bound_returns_best_evaluated_candidate():
    session = Session()
    result = AutoAdjuster().run(session, before(), options(max_iterations=3))
    assert result.iterations == 3
    assert (result.proposed["exposure"], result.proposed["gain"]) in [(1, 1), (8, 1), (1, 4)]
    assert result.converged is False and result.stop_reason == "max_iterations"
    assert result.metrics["converged"] is False
    assert result.metrics["stop_reason"] == "max_iterations"


def test_final_metrics_are_measured_after_best_candidate_is_reapplied():
    session = Session()
    result = AutoAdjuster().run(session, before(), options())
    final = _metrics(session.read(), options())
    for name, value in final.items():
        assert result.metrics[name] == value
    assert result.converged == result.metrics["converged"]
    assert result.stop_reason == result.metrics["stop_reason"]


def test_failed_final_verification_restores_original_controls():
    session = Session()
    original_read = session.read
    def read():
        if session.reads >= 3:
            return None
        return original_read()
    session.read = read
    with pytest.raises(AdjustmentError, match="valid uint8"):
        AutoAdjuster().run(session, before(), options(max_iterations=3))
    assert session.values == session.initial


def test_time_budget_is_checked_after_native_call_and_rolls_back(monkeypatch):
    import arducam_photo.auto_adjust as module
    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    session = Session()
    original_read = session.read
    def slow_read():
        clock[0] += 20
        return original_read()
    session.read = slow_read
    with pytest.raises(AdjustmentError, match="time budget"):
        AutoAdjuster().run(session, before(), options(time_budget=1))
    assert session.values == session.initial


def test_roi_brightness_saturation_and_local_contrast():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[:, 2:] = 255
    metrics = _metrics(frame, options(roi=(0.5, 0, 0.5, 1)))
    assert metrics == {"brightness": pytest.approx(1), "saturated_fraction": 1,
                       "local_contrast": 0}
    full = _metrics(frame, options())
    assert full["brightness"] == pytest.approx(0.5)
    assert full["saturated_fraction"] == 0.5 and full["local_contrast"] > 0


@pytest.mark.parametrize("changes", [
    {"roi": (0, 0, 0, 1)}, {"roi": (0.5, 0, 0.6, 1)},
    {"roi": (0, float("nan"), 1, 1)}, {"exposure_bounds": None},
    {"gain_bounds": (2, 2)}, {"exposure_step": 0}, {"gain_step": float("inf")},
    {"time_budget": -1}, {"max_iterations": 2}, {"settle_frames": -1},
    {"max_saturated_fraction": 2}, {"response_threshold": 0},
])
def test_invalid_options_do_not_write(changes):
    session = Session()
    with pytest.raises(AdjustmentError):
        AutoAdjuster().run(session, before(), options(**changes))
    assert not session.calls


def test_restore_unavailable_values_is_explicit_and_continues():
    session = Session()
    warnings = restore_settings(session, {"exposure": None, "gain": 2, "auto_exposure": 17})
    assert len(warnings) == 1 and "restoration impossible" in warnings[0]
    assert session.values["gain"] == 2


def test_restore_exception_does_not_skip_remaining_controls():
    session = Session()
    apply = session.apply_settings
    def fails(values):
        if "exposure" in values:
            raise RuntimeError("unplugged")
        return apply(values)
    session.apply_settings = fails
    warnings = restore_settings(session, {"exposure": 1, "gain": 2, "auto_exposure": 17})
    assert len(warnings) == 1 and "unplugged" in warnings[0]
    assert session.values["gain"] == 2 and session.values["auto_exposure"] == 17
