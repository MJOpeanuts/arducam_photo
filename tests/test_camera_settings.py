from types import SimpleNamespace

from arducam_photo.camera_settings import apply_camera_settings, read_camera_settings


def test_auto_controls_are_attempted_before_manual_controls():
    constants = SimpleNamespace(
        CAP_PROP_AUTO_EXPOSURE=1, CAP_PROP_AUTO_WB=2, CAP_PROP_FOCUS=3,
        CAP_PROP_EXPOSURE=4, CAP_PROP_GAIN=5, CAP_PROP_WB_TEMPERATURE=6,
        CAP_PROP_BRIGHTNESS=7, CAP_PROP_CONTRAST=8, CAP_PROP_SATURATION=9,
    )

    class Camera:
        def __init__(self):
            self.calls = []

        def set(self, prop, value):
            self.calls.append((prop, value))
            return True

        def get(self, prop):
            return 0.0

    camera = Camera()
    reports = apply_camera_settings(camera, {
        "focus": 45, "auto_exposure": 0.25, "exposure": 2.5,
        "auto_wb": 1, "wb_temperature": 4800,
    }, constants)
    assert [prop for prop, _ in camera.calls] == [1, 2, 3, 4, 6]
    assert reports["focus"].readback == 0.0
    assert reports["exposure"].accepted is True
    assert reports["exposure"].mode is None


def test_unset_unsupported_rejected_and_unreadable_controls_are_distinguished():
    constants = SimpleNamespace(CAP_PROP_GAIN=5, CAP_PROP_CONTRAST=8)

    class Camera:
        def set(self, prop, value):
            if prop == 5:
                return False
            raise RuntimeError("set exception")

        def get(self, prop):
            if prop == 5:
                return 0.0
            raise RuntimeError("get exception")

    reports = apply_camera_settings(
        Camera(), {"gain": 4, "contrast": 1, "brightness": 2}, constants
    )
    assert "focus" not in reports
    assert reports["gain"].accepted is False and reports["gain"].readback == 0.0
    assert reports["contrast"].error == "set exception"
    assert reports["contrast"].readback_error == "get exception"
    assert reports["brightness"].attempted is False


def test_readback_refresh_preserves_set_result():
    constants = SimpleNamespace(CAP_PROP_FOCUS=3)

    class Camera:
        def set(self, prop, value):
            return True

        def get(self, prop):
            return 0.0

    before = {"focus": apply_camera_settings(Camera(), {"focus": 9}, constants)["focus"]}
    after = read_camera_settings(Camera(), before, constants)["focus"]
    assert after.accepted is True
    assert after.requested == 9 and after.readback == 0.0


def test_automatic_modes_prevent_incompatible_manual_commands():
    constants = SimpleNamespace(CAP_PROP_AUTO_EXPOSURE=1, CAP_PROP_AUTO_WB=2,
                                CAP_PROP_EXPOSURE=4, CAP_PROP_GAIN=5,
                                CAP_PROP_WB_TEMPERATURE=6)

    class Camera:
        def __init__(self):
            self.calls = []

        def set(self, prop, value):
            self.calls.append(prop)
            return True

        def get(self, prop):
            return 1.0

    camera = Camera()
    reports = apply_camera_settings(camera, {
        "auto_exposure": 0.25, "auto_exposure_mode": "automatic",
        "exposure": 4, "gain": 2, "auto_wb": 1,
        "auto_wb_mode": "automatic", "wb_temperature": 5000,
    }, constants)
    assert camera.calls == [1, 2]
    assert reports["exposure"].attempted is False
    assert reports["gain"].attempted is False
    assert reports["wb_temperature"].attempted is False
    assert reports["exposure"].mode == "automatic"
