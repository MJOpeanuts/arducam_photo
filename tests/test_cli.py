"""Command-line acquisition modes share the library's explicit mode registry."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest


CLI = Path(__file__).resolve().parents[1] / "examples" / "capture_cli.py"


@pytest.mark.parametrize("mode", ["color_720p", "color_4k", "color_12mp", "native_108mp"])
def test_cli_explicit_modes_and_requested_rate(monkeypatch, capsys, mode):
    import arducam_photo

    calls = []

    def save(config, path, overwrite=False):
        calls.append((config, path, overwrite))
        return SimpleNamespace(width=1280, height=720,
                               info=SimpleNamespace(ccm_applied=False, duration_s=0.1))

    monkeypatch.setattr(arducam_photo, "capture_and_save", save)
    main = runpy.run_path(str(CLI))["main"]
    assert main(["--path", mode, "--fps", "7", "--no-ccm", "-o", "photo.png"]) == 0
    config, path, overwrite = calls[0]
    assert config.path == mode and config.fps == 7
    assert path == "photo.png" and not overwrite
    assert "saved photo.png" in capsys.readouterr().out


def test_cli_unknown_mode_is_not_substituted():
    main = runpy.run_path(str(CLI))["main"]
    with pytest.raises(SystemExit) as error:
        main(["--path", "color_108mp", "-o", "photo.png"])
    assert error.value.code == 2


def test_cli_reports_preserved_original_after_render_failure(monkeypatch, capsys):
    import arducam_photo

    def fail(*args, **kwargs):
        error = arducam_photo.IspError("render failed")
        error.archive_path = "/photos/source/acquisition.json"
        raise error

    monkeypatch.setattr(arducam_photo, "capture_and_save", fail)
    main = runpy.run_path(str(CLI))["main"]
    assert main(["--path", "native_108mp", "-o", "photo.png"]) == 1
    assert "Original archivé : /photos/source/acquisition.json" in capsys.readouterr().err
