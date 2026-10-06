"""Verify original resources in wheels and the portable build specification."""

from pathlib import Path
import os
import runpy
import shutil
import subprocess
import sys
from types import SimpleNamespace
from zipfile import ZipFile

import pytest


ROOT = Path(__file__).resolve().parents[1]
RESOURCE_PATH = Path("src/arducam_photo/resources")
REQUIRED_RESOURCES = (
    "arducam_108mp.json",
    "icons/nuts-app.svg",
    "icons/nuts-app.png",
    "icons/nuts-app.ico",
    "icons/squirrel.svg",
    "icons/refresh-cw.svg",
    "icons/LICENSE",
    "powered by_white.png",
)
ORIGINAL_BRANDING = REQUIRED_RESOURCES[1:4] + ("powered by_white.png",)


@pytest.fixture(scope="module")
def wheel(tmp_path_factory):
    project = tmp_path_factory.mktemp("wheel-project")
    shutil.copytree(ROOT / "src", project / "src")
    for name in ("pyproject.toml", "README.md"):
        shutil.copy2(ROOT / name, project / name)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from setuptools.build_meta import build_wheel; build_wheel('dist')",
        ],
        cwd=project,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheels = list((project / "dist").glob("*.whl"))
    assert len(wheels) == 1
    return wheels[0]


@pytest.mark.parametrize("name", REQUIRED_RESOURCES)
def test_wheel_contains_unmodified_original_resources(wheel, name):
    with ZipFile(wheel) as archive:
        bundled = archive.read(f"arducam_photo/resources/{name}")
    assert bundled == (ROOT / RESOURCE_PATH / name).read_bytes()
    assert bundled


def test_installed_resources_resolve_outside_source_tree(wheel, tmp_path):
    installed = tmp_path / "installed"
    with ZipFile(wheel) as archive:
        archive.extractall(installed)
    script = (
        "import sys; from importlib import resources; "
        "sys.path.insert(0, sys.argv[1]); "
        "root = resources.files('arducam_photo.resources'); "
        "assert str(root).startswith(sys.argv[1]); "
        "assert all(root.joinpath(name).read_bytes() for name in sys.argv[2:])"
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", script, str(installed), *REQUIRED_RESOURCES],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def evaluate_spec(root):
    calls = {}

    def analysis(*args, **kwargs):
        calls["analysis"] = (args, kwargs)
        return SimpleNamespace(pure=[], scripts=[], binaries=[], datas=kwargs["datas"])

    def exe(*args, **kwargs):
        calls["exe"] = (args, kwargs)
        return object()

    def collect(*args, **kwargs):
        calls["collect"] = (args, kwargs)

    runpy.run_path(
        str(ROOT / "arducamcapture.spec"),
        init_globals={
            "SPECPATH": str(root),
            "Analysis": analysis,
            "PYZ": lambda pure: object(),
            "EXE": exe,
            "COLLECT": collect,
        },
    )
    return calls


def test_spec_is_windowed_onedir_and_includes_qt_hooks():
    calls = evaluate_spec(ROOT)
    _, analysis = calls["analysis"]
    assert analysis["datas"] == [
        (str(ROOT / RESOURCE_PATH), "arducam_photo/resources")
    ]
    assert {
        "arducam_photo.resources",
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        "PySide6.QtSvg",
    } <= set(analysis["hiddenimports"])
    _, exe = calls["exe"]
    assert exe["console"] is False
    assert exe["exclude_binaries"] is True
    assert exe["icon"] == str(ROOT / RESOURCE_PATH / "icons/nuts-app.ico")
    assert calls["collect"][1]["name"] == "ArducamCapture"


@pytest.mark.parametrize("missing", [(name,) for name in ORIGINAL_BRANDING] + [ORIGINAL_BRANDING])
def test_spec_reports_exact_missing_original_assets(tmp_path, missing):
    shutil.copytree(ROOT / RESOURCE_PATH, tmp_path / RESOURCE_PATH)
    for name in missing:
        (tmp_path / RESOURCE_PATH / name).unlink()
    with pytest.raises(FileNotFoundError) as error:
        evaluate_spec(tmp_path)
    assert str(error.value) == (
        "Missing required original package resources: "
        + ", ".join(f"src/arducam_photo/resources/{name}" for name in missing)
    )
