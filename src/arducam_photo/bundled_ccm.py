"""Package resource lookup for the default tuning file (resources/arducam_108mp.json)."""

from __future__ import annotations

import contextlib
from importlib import resources
from typing import Optional

from .errors import CaptureConfigError

RESOURCE_PACKAGE = "arducam_photo.resources"
RESOURCE_NAME = "arducam_108mp.json"


def bundled_ccm_exists() -> bool:
    return resources.files(RESOURCE_PACKAGE).joinpath(RESOURCE_NAME).is_file()


def bundled_ccm_path(stack: contextlib.ExitStack) -> str:
    """Filesystem path of the bundled file. The caller owns `stack`: a temporary extraction (zip
    installs) lives until the stack is closed. Raises CaptureConfigError if the resource is absent."""
    ref = resources.files(RESOURCE_PACKAGE).joinpath(RESOURCE_NAME)
    if not ref.is_file():
        raise CaptureConfigError(
            f"ressource {RESOURCE_PACKAGE}/{RESOURCE_NAME} absente de l'installation: "
            "réinstallez le projet ou choisissez un fichier (option avancée)")
    return str(stack.enter_context(resources.as_file(ref)))
