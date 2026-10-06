"""Local Lucide SVGs (ISC), rendered without filesystem or network dependencies."""

from importlib import resources

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer


def resource_bytes(name: str) -> bytes:
    ref = resources.files("arducam_photo.resources").joinpath(name)
    if not ref.is_file():
        raise FileNotFoundError(f"Ressource fournie manquante : arducam_photo/resources/{name}")
    return ref.read_bytes()


def bundled_pixmap(name: str) -> QPixmap:
    pixmap = QPixmap()
    if not pixmap.loadFromData(resource_bytes(name)):
        raise ValueError(f"Image fournie invalide : arducam_photo/resources/{name}")
    return pixmap


def application_icon() -> QIcon:
    for name in ("nuts-app.svg", "nuts-app.png", "nuts-app.ico"):
        resource_bytes(f"icons/{name}")
    icon = QIcon()
    # Copy every original ICO size into memory, including for zip-installed packages.
    ref = resources.files("arducam_photo.resources").joinpath("icons/nuts-app.ico")
    with resources.as_file(ref) as path:
        original = QIcon(str(path))
        for size in original.availableSizes():
            icon.addPixmap(original.pixmap(size))
    icon.addPixmap(bundled_pixmap("icons/nuts-app.png"))
    return icon


def lucide_icon(name: str, color: str, size: int = 24) -> QIcon:
    if name not in ("squirrel", "refresh-cw"):
        raise ValueError(f"Unknown bundled icon: {name}")
    svg = resources.files("arducam_photo.resources").joinpath("icons", f"{name}.svg").read_bytes()
    renderer = QSvgRenderer(QByteArray(svg.replace(b"currentColor", color.encode("ascii"))))
    if not renderer.isValid():
        raise ValueError(f"Invalid bundled SVG: {name}")
    icon = QIcon()
    for scale in (1, 1.25, 1.5, 2):
        pixels = round(size * scale)
        pixmap = QPixmap(pixels, pixels)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        pixmap.setDevicePixelRatio(scale)
        icon.addPixmap(pixmap)
    return icon
