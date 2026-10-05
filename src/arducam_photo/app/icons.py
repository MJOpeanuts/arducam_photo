"""Local Lucide SVGs (ISC), rendered without filesystem or network dependencies."""

from importlib import resources

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer


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
