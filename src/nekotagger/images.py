"""Image discovery and loading helpers."""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageOps

SUPPORTED_EXTENSIONS = frozenset(
    {
        ".bmp",
        ".gif",
        ".jpeg",
        ".jpg",
        ".png",
        ".tif",
        ".tiff",
        ".webp",
    }
)


def discover_images(root: Path, *, recursive: bool) -> list[Path]:
    """Find supported images deterministically without following directory symlinks."""

    found: list[Path] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        dirnames[:] = sorted(
            (
                name
                for name in dirnames
                if not (directory_path / name).is_symlink()
            ),
            key=str.casefold,
        )
        for filename in sorted(filenames, key=str.casefold):
            path = directory_path / filename
            if path.suffix.casefold() in SUPPORTED_EXTENSIONS and path.is_file():
                found.append(path)
        if not recursive:
            break
    return sorted(
        found,
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )


def load_image(path: Path) -> Image.Image:
    """Load the first frame and detach it from the underlying file handle."""

    with Image.open(path) as opened:
        if getattr(opened, "is_animated", False):
            opened.seek(0)
        oriented = ImageOps.exif_transpose(opened)
        oriented.load()
        result = oriented.copy()
        result.info.update(oriented.info)
        return result
