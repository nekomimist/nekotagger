from __future__ import annotations

from pathlib import Path

from PIL import Image

from nekotagger.images import discover_images, load_image


def _save_image(path: Path, *, color: str = "red") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), color).save(path)


def test_discovery_is_nonrecursive_by_default(tmp_path: Path) -> None:
    top = tmp_path / "top.png"
    nested = tmp_path / "nested" / "deep.JPG"
    _save_image(top)
    _save_image(nested)
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")
    assert discover_images(tmp_path, recursive=False) == [top]
    assert discover_images(tmp_path, recursive=True) == [nested, top]


def test_load_image_uses_first_gif_frame(tmp_path: Path) -> None:
    path = tmp_path / "animated.gif"
    frames = [Image.new("RGB", (4, 4), color) for color in ("red", "blue")]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=10, loop=0)
    loaded = load_image(path)
    try:
        pixel = loaded.convert("RGB").getpixel((0, 0))
        assert pixel[0] > pixel[2]
    finally:
        loaded.close()
