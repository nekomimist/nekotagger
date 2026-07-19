from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from nekotagger.errors import ModelError
from nekotagger.taggers.oppai_oracle import (
    PreprocessConfig,
    _load_vocabulary,
    preprocess_image,
)


def _config(size: int = 8) -> PreprocessConfig:
    return PreprocessConfig(
        image_size=size,
        pad_color=(114, 114, 114),
        mean=np.asarray([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3),
        std=np.asarray([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3),
    )


def test_preprocess_does_not_upscale_small_images() -> None:
    image = Image.new("RGB", (4, 2), (255, 0, 0))
    pixels, mask = preprocess_image(image, _config())
    assert pixels.shape == (3, 8, 8)
    assert mask.shape == (8, 8)
    assert np.count_nonzero(~mask) == 8
    assert np.all(~mask[3:5, 2:6])
    assert pixels[0, 3, 2] == pytest.approx(1.0)
    assert pixels[1, 3, 2] == pytest.approx(-1.0)


def test_preprocess_downscales_and_letterboxes() -> None:
    image = Image.new("RGB", (16, 8), "white")
    _, mask = preprocess_image(image, _config())
    assert np.count_nonzero(~mask) == 32
    assert np.all(~mask[2:6, :])


def test_transparency_is_composited_over_pad_color() -> None:
    image = Image.new("RGBA", (8, 8), (255, 0, 0, 0))
    pixels, mask = preprocess_image(image, _config())
    expected = (114 / 255.0 - 0.5) / 0.5
    assert not np.any(mask)
    assert np.allclose(pixels, expected)


def test_vocabulary_checksum_and_indices(tmp_path: Path) -> None:
    path = tmp_path / "vocabulary.json"
    path.write_text(
        json.dumps({"tag_to_index": {"<PAD>": 0, "<UNK>": 1, "1girl": 2}}),
        encoding="utf-8",
    )
    model_config = {
        "num_labels": 3,
        "vocab_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    names, skipped = _load_vocabulary(path, model_config)
    assert names == ["<PAD>", "<UNK>", "1girl"]
    assert skipped == {0, 1}


def test_vocabulary_checksum_mismatch_fails(tmp_path: Path) -> None:
    path = tmp_path / "vocabulary.json"
    path.write_text('{"tag_to_index":{"tag":0}}', encoding="utf-8")
    with pytest.raises(ModelError, match="checksum mismatch"):
        _load_vocabulary(path, {"num_labels": 1, "vocab_sha256": "0" * 64})


def test_embedded_vocabulary_overrides_stale_config_checksum(
    tmp_path: Path,
) -> None:
    path = tmp_path / "vocabulary.json"
    path.write_text('{"tag_to_index":{"tag":0}}', encoding="utf-8")
    vocabulary_bytes = path.read_bytes()
    actual_hash = hashlib.sha256(vocabulary_bytes).hexdigest()
    metadata = {
        "vocab_sha256": actual_hash,
        "vocab_b64_gzip": base64.b64encode(
            gzip.compress(vocabulary_bytes)
        ).decode("ascii"),
    }

    with pytest.warns(UserWarning, match="stale vocabulary checksum"):
        names, skipped = _load_vocabulary(
            path,
            {"num_labels": 1, "vocab_sha256": "0" * 64},
            metadata,
        )

    assert names == ["tag"]
    assert skipped == set()


def test_downloaded_vocabulary_must_match_embedded_copy(tmp_path: Path) -> None:
    path = tmp_path / "vocabulary.json"
    path.write_text('{"tag_to_index":{"tag":0}}', encoding="utf-8")
    embedded = b'{"tag_to_index":{"different":0}}'
    metadata = {
        "vocab_sha256": hashlib.sha256(embedded).hexdigest(),
        "vocab_b64_gzip": base64.b64encode(gzip.compress(embedded)).decode("ascii"),
    }

    with pytest.raises(ModelError, match="does not match"):
        _load_vocabulary(path, {"num_labels": 1}, metadata)
