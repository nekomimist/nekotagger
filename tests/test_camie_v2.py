from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from nekotagger.errors import ModelError
from nekotagger.taggers.camie_v2 import (
    PADDING_COLOR,
    CamieV2Tagger,
    _load_metadata,
    preprocess_image,
)


def _write_metadata(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "model_info": {"img_size": 8},
                "input_spec": {
                    "format": "RGB image tensor",
                    "shape": [3, 8, 8],
                    "dtype": "float32",
                },
                "dataset_info": {
                    "total_tags": 3,
                    "categories": ["general", "character"],
                    "tag_mapping": {
                        "idx_to_tag": {
                            "0": "1girl",
                            "1": "alice",
                            "2": "long_hair",
                        },
                        "tag_to_category": {
                            "1girl": "general",
                            "alice": "character",
                            "long_hair": "general",
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_metadata_loads_contiguous_vocabulary_and_categories(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    _write_metadata(path)
    metadata = _load_metadata(path)
    assert metadata.image_size == 8
    assert metadata.tag_names == ("1girl", "alice", "long_hair")
    assert metadata.tag_categories == ("general", "character", "general")
    assert metadata.category_names == {"general", "character"}


def test_metadata_requires_every_tag_to_have_a_category(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    _write_metadata(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["dataset_info"]["tag_mapping"]["tag_to_category"]["alice"]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ModelError, match="no category"):
        _load_metadata(path)


def test_preprocess_upscales_and_letterboxes() -> None:
    image = Image.new("RGB", (4, 2), (255, 0, 0))
    pixels = preprocess_image(image, 8)
    assert pixels.shape == (3, 8, 8)
    assert pixels.dtype == np.float32
    assert pixels[0, 2, 0] == pytest.approx((1.0 - 0.485) / 0.229)
    assert pixels[1, 2, 0] == pytest.approx((0.0 - 0.456) / 0.224)
    assert pixels[0, 0, 0] == pytest.approx(
        (PADDING_COLOR[0] / 255.0 - 0.485) / 0.229,
        abs=1e-6,
    )


def test_transparency_is_composited_over_padding_color() -> None:
    image = Image.new("RGBA", (8, 8), (255, 0, 0, 0))
    pixels = preprocess_image(image, 8)
    expected = np.asarray(
        [
            (PADDING_COLOR[0] / 255.0 - 0.485) / 0.229,
            (PADDING_COLOR[1] / 255.0 - 0.456) / 0.224,
            (PADDING_COLOR[2] / 255.0 - 0.406) / 0.225,
        ],
        dtype=np.float32,
    ).reshape(3, 1, 1)
    assert np.allclose(pixels, expected)


class _Node:
    def __init__(self, name: str, shape: list[int | str | None]) -> None:
        self.name = name
        self.shape = shape


class _Session:
    def __init__(self) -> None:
        self.last_outputs: list[str] | None = None
        self.last_feed: dict[str, np.ndarray] | None = None

    def get_inputs(self) -> list[_Node]:
        return [_Node("images", ["batch_size", 3, 8, 8])]

    def get_outputs(self) -> list[_Node]:
        return [
            _Node("initial_predictions", ["batch_size", 3]),
            _Node("refined_predictions", ["batch_size", 3]),
            _Node("selected_candidates", ["batch_size", 256]),
        ]

    def run(
        self,
        outputs: list[str],
        feed: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        self.last_outputs = outputs
        self.last_feed = feed
        return [np.asarray([[0.0, 2.0, -2.0]], dtype=np.float32)]


def test_predict_uses_refined_logits_and_native_categories(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    _write_metadata(path)
    metadata = _load_metadata(path)
    session = _Session()
    tagger = CamieV2Tagger(device="cpu")
    tagger._inspect_session(session, metadata)  # type: ignore[arg-type]
    tagger._metadata = metadata
    tagger._session = session  # type: ignore[assignment]
    tagger.active_device = "cpu"

    predictions = tagger.predict_batch(
        [Image.new("RGB", (8, 8), "white")],
        threshold=0.5,
        max_tags=None,
    )

    assert [prediction.name for prediction in predictions[0]] == ["alice", "1girl"]
    assert predictions[0][0].confidence == pytest.approx(0.880797)
    assert session.last_outputs == ["refined_predictions"]
    assert session.last_feed is not None
    assert session.last_feed["images"].shape == (1, 3, 8, 8)
    assert tagger.tags_in_categories({"character"}) == {"alice"}
