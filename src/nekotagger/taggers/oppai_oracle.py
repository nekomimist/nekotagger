"""ONNX Runtime adapter for Grio43/OppaiOracle."""

from __future__ import annotations

import base64
import binascii
import gzip
import hashlib
import json
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image

from nekotagger.errors import InputError, ModelError
from nekotagger.taggers.onnx_runtime import create_session
from nekotagger.types import TagPrediction

MODEL_REPO = "Grio43/OppaiOracle"
MODEL_REVISION = "96992fa30568c386e9fe7c8a1a68f798a3202c09"


@dataclass(frozen=True, slots=True)
class VariantSpec:
    name: str
    subdirectory: str


VARIANTS = {
    "v1.1": VariantSpec(name="v1.1", subdirectory="V1.1_onnx"),
    "v1": VariantSpec(name="v1", subdirectory="V1_onnx"),
}


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    image_size: int
    pad_color: tuple[int, int, int]
    mean: np.ndarray
    std: np.ndarray


@dataclass(frozen=True, slots=True)
class ModelAssets:
    model: Path
    config: Path
    preprocessing: Path
    vocabulary: Path


def _download_assets(spec: VariantSpec) -> ModelAssets:
    def download(filename: str) -> Path:
        try:
            return Path(
                hf_hub_download(
                    repo_id=MODEL_REPO,
                    filename=f"{spec.subdirectory}/{filename}",
                    revision=MODEL_REVISION,
                )
            )
        except Exception as error:
            raise ModelError(
                f"failed to download {spec.subdirectory}/{filename}: {error}"
            ) from error

    return ModelAssets(
        model=download("model.onnx"),
        config=download("config.json"),
        preprocessing=download("preprocessing.json"),
        vocabulary=download("vocabulary.json"),
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ModelError(f"failed to read model metadata {path}: {error}") from error
    if not isinstance(value, dict):
        raise ModelError(f"model metadata must contain a JSON object: {path}")
    return value


def _load_preprocess_config(data: dict[str, Any]) -> PreprocessConfig:
    try:
        image_size = int(data["image_size"])
        pad_values = tuple(int(value) for value in data["pad_color_rgb"])
        mean = np.asarray(data["normalize_mean"], dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(data["normalize_std"], dtype=np.float32).reshape(1, 1, 3)
    except (KeyError, TypeError, ValueError) as error:
        raise ModelError(f"invalid preprocessing metadata: {error}") from error
    if image_size <= 0 or len(pad_values) != 3 or np.any(std == 0):
        raise ModelError("invalid preprocessing dimensions, pad color, or standard deviation")
    if data.get("color_order") != "RGB" or data.get("input_layout") != "BCHW":
        raise ModelError("unsupported OppaiOracle preprocessing format")
    return PreprocessConfig(
        image_size=image_size,
        pad_color=(pad_values[0], pad_values[1], pad_values[2]),
        mean=mean,
        std=std,
    )


def _load_vocabulary(
    vocabulary_path: Path,
    model_config: dict[str, Any],
    model_metadata: dict[str, str] | None = None,
) -> tuple[list[str], set[int]]:
    try:
        vocabulary_bytes = vocabulary_path.read_bytes()
    except OSError as error:
        raise ModelError(f"failed to read vocabulary {vocabulary_path}: {error}") from error

    actual_hash = hashlib.sha256(vocabulary_bytes).hexdigest()
    configured_hash = model_config.get("vocab_sha256")
    embedded_hash = (model_metadata or {}).get("vocab_sha256")
    embedded_payload = (model_metadata or {}).get("vocab_b64_gzip")

    if embedded_payload is not None:
        try:
            embedded_bytes = gzip.decompress(
                base64.b64decode(embedded_payload, validate=True)
            )
        except (binascii.Error, OSError, ValueError) as error:
            raise ModelError(f"invalid vocabulary embedded in ONNX model: {error}") from error
        calculated_embedded_hash = hashlib.sha256(embedded_bytes).hexdigest()
        if embedded_hash and calculated_embedded_hash != embedded_hash:
            raise ModelError(
                "ONNX-embedded vocabulary checksum mismatch: "
                f"expected {embedded_hash}, got {calculated_embedded_hash}"
            )
        if embedded_bytes != vocabulary_bytes:
            raise ModelError(
                "downloaded vocabulary does not match the vocabulary embedded in ONNX model"
            )
        embedded_hash = calculated_embedded_hash

    expected_hash = embedded_hash or configured_hash
    if expected_hash and actual_hash != expected_hash:
        raise ModelError(
            f"vocabulary checksum mismatch: expected {expected_hash}, got {actual_hash}"
        )
    if configured_hash and configured_hash != actual_hash:
        if embedded_hash == actual_hash:
            warnings.warn(
                "config.json contains a stale vocabulary checksum; "
                "the downloaded vocabulary exactly matches the copy embedded in the ONNX model",
                stacklevel=2,
            )
        else:
            raise ModelError(
                f"vocabulary checksum mismatch: expected {configured_hash}, got {actual_hash}"
            )

    vocabulary_data = _read_json(vocabulary_path)
    tag_to_index = vocabulary_data.get("tag_to_index")
    if not isinstance(tag_to_index, dict):
        raise ModelError("vocabulary.json does not contain tag_to_index")
    try:
        indexed = {int(index): str(tag) for tag, index in tag_to_index.items()}
    except (TypeError, ValueError) as error:
        raise ModelError(f"invalid vocabulary index: {error}") from error
    expected_count = int(model_config.get("num_labels", len(indexed)))
    if set(indexed) != set(range(expected_count)):
        raise ModelError("vocabulary indices are not contiguous or do not match num_labels")
    names = [indexed[index] for index in range(expected_count)]
    skipped = {
        index
        for index, name in enumerate(names)
        if name in {"<PAD>", "<UNK>"}
    }
    return names, skipped


def _has_transparency(image: Image.Image) -> bool:
    return image.mode in {"RGBA", "LA"} or "transparency" in image.info


def _to_rgb(image: Image.Image, pad_color: tuple[int, int, int]) -> Image.Image:
    if not _has_transparency(image):
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (*pad_color, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def preprocess_image(
    image: Image.Image,
    config: PreprocessConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the model's downscale-only letterbox preprocessing."""

    rgb = _to_rgb(image, config.pad_color)
    width, height = rgb.size
    if width <= 0 or height <= 0:
        raise ValueError("image has invalid dimensions")
    ratio = min(config.image_size / width, config.image_size / height)
    scale = min(1.0, ratio)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    if (resized_width, resized_height) != (width, height):
        rgb = rgb.resize((resized_width, resized_height), Image.Resampling.BILINEAR)

    canvas = np.full(
        (config.image_size, config.image_size, 3),
        config.pad_color,
        dtype=np.uint8,
    )
    top = (config.image_size - resized_height) // 2
    left = (config.image_size - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = np.asarray(
        rgb,
        dtype=np.uint8,
    )
    padding_mask = np.ones((config.image_size, config.image_size), dtype=np.bool_)
    padding_mask[top : top + resized_height, left : left + resized_width] = False

    pixels = canvas.astype(np.float32) / 255.0
    pixels = (pixels - config.mean) / config.std
    return pixels.transpose(2, 0, 1), padding_mask


class OppaiOracleTagger:
    """OppaiOracle V1/V1.1 inference through ONNX Runtime."""

    name = "oppai-oracle"
    default_threshold = 0.65
    default_max_tags = 50

    def __init__(self, *, variant: str = "v1.1", device: str = "auto") -> None:
        try:
            self._spec = VARIANTS[variant]
        except KeyError as error:
            raise InputError(f"unknown OppaiOracle variant: {variant}") from error
        self._device_request = device
        self._session: ort.InferenceSession | None = None
        self._preprocess: PreprocessConfig | None = None
        self._tag_names: list[str] = []
        self._skipped_indices: set[int] = set()
        self._primary_input = ""
        self._mask_input: str | None = None
        self._output_name: str | None = None
        self.active_device = "unloaded"

    @property
    def vocabulary(self) -> Sequence[str]:
        return self._tag_names

    def load(self) -> None:
        if self._session is not None:
            return
        assets = _download_assets(self._spec)
        model_config = _read_json(assets.config)
        preprocessing_data = _read_json(assets.preprocessing)
        self._preprocess = _load_preprocess_config(preprocessing_data)

        self._session, self.active_device = create_session(
            assets.model,
            self._device_request,
        )
        try:
            self._inspect_session(model_config)
            try:
                model_metadata = dict(
                    self._session.get_modelmeta().custom_metadata_map
                )
            except Exception as error:
                raise ModelError(f"failed to read ONNX model metadata: {error}") from error
            self._tag_names, self._skipped_indices = _load_vocabulary(
                assets.vocabulary,
                model_config,
                model_metadata,
            )
        except Exception:
            self._session = None
            self.active_device = "unloaded"
            raise

    def _inspect_session(self, model_config: dict[str, Any]) -> None:
        assert self._session is not None
        inputs = {item.name: item for item in self._session.get_inputs()}
        expected_inputs = model_config.get("onnx_inputs", [])
        if not all(name in inputs for name in expected_inputs):
            raise ModelError(
                f"ONNX inputs do not match config: expected {expected_inputs}, got {list(inputs)}"
            )
        self._primary_input = (
            "pixel_values" if "pixel_values" in inputs else next(iter(inputs))
        )
        self._mask_input = "padding_mask" if "padding_mask" in inputs else None
        outputs = [item.name for item in self._session.get_outputs()]
        configured_outputs = model_config.get("onnx_outputs", [])
        if configured_outputs and not all(name in outputs for name in configured_outputs):
            raise ModelError(
                f"ONNX outputs do not match config: expected {configured_outputs}, got {outputs}"
            )
        self._output_name = "probabilities" if "probabilities" in outputs else outputs[0]

    def predict_batch(
        self,
        images: Sequence[Image.Image],
        *,
        threshold: float,
        max_tags: int | None,
    ) -> list[list[TagPrediction]]:
        if self._session is None:
            self.load()
        assert self._session is not None
        assert self._preprocess is not None

        processed = [preprocess_image(image, self._preprocess) for image in images]
        pixels = np.stack([item[0] for item in processed], axis=0)
        masks = np.stack([item[1] for item in processed], axis=0)
        feed = {self._primary_input: pixels}
        if self._mask_input is not None:
            feed[self._mask_input] = masks
        try:
            probabilities = self._session.run([self._output_name], feed)[0]
        except Exception as error:
            raise ModelError(
                f"OppaiOracle inference failed on {self.active_device}: {error}"
            ) from error
        if probabilities.shape != (len(images), len(self._tag_names)):
            raise ModelError(
                "unexpected model output shape: "
                f"{probabilities.shape}, expected {(len(images), len(self._tag_names))}"
            )

        results: list[list[TagPrediction]] = []
        for scores in probabilities:
            order = np.argsort(-scores, kind="stable")
            predictions: list[TagPrediction] = []
            for raw_index in order:
                index = int(raw_index)
                if index in self._skipped_indices:
                    continue
                confidence = float(scores[index])
                if confidence < threshold:
                    break
                predictions.append(
                    TagPrediction(
                        name=self._tag_names[index],
                        confidence=confidence,
                    )
                )
                if max_tags is not None and len(predictions) >= max_tags:
                    break
            results.append(predictions)
        return results
