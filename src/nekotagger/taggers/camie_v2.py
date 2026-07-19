"""Independent ONNX Runtime adapter for Camais03/camie-tagger-v2."""

from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
from PIL import Image

from nekotagger.errors import ModelError
from nekotagger.taggers.onnx_runtime import create_session
from nekotagger.types import TagPrediction

MODEL_REPO = "Camais03/camie-tagger-v2"
MODEL_REVISION = "7d40c1b85b86ab4f607b2caf26b1b50c99db743e"
MODEL_FILENAME = "camie-tagger-v2.onnx"
METADATA_FILENAME = "camie-tagger-v2-metadata.json"
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
PADDING_COLOR = (124, 116, 104)


@dataclass(frozen=True, slots=True)
class ModelAssets:
    model: Path
    metadata: Path


@dataclass(frozen=True, slots=True)
class CamieMetadata:
    image_size: int
    tag_names: tuple[str, ...]
    tag_categories: tuple[str, ...]
    category_names: frozenset[str]


def _download_file(filename: str) -> Path:
    try:
        return Path(
            hf_hub_download(
                repo_id=MODEL_REPO,
                filename=filename,
                revision=MODEL_REVISION,
                local_files_only=True,
            )
        )
    except LocalEntryNotFoundError:
        try:
            return Path(
                hf_hub_download(
                    repo_id=MODEL_REPO,
                    filename=filename,
                    revision=MODEL_REVISION,
                )
            )
        except Exception as error:
            raise ModelError(f"failed to download {filename}: {error}") from error
    except Exception as error:
        raise ModelError(f"failed to access cached {filename}: {error}") from error


def _download_assets() -> ModelAssets:
    return ModelAssets(
        model=_download_file(MODEL_FILENAME),
        metadata=_download_file(METADATA_FILENAME),
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


def _load_metadata(path: Path) -> CamieMetadata:
    data = _read_json(path)
    try:
        model_info = data["model_info"]
        input_spec = data["input_spec"]
        dataset_info = data["dataset_info"]
        tag_mapping = dataset_info["tag_mapping"]
        image_size = int(model_info["img_size"])
        total_tags = int(dataset_info["total_tags"])
        raw_indices = tag_mapping["idx_to_tag"]
        raw_categories = tag_mapping["tag_to_category"]
        declared_categories = frozenset(str(item) for item in dataset_info["categories"])
    except (KeyError, TypeError, ValueError) as error:
        raise ModelError(f"invalid Camie metadata structure: {error}") from error

    if (
        not isinstance(model_info, dict)
        or not isinstance(input_spec, dict)
        or not isinstance(dataset_info, dict)
        or not isinstance(tag_mapping, dict)
        or not isinstance(raw_indices, dict)
        or not isinstance(raw_categories, dict)
    ):
        raise ModelError("Camie metadata sections and tag mappings must be JSON objects")
    try:
        indexed = {int(index): str(name) for index, name in raw_indices.items()}
    except (TypeError, ValueError) as error:
        raise ModelError(f"invalid Camie tag index: {error}") from error
    if total_tags <= 0 or set(indexed) != set(range(total_tags)):
        raise ModelError("Camie tag indices are not contiguous or do not match total_tags")

    tag_names = tuple(indexed[index] for index in range(total_tags))
    if len(set(tag_names)) != total_tags:
        raise ModelError("Camie metadata contains duplicate tag names")
    try:
        tag_categories = tuple(str(raw_categories[name]) for name in tag_names)
    except KeyError as error:
        raise ModelError(f"Camie metadata has no category for tag {error.args[0]!r}") from error
    actual_categories = frozenset(tag_categories)
    if not actual_categories or actual_categories != declared_categories:
        raise ModelError(
            "Camie declared categories do not match its tag category mapping"
        )

    expected_shape = [3, image_size, image_size]
    if (
        image_size <= 0
        or input_spec.get("format") != "RGB image tensor"
        or input_spec.get("dtype") != "float32"
        or input_spec.get("shape") != expected_shape
    ):
        raise ModelError("unsupported Camie input specification")
    return CamieMetadata(
        image_size=image_size,
        tag_names=tag_names,
        tag_categories=tag_categories,
        category_names=actual_categories,
    )


def _has_transparency(image: Image.Image) -> bool:
    return image.mode in {"RGBA", "LA"} or "transparency" in image.info


def _to_rgb(image: Image.Image) -> Image.Image:
    if not _has_transparency(image):
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (*PADDING_COLOR, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def preprocess_image(image: Image.Image, image_size: int) -> np.ndarray:
    """Resize with aspect ratio padding and apply ImageNet normalization."""

    rgb = _to_rgb(image)
    width, height = rgb.size
    if width <= 0 or height <= 0:
        raise ValueError("image has invalid dimensions")
    scale = min(image_size / width, image_size / height)
    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))
    if (resized_width, resized_height) != (width, height):
        rgb = rgb.resize(
            (resized_width, resized_height),
            Image.Resampling.LANCZOS,
        )

    canvas = np.full(
        (image_size, image_size, 3),
        PADDING_COLOR,
        dtype=np.uint8,
    )
    top = (image_size - resized_height) // 2
    left = (image_size - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = np.asarray(
        rgb,
        dtype=np.uint8,
    )
    pixels = canvas.astype(np.float32) / 255.0
    pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD
    return pixels.transpose(2, 0, 1)


class CamieV2Tagger:
    """Camie Tagger v2 inference through ONNX Runtime."""

    name = "camie-v2"
    default_threshold = 0.492
    default_max_tags = 50

    def __init__(self, *, device: str = "auto") -> None:
        self._device_request = device
        self._session: ort.InferenceSession | None = None
        self._metadata: CamieMetadata | None = None
        self._input_name = ""
        self._output_name = ""
        self.active_device = "unloaded"

    @property
    def vocabulary(self) -> Sequence[str]:
        if self._metadata is None:
            return ()
        return self._metadata.tag_names

    @property
    def category_names(self) -> Collection[str]:
        if self._metadata is None:
            return frozenset()
        return self._metadata.category_names

    def tags_in_categories(self, categories: Collection[str]) -> Collection[str]:
        if self._metadata is None:
            raise ModelError("Camie metadata is not loaded")
        requested = frozenset(categories)
        return frozenset(
            name
            for name, category in zip(
                self._metadata.tag_names,
                self._metadata.tag_categories,
                strict=True,
            )
            if category in requested
        )

    def load(self) -> None:
        if self._session is not None:
            return
        assets = _download_assets()
        metadata = _load_metadata(assets.metadata)
        session, active_device = create_session(assets.model, self._device_request)
        try:
            self._inspect_session(session, metadata)
        except Exception:
            self.active_device = "unloaded"
            raise
        self._metadata = metadata
        self._session = session
        self.active_device = active_device

    def _inspect_session(
        self,
        session: ort.InferenceSession,
        metadata: CamieMetadata,
    ) -> None:
        inputs = session.get_inputs()
        if len(inputs) != 1:
            raise ModelError(f"Camie ONNX model has {len(inputs)} inputs; expected 1")
        input_shape = inputs[0].shape
        if (
            len(input_shape) != 4
            or input_shape[1] not in {3, "3"}
            or input_shape[2] not in {metadata.image_size, str(metadata.image_size)}
            or input_shape[3] not in {metadata.image_size, str(metadata.image_size)}
        ):
            raise ModelError(f"unexpected Camie ONNX input shape: {input_shape}")
        self._input_name = inputs[0].name

        outputs = session.get_outputs()
        if len(outputs) < 2:
            raise ModelError(
                f"Camie ONNX model has {len(outputs)} output(s); expected at least 2"
            )
        refined = next(
            (output for output in outputs if output.name == "refined_predictions"),
            outputs[1],
        )
        if (
            len(refined.shape) != 2
            or refined.shape[1] not in {len(metadata.tag_names), str(len(metadata.tag_names))}
        ):
            raise ModelError(f"unexpected Camie refined output shape: {refined.shape}")
        self._output_name = refined.name

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
        assert self._metadata is not None

        pixels = np.stack(
            [
                preprocess_image(image, self._metadata.image_size)
                for image in images
            ],
            axis=0,
        )
        try:
            logits = self._session.run(
                [self._output_name],
                {self._input_name: pixels},
            )[0]
        except Exception as error:
            raise ModelError(
                f"Camie v2 inference failed on {self.active_device}: {error}"
            ) from error
        expected_shape = (len(images), len(self._metadata.tag_names))
        if logits.shape != expected_shape:
            raise ModelError(
                f"unexpected model output shape: {logits.shape}, expected {expected_shape}"
            )
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))

        results: list[list[TagPrediction]] = []
        for scores in probabilities:
            order = np.argsort(-scores, kind="stable")
            predictions: list[TagPrediction] = []
            for raw_index in order:
                index = int(raw_index)
                confidence = float(scores[index])
                if confidence < threshold:
                    break
                predictions.append(
                    TagPrediction(
                        name=self._metadata.tag_names[index],
                        confidence=confidence,
                    )
                )
                if max_tags is not None and len(predictions) >= max_tags:
                    break
            results.append(predictions)
        return results
