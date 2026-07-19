"""Shared types forming the extension boundary for image taggers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from PIL import Image


@dataclass(frozen=True, slots=True)
class TagPrediction:
    """A tag name and its confidence score."""

    name: str
    confidence: float


class Tagger(Protocol):
    """Interface implemented by model-specific tagger adapters."""

    name: str
    default_threshold: float
    default_max_tags: int
    active_device: str

    def load(self) -> None:
        """Download assets if needed and initialize the inference backend."""

    def predict_batch(
        self,
        images: Sequence[Image.Image],
        *,
        threshold: float,
        max_tags: int | None,
    ) -> list[list[TagPrediction]]:
        """Return confidence-sorted predictions for every image."""
