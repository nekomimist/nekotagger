"""Common post-inference tag filtering."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from nekotagger.errors import InputError
from nekotagger.types import TagPrediction


class TagCategory(str, Enum):
    GENERAL = "general"
    ARTIST = "artist"
    COPYRIGHT = "copyright"
    CHARACTER = "character"
    META = "meta"
    RATING = "rating"
    YEAR = "year"


NAMED_CATEGORIES = frozenset(
    {
        TagCategory.ARTIST,
        TagCategory.COPYRIGHT,
        TagCategory.CHARACTER,
    }
)


def normalize_tag_name(name: str) -> str:
    """Normalize user input to the raw lowercase underscore form."""

    return name.strip().replace(" ", "_").casefold()


def load_excluded_tags(
    inline_tags: Iterable[str],
    files: Iterable[Path],
) -> frozenset[str]:
    """Load exact tag exclusions from CLI values and UTF-8 line files."""

    tags = {
        normalized
        for tag in inline_tags
        if (normalized := normalize_tag_name(tag))
    }
    for path in files:
        resolved = path.expanduser().resolve()
        try:
            lines = resolved.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise InputError(f"failed to read excluded tags file {resolved}: {error}") from error
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tags.add(normalize_tag_name(stripped))
    return frozenset(tags)


@dataclass(frozen=True, slots=True)
class PredictionFilter:
    """Exclude raw tag names, then apply the requested result limit."""

    excluded_tags: frozenset[str]
    max_tags: int | None

    @property
    def has_exclusions(self) -> bool:
        return bool(self.excluded_tags)

    def apply(
        self,
        predictions: Sequence[TagPrediction],
    ) -> list[TagPrediction]:
        kept: list[TagPrediction] = []
        for prediction in predictions:
            if normalize_tag_name(prediction.name) in self.excluded_tags:
                continue
            kept.append(prediction)
            if self.max_tags is not None and len(kept) >= self.max_tags:
                break
        return kept
