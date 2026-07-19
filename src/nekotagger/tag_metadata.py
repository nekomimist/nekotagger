"""Category metadata backed by a cached Danbooru tag dictionary snapshot."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError

from nekotagger.errors import InputError, MetadataError
from nekotagger.tag_filters import TagCategory

METADATA_REPO = "ThetaCursed/danbooru-2026-clean-metadata"
METADATA_REVISION = "1c24eca1b94ba0f7b93c02d50927c1ed86b86820"
METADATA_FILENAME = "tags_dictionary.parquet"
REQUIRED_COLUMNS = frozenset({"tag", "category"})


@dataclass(frozen=True, slots=True)
class CategoryLookup:
    """Category matches for one model vocabulary."""

    excluded_tags: frozenset[str]
    matched_count: int
    vocabulary_count: int
    source_path: Path

    @property
    def unknown_count(self) -> int:
        return self.vocabulary_count - self.matched_count


def _download_metadata() -> Path:
    try:
        return Path(
            hf_hub_download(
                repo_id=METADATA_REPO,
                repo_type="dataset",
                filename=METADATA_FILENAME,
                revision=METADATA_REVISION,
                local_files_only=True,
            )
        )
    except LocalEntryNotFoundError:
        try:
            return Path(
                hf_hub_download(
                    repo_id=METADATA_REPO,
                    repo_type="dataset",
                    filename=METADATA_FILENAME,
                    revision=METADATA_REVISION,
                )
            )
        except Exception as error:
            raise MetadataError(f"failed to download tag metadata: {error}") from error
    except Exception as error:
        raise MetadataError(f"failed to access cached tag metadata: {error}") from error


def _metadata_path(path: Path | None) -> Path:
    if path is None:
        return _download_metadata()
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise InputError(f"tag metadata file does not exist: {resolved}")
    return resolved


def lookup_category_exclusions(
    vocabulary: Sequence[str],
    excluded_categories: Collection[TagCategory],
    *,
    metadata_path: Path | None = None,
) -> CategoryLookup:
    """Find vocabulary tags belonging to selected categories."""

    source_path = _metadata_path(metadata_path)
    try:
        parquet = pq.ParquetFile(source_path)
        columns = frozenset(parquet.schema_arrow.names)
        missing = REQUIRED_COLUMNS - columns
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise MetadataError(
                f"tag metadata is missing required column(s): {missing_text}"
            )
        table = parquet.read(columns=["tag", "category"])
    except MetadataError:
        raise
    except Exception as error:
        raise MetadataError(f"failed to read tag metadata {source_path}: {error}") from error

    normalized_vocabulary = frozenset(name.casefold() for name in vocabulary)
    try:
        vocabulary_values = pa.array(
            normalized_vocabulary,
            type=pa.large_string(),
        )
        matching = table.filter(
            pc.is_in(table["tag"], value_set=vocabulary_values)
        )
        matched_tags = matching["tag"].to_pylist()
        category_values = pa.array(
            [category.value for category in excluded_categories],
            type=pa.large_string(),
        )
        excluded = matching.filter(
            pc.is_in(matching["category"], value_set=category_values)
        )["tag"].to_pylist()
    except Exception as error:
        raise MetadataError(f"failed to query tag metadata {source_path}: {error}") from error

    return CategoryLookup(
        excluded_tags=frozenset(str(tag).casefold() for tag in excluded),
        matched_count=len(set(matched_tags)),
        vocabulary_count=len(normalized_vocabulary),
        source_path=source_path,
    )
