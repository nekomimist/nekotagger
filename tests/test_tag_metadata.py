from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import nekotagger.tag_metadata
from nekotagger.errors import MetadataError
from nekotagger.tag_filters import TagCategory
from nekotagger.tag_metadata import _download_metadata, lookup_category_exclusions


def _write_metadata(path: Path) -> None:
    table = pa.table(
        {
            "tag": pa.array(
                ["long_hair", "hatsune_miku", "kantoku", "touhou"],
                type=pa.large_string(),
            ),
            "category": pa.array(
                ["general", "character", "artist", "copyright"],
                type=pa.large_string(),
            ),
            "count": [1, 2, 3, 4],
        }
    )
    pq.write_table(table, path)


def test_lookup_category_exclusions_keeps_unknown_tags(tmp_path: Path) -> None:
    path = tmp_path / "tags.parquet"
    _write_metadata(path)
    lookup = lookup_category_exclusions(
        ["long_hair", "hatsune_miku", "kantoku", "missing_tag"],
        {TagCategory.CHARACTER, TagCategory.ARTIST},
        metadata_path=path,
    )
    assert lookup.excluded_tags == frozenset({"hatsune_miku", "kantoku"})
    assert lookup.matched_count == 3
    assert lookup.vocabulary_count == 4
    assert lookup.unknown_count == 1


def test_metadata_requires_tag_and_category_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.parquet"
    pq.write_table(pa.table({"name": ["tag"]}), path)
    with pytest.raises(MetadataError, match="category, tag"):
        lookup_category_exclusions(
            ["tag"],
            {TagCategory.CHARACTER},
            metadata_path=path,
        )


def test_download_uses_cached_snapshot_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "cached.parquet"
    path.touch()
    calls: list[dict[str, object]] = []

    def download(**arguments: object) -> str:
        calls.append(arguments)
        return str(path)

    monkeypatch.setattr(nekotagger.tag_metadata, "hf_hub_download", download)
    assert _download_metadata() == path
    assert len(calls) == 1
    assert calls[0]["local_files_only"] is True
