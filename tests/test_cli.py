from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from PIL import Image

import nekotagger.cli
from nekotagger.cli import main
from nekotagger.tag_filters import NAMED_CATEGORIES, TagCategory
from nekotagger.tag_metadata import CategoryLookup
from nekotagger.types import TagPrediction


class FakeTagger:
    name = "fake"
    default_threshold = 0.65
    default_max_tags = 50
    active_device = "cpu"

    @property
    def vocabulary(self) -> Sequence[str]:
        return ("long_hair", "1girl")

    def load(self) -> None:
        return None

    def predict_batch(
        self,
        images: Sequence[Image.Image],
        *,
        threshold: float,
        max_tags: int | None,
    ) -> list[list[TagPrediction]]:
        predictions = [
            TagPrediction("long_hair", 0.9),
            TagPrediction("1girl", 0.8),
        ]
        if max_tags is not None:
            predictions = predictions[:max_tags]
        return [predictions for _ in images]


@pytest.fixture(autouse=True)
def fake_tagger(monkeypatch: pytest.MonkeyPatch) -> None:
    def factory(name: str, *, variant: str, device: str) -> FakeTagger:
        assert name == "oppai-oracle"
        assert variant in {"v1", "v1.1"}
        assert device
        return FakeTagger()

    monkeypatch.setattr(nekotagger.cli, "create_tagger", factory)


def _save_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 8), "purple").save(path)


def test_single_image_prints_only_tags_to_stdout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    image = tmp_path / "a.png"
    _save_image(image)
    assert main([str(image)]) == 0
    captured = capsys.readouterr()
    assert captured.out == "long hair, 1girl\n"
    assert "Using oppai-oracle v1.1 on cpu" in captured.err


def test_single_image_can_keep_underscores(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    image = tmp_path / "a.png"
    _save_image(image)
    assert main([str(image), "--keep-underscores", "--max-tags", "1"]) == 0
    assert capsys.readouterr().out == "long_hair\n"


def test_exact_exclusion_is_applied_before_max_tags(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    image = tmp_path / "a.png"
    _save_image(image)
    assert main(
        [str(image), "--exclude-tag", "long hair", "--max-tags", "1"]
    ) == 0
    assert capsys.readouterr().out == "1girl\n"


def test_excluded_tags_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    image = tmp_path / "a.png"
    exclusions = tmp_path / "exclude.txt"
    _save_image(image)
    exclusions.write_text("# custom exclusions\nlong_hair\n", encoding="utf-8")
    assert main(
        [str(image), "--exclude-tags-file", str(exclusions)]
    ) == 0
    assert capsys.readouterr().out == "1girl\n"


def test_category_exclusion_uses_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    image = tmp_path / "a.png"
    metadata = tmp_path / "tags.parquet"
    _save_image(image)
    metadata.touch()

    def lookup(
        vocabulary: Sequence[str],
        categories: set[TagCategory],
        *,
        metadata_path: Path | None,
    ) -> CategoryLookup:
        assert tuple(vocabulary) == ("long_hair", "1girl")
        assert categories == {TagCategory.CHARACTER}
        assert metadata_path == metadata
        return CategoryLookup(
            excluded_tags=frozenset({"long_hair"}),
            matched_count=2,
            vocabulary_count=2,
            source_path=metadata,
        )

    monkeypatch.setattr(nekotagger.cli, "lookup_category_exclusions", lookup)
    assert main(
        [
            str(image),
            "--exclude-category",
            "character",
            "--tag-metadata",
            str(metadata),
        ]
    ) == 0
    captured = capsys.readouterr()
    assert captured.out == "1girl\n"
    assert "matched 2/2" in captured.err


def test_exclude_named_expands_all_named_categories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "a.png"
    _save_image(image)

    def lookup(
        vocabulary: Sequence[str],
        categories: set[TagCategory],
        *,
        metadata_path: Path | None,
    ) -> CategoryLookup:
        assert categories == set(NAMED_CATEGORIES)
        assert metadata_path is None
        return CategoryLookup(
            excluded_tags=frozenset(),
            matched_count=len(vocabulary),
            vocabulary_count=len(vocabulary),
            source_path=tmp_path / "cached.parquet",
        )

    monkeypatch.setattr(nekotagger.cli, "lookup_category_exclusions", lookup)
    assert main([str(image), "--exclude-named"]) == 0


def test_single_image_rejects_output_options(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    image = tmp_path / "a.png"
    _save_image(image)
    assert main([str(image), "--same-dir"]) == 2
    assert "always written to stdout" in capsys.readouterr().err


def test_directory_mirrors_relative_structure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "data" / "images"
    image = input_root / "set" / "a.png"
    work = tmp_path / "work"
    work.mkdir()
    _save_image(image)
    monkeypatch.chdir(work)
    assert main([str(input_root), "--recursive"]) == 0
    assert (work / "set" / "a.txt").read_text(encoding="utf-8") == "long hair, 1girl\n"


def test_directory_numbered_policy_preserves_existing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "images"
    image = input_root / "a.png"
    work = tmp_path / "work"
    work.mkdir()
    _save_image(image)
    existing = work / "a.txt"
    existing.write_text("original\n", encoding="utf-8")
    monkeypatch.chdir(work)
    assert main([str(input_root), "--numbered"]) == 0
    assert existing.read_text(encoding="utf-8") == "original\n"
    assert (work / "a_1.txt").read_text(encoding="utf-8") == "long hair, 1girl\n"


def test_bad_image_does_not_stop_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_root = tmp_path / "images"
    good = input_root / "a.png"
    bad = input_root / "b.jpg"
    work = tmp_path / "work"
    work.mkdir()
    _save_image(good)
    bad.write_bytes(b"not an image")
    monkeypatch.chdir(work)
    assert main([str(input_root)]) == 1
    assert (work / "a.txt").exists()
    assert "b.jpg: image load failed" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--threshold", "1.1"], "--threshold must be between 0 and 1"),
        (["--max-tags", "-1"], "--max-tags must be 0 or greater"),
        (["--batch-size", "0"], "--batch-size must be greater than 0"),
        (
            ["--tag-metadata", "tags.parquet"],
            "--tag-metadata requires --exclude-category or --exclude-named",
        ),
    ],
)
def test_numeric_validation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    message: str,
) -> None:
    image = tmp_path / "a.png"
    _save_image(image)
    assert main([str(image), *arguments]) == 2
    assert message in capsys.readouterr().err
