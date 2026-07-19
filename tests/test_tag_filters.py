from __future__ import annotations

from pathlib import Path

from nekotagger.tag_filters import PredictionFilter, load_excluded_tags
from nekotagger.types import TagPrediction


def test_load_excluded_tags_normalizes_and_ignores_comments(
    tmp_path: Path,
) -> None:
    path = tmp_path / "exclude.txt"
    path.write_text(
        "\n# comment\nHatsune Miku\nlong_hair\n",
        encoding="utf-8",
    )
    assert load_excluded_tags(["Blue Eyes"], [path]) == frozenset(
        {"blue_eyes", "hatsune_miku", "long_hair"}
    )


def test_prediction_filter_excludes_before_limiting() -> None:
    predictions = [
        TagPrediction("character_name", 0.99),
        TagPrediction("long_hair", 0.95),
        TagPrediction("blue_eyes", 0.90),
    ]
    tag_filter = PredictionFilter(
        excluded_tags=frozenset({"character_name"}),
        max_tags=2,
    )
    assert tag_filter.apply(predictions) == predictions[1:]
