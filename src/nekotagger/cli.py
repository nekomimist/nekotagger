"""Command-line entry point for nekotagger."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TypeVar

from PIL import Image
from tqdm import tqdm

from nekotagger import __version__
from nekotagger.errors import InputError, ModelError, NekoTaggerError
from nekotagger.images import discover_images, load_image
from nekotagger.output import (
    ConflictMode,
    OutputJob,
    plan_output_jobs,
    resolve_output_conflicts,
    write_text_atomic,
)
from nekotagger.tag_filters import (
    NAMED_CATEGORIES,
    PredictionFilter,
    TagCategory,
    load_excluded_tags,
    normalize_tag_name,
)
from nekotagger.tag_metadata import lookup_category_exclusions
from nekotagger.taggers.registry import create_tagger, tagger_names
from nekotagger.types import Tagger, TagPrediction, VocabularyProvider

T = TypeVar("T")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nekotagger",
        description="Tag one image or a directory tree with an extensible image tagger.",
    )
    parser.add_argument("input", type=Path, help="an image file or image directory")
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="search subdirectories when INPUT is a directory",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="directory output root (default: current working directory)",
    )
    output_group.add_argument(
        "--same-dir",
        action="store_true",
        help="write each .txt next to its source image",
    )
    conflict_group = parser.add_mutually_exclusive_group()
    conflict_group.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing .txt files without prompting",
    )
    conflict_group.add_argument(
        "--numbered",
        action="store_true",
        help="preserve existing files and use _1, _2, ... names",
    )
    parser.add_argument(
        "--tagger",
        choices=tagger_names(),
        default="oppai-oracle",
        help="tagger adapter (default: %(default)s)",
    )
    parser.add_argument(
        "--variant",
        choices=("v1.1", "v1"),
        default="v1.1",
        help="OppaiOracle variant (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        metavar="DEVICE",
        help="auto, cpu, cuda, or cuda:N (default: %(default)s)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="minimum confidence (OppaiOracle default: 0.65)",
    )
    parser.add_argument(
        "--max-tags",
        type=int,
        help="maximum tags per image; 0 means unlimited (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="inference batch size (default: 4 on CUDA, 1 on CPU)",
    )
    parser.add_argument(
        "--keep-underscores",
        action="store_true",
        help="keep raw tag underscores instead of converting them to spaces",
    )
    parser.add_argument(
        "--exclude-category",
        action="append",
        choices=tuple(category.value for category in TagCategory),
        default=[],
        metavar="CATEGORY",
        help=(
            "exclude a tag category; repeat for multiple categories "
            "(general, artist, copyright, character, meta)"
        ),
    )
    parser.add_argument(
        "--exclude-named",
        action="store_true",
        help="exclude artist, copyright, and character tags",
    )
    parser.add_argument(
        "--exclude-tag",
        action="append",
        default=[],
        metavar="TAG",
        help="exclude one exact raw tag; repeat for multiple tags",
    )
    parser.add_argument(
        "--exclude-tags-file",
        action="append",
        type=Path,
        default=[],
        metavar="FILE",
        help="exclude raw tags listed one per line in a UTF-8 file",
    )
    parser.add_argument(
        "--tag-metadata",
        type=Path,
        metavar="PARQUET",
        help="use a local tags_dictionary.parquet for category filtering",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def _validate_numeric_options(args: argparse.Namespace) -> None:
    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        raise InputError("--threshold must be between 0 and 1")
    if args.max_tags is not None and args.max_tags < 0:
        raise InputError("--max-tags must be 0 or greater")
    if args.batch_size is not None and args.batch_size <= 0:
        raise InputError("--batch-size must be greater than 0")
    if args.tag_metadata is not None and not (
        args.exclude_category or args.exclude_named
    ):
        raise InputError(
            "--tag-metadata requires --exclude-category or --exclude-named"
        )


def _conflict_mode(args: argparse.Namespace) -> ConflictMode:
    if args.overwrite:
        return ConflictMode.OVERWRITE
    if args.numbered:
        return ConflictMode.NUMBERED
    return ConflictMode.ASK


def _format_predictions(
    predictions: Sequence[TagPrediction],
    *,
    keep_underscores: bool,
) -> str:
    names = [
        prediction.name if keep_underscores else prediction.name.replace("_", " ")
        for prediction in predictions
    ]
    return ", ".join(names)


def _build_prediction_filter(
    args: argparse.Namespace,
    tagger: Tagger,
    max_tags: int | None,
) -> PredictionFilter:
    excluded_tags = set(
        load_excluded_tags(args.exclude_tag, args.exclude_tags_file)
    )
    vocabulary = (
        tagger.vocabulary
        if isinstance(tagger, VocabularyProvider)
        else None
    )
    categories = {
        TagCategory(category_name)
        for category_name in args.exclude_category
    }
    if args.exclude_named:
        categories.update(NAMED_CATEGORIES)
    if categories:
        if vocabulary is None:
            raise InputError(
                f"tagger {args.tagger} does not expose a vocabulary "
                "required for category filtering"
            )
        lookup = lookup_category_exclusions(
            vocabulary,
            categories,
            metadata_path=args.tag_metadata,
        )
        excluded_tags.update(lookup.excluded_tags)
        category_text = ", ".join(
            category.value for category in sorted(categories, key=lambda item: item.value)
        )
        print(
            f"Tag metadata matched {lookup.matched_count}/{lookup.vocabulary_count} "
            f"vocabulary tags; {lookup.unknown_count} unknown tag(s) will be kept.",
            file=sys.stderr,
        )
        print(
            f"Excluding {len(lookup.excluded_tags)} tag(s) in categories: "
            f"{category_text}",
            file=sys.stderr,
        )
    if vocabulary is not None:
        normalized_vocabulary = {
            normalize_tag_name(tag_name)
            for tag_name in vocabulary
        }
        excluded_tags.intersection_update(normalized_vocabulary)
    return PredictionFilter(
        excluded_tags=frozenset(excluded_tags),
        max_tags=max_tags,
    )


def _load_tagger(
    args: argparse.Namespace,
) -> tuple[Tagger, float, PredictionFilter, int]:
    tagger = create_tagger(
        args.tagger,
        variant=args.variant,
        device=args.device,
    )
    tagger.load()
    threshold = (
        tagger.default_threshold if args.threshold is None else float(args.threshold)
    )
    configured_max = (
        tagger.default_max_tags if args.max_tags is None else int(args.max_tags)
    )
    max_tags = None if configured_max == 0 else configured_max
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 4 if tagger.active_device.startswith("cuda:") else 1
    print(
        f"Using {args.tagger} {args.variant} on {tagger.active_device}",
        file=sys.stderr,
    )
    prediction_filter = _build_prediction_filter(args, tagger, max_tags)
    return tagger, threshold, prediction_filter, int(batch_size)


def _predict_batch(
    tagger: Tagger,
    images: Sequence[Image.Image],
    *,
    threshold: float,
    prediction_filter: PredictionFilter,
) -> list[list[TagPrediction]]:
    raw_predictions = tagger.predict_batch(
        images,
        threshold=threshold,
        max_tags=None if prediction_filter.has_exclusions else prediction_filter.max_tags,
    )
    return [
        prediction_filter.apply(predictions)
        for predictions in raw_predictions
    ]


def _run_single_file(path: Path, args: argparse.Namespace) -> int:
    if (
        args.recursive
        or args.output_dir is not None
        or args.same_dir
        or args.overwrite
        or args.numbered
    ):
        raise InputError(
            "directory/output/conflict options cannot be used with a single image; "
            "single-image tags are always written to stdout"
        )
    try:
        image = load_image(path)
    except Exception as error:
        raise InputError(f"failed to open image {path}: {error}") from error
    try:
        tagger, threshold, prediction_filter, _ = _load_tagger(args)
        predictions = _predict_batch(
            tagger,
            [image],
            threshold=threshold,
            prediction_filter=prediction_filter,
        )
    finally:
        image.close()
    if len(predictions) != 1:
        raise ModelError(f"tagger returned {len(predictions)} results for one image")
    print(
        _format_predictions(
            predictions[0],
            keep_underscores=args.keep_underscores,
        )
    )
    return 0


def _iter_batches(items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for offset in range(0, len(items), size):
        yield items[offset : offset + size]


def _run_directory(root: Path, args: argparse.Namespace) -> int:
    sources = discover_images(root, recursive=args.recursive)
    if not sources:
        scope = "recursively" if args.recursive else "in the top level"
        raise InputError(f"no supported images found {scope} of {root}")

    output_root = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else Path.cwd().resolve()
    )
    jobs = plan_output_jobs(
        sources,
        input_root=root,
        output_root=output_root,
        same_dir=args.same_dir,
    )
    jobs = resolve_output_conflicts(
        jobs,
        mode=_conflict_mode(args),
        stdin=sys.stdin,
        stderr=sys.stderr,
    )

    tagger, threshold, prediction_filter, batch_size = _load_tagger(args)
    failures: list[tuple[Path, str]] = []
    written = 0
    batches = _iter_batches(jobs, batch_size)
    progress = tqdm(
        batches,
        total=(len(jobs) + batch_size - 1) // batch_size,
        unit="batch",
        desc="Tagging",
        file=sys.stderr,
        disable=not sys.stderr.isatty(),
    )
    for batch in progress:
        loaded: list[tuple[OutputJob, Image.Image]] = []
        for job in batch:
            try:
                loaded.append((job, load_image(job.source)))
            except Exception as error:
                failures.append((job.source, f"image load failed: {error}"))
        if not loaded:
            continue
        try:
            predictions = _predict_batch(
                tagger,
                [image for _, image in loaded],
                threshold=threshold,
                prediction_filter=prediction_filter,
            )
            if len(predictions) != len(loaded):
                raise ModelError(
                    f"tagger returned {len(predictions)} results for {len(loaded)} images"
                )
            for (job, _), image_predictions in zip(loaded, predictions, strict=True):
                text = _format_predictions(
                    image_predictions,
                    keep_underscores=args.keep_underscores,
                )
                try:
                    write_text_atomic(job.target, f"{text}\n")
                    written += 1
                except OSError as error:
                    failures.append((job.source, f"write to {job.target} failed: {error}"))
        finally:
            for _, image in loaded:
                image.close()

    print(f"Wrote {written} tag file(s); {len(failures)} failed.", file=sys.stderr)
    for path, message in failures:
        print(f"  {path}: {message}", file=sys.stderr)
    return 1 if failures else 0


def run(args: argparse.Namespace) -> int:
    _validate_numeric_options(args)
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise InputError(f"input does not exist: {input_path}")
    if input_path.is_file():
        return _run_single_file(input_path, args)
    if input_path.is_dir():
        return _run_directory(input_path, args)
    raise InputError(f"input is neither a regular file nor a directory: {input_path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except NekoTaggerError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2 if isinstance(error, InputError) else 1
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130
