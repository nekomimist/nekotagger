from __future__ import annotations

import io
from pathlib import Path

import pytest

from nekotagger.errors import ConflictError
from nekotagger.output import (
    ConflictMode,
    OutputJob,
    plan_output_jobs,
    resolve_output_conflicts,
    write_text_atomic,
)


class TtyStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_relative_output_structure_and_same_stem_collision(tmp_path: Path) -> None:
    root = tmp_path / "data"
    first = root / "images" / "set" / "a.jpg"
    second = root / "images" / "set" / "a.png"
    jobs = plan_output_jobs(
        [first, second],
        input_root=root,
        output_root=tmp_path / "work",
        same_dir=False,
    )
    assert [job.target for job in jobs] == [
        tmp_path / "work" / "images" / "set" / "a.txt",
        tmp_path / "work" / "images" / "set" / "a_1.txt",
    ]


def test_same_dir_output(tmp_path: Path) -> None:
    source = tmp_path / "data" / "a.webp"
    jobs = plan_output_jobs(
        [source],
        input_root=tmp_path / "data",
        output_root=tmp_path / "unused",
        same_dir=True,
    )
    assert jobs == [OutputJob(source=source, target=tmp_path / "data" / "a.txt")]


def test_numbered_conflicts_reserve_other_planned_targets(tmp_path: Path) -> None:
    base = tmp_path / "a.txt"
    sibling = tmp_path / "a_1.txt"
    base.write_text("old", encoding="utf-8")
    jobs = [
        OutputJob(tmp_path / "a.jpg", base),
        OutputJob(tmp_path / "a.png", sibling),
    ]
    resolved = resolve_output_conflicts(jobs, mode=ConflictMode.NUMBERED)
    assert [job.target for job in resolved] == [tmp_path / "a_2.txt", sibling]


def test_prompt_applies_once_to_all_conflicts(tmp_path: Path) -> None:
    targets = [tmp_path / "a.txt", tmp_path / "b.txt"]
    for target in targets:
        target.write_text("old", encoding="utf-8")
    stderr = io.StringIO()
    resolved = resolve_output_conflicts(
        [
            OutputJob(tmp_path / "a.jpg", targets[0]),
            OutputJob(tmp_path / "b.jpg", targets[1]),
        ],
        mode=ConflictMode.ASK,
        stdin=TtyStringIO("n\n"),
        stderr=stderr,
    )
    assert [job.target.name for job in resolved] == ["a_1.txt", "b_1.txt"]
    assert stderr.getvalue().count("already exist") == 1


def test_noninteractive_conflict_requires_explicit_policy(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("old", encoding="utf-8")
    with pytest.raises(ConflictError, match="--overwrite or --numbered"):
        resolve_output_conflicts(
            [OutputJob(tmp_path / "a.jpg", target)],
            mode=ConflictMode.ASK,
            stdin=io.StringIO(),
            stderr=io.StringIO(),
        )


def test_atomic_write_replaces_existing_text(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "a.txt"
    write_text_atomic(target, "first\n")
    write_text_atomic(target, "second\n")
    assert target.read_bytes() == b"second\n"
    assert not list(target.parent.glob(".*.tmp"))
