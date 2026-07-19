"""Output path planning, conflict handling, and atomic text writes."""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import TextIO

from nekotagger.errors import ConflictError


class ConflictMode(str, Enum):
    ASK = "ask"
    OVERWRITE = "overwrite"
    NUMBERED = "numbered"


@dataclass(frozen=True, slots=True)
class OutputJob:
    source: Path
    target: Path


def numbered_path(path: Path, number: int) -> Path:
    return path.with_name(f"{path.stem}_{number}{path.suffix}")


def _next_numbered_path(path: Path, unavailable: set[Path]) -> Path:
    number = 1
    while True:
        candidate = numbered_path(path, number)
        if candidate not in unavailable and not candidate.exists():
            return candidate
        number += 1


def plan_output_jobs(
    sources: list[Path],
    *,
    input_root: Path,
    output_root: Path,
    same_dir: bool,
) -> list[OutputJob]:
    """Map images to unique .txt targets while retaining relative structure."""

    jobs: list[OutputJob] = []
    reserved: set[Path] = set()
    for source in sources:
        if same_dir:
            target = source.with_suffix(".txt")
        else:
            relative = source.relative_to(input_root).with_suffix(".txt")
            target = output_root / relative
        if target in reserved:
            target = _next_numbered_path(target, reserved)
        reserved.add(target)
        jobs.append(OutputJob(source=source, target=target))
    return jobs


def _prompt_conflict_mode(
    conflict_count: int,
    *,
    stdin: TextIO,
    stderr: TextIO,
) -> ConflictMode:
    if not stdin.isatty():
        raise ConflictError(
            f"{conflict_count} output file(s) already exist; "
            "use --overwrite or --numbered in a non-interactive environment"
        )
    while True:
        print(
            f"{conflict_count} output file(s) already exist. "
            "[o]verwrite all, [n]umber all, [a]bort?",
            file=stderr,
        )
        answer = stdin.readline()
        if not answer:
            raise ConflictError("conflict prompt was closed")
        normalized = answer.strip().casefold()
        if normalized in {"o", "overwrite"}:
            return ConflictMode.OVERWRITE
        if normalized in {"n", "number", "numbered"}:
            return ConflictMode.NUMBERED
        if normalized in {"a", "abort", "q", "quit"}:
            raise ConflictError("aborted because output files already exist")
        print("Please answer o, n, or a.", file=stderr)


def resolve_output_conflicts(
    jobs: list[OutputJob],
    *,
    mode: ConflictMode,
    stdin: TextIO = sys.stdin,
    stderr: TextIO = sys.stderr,
) -> list[OutputJob]:
    """Apply one conflict policy to all existing output paths."""

    conflict_count = sum(job.target.exists() for job in jobs)
    if conflict_count == 0:
        return jobs
    selected_mode = mode
    if selected_mode is ConflictMode.ASK:
        selected_mode = _prompt_conflict_mode(
            conflict_count,
            stdin=stdin,
            stderr=stderr,
        )
    if selected_mode is ConflictMode.OVERWRITE:
        return jobs

    planned = {job.target for job in jobs}
    used: set[Path] = set()
    resolved: list[OutputJob] = []
    for job in jobs:
        unavailable = used | (planned - {job.target})
        target = job.target
        if target.exists() or target in used:
            target = _next_numbered_path(target, unavailable)
        used.add(target)
        resolved.append(replace(job, target=target))
    return resolved


def write_text_atomic(path: Path, text: str) -> None:
    """Atomically replace a UTF-8 text file in its destination directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
