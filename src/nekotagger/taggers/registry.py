"""Tagger adapter registry."""

from __future__ import annotations

from nekotagger.errors import InputError
from nekotagger.types import Tagger

_TAGGER_NAMES = ("oppai-oracle",)


def tagger_names() -> tuple[str, ...]:
    return _TAGGER_NAMES


def create_tagger(name: str, *, variant: str, device: str) -> Tagger:
    """Construct a tagger without importing heavyweight backends until needed."""

    if name == "oppai-oracle":
        from nekotagger.taggers.oppai_oracle import OppaiOracleTagger

        return OppaiOracleTagger(variant=variant, device=device)
    raise InputError(f"unknown tagger: {name}")
