"""Tagger adapter registry."""

from __future__ import annotations

from nekotagger.errors import InputError
from nekotagger.types import Tagger

_TAGGER_NAMES = ("oppai-oracle", "camie-v2")


def tagger_names() -> tuple[str, ...]:
    return _TAGGER_NAMES


def create_tagger(name: str, *, variant: str | None, device: str) -> Tagger:
    """Construct a tagger without importing heavyweight backends until needed."""

    if name == "oppai-oracle":
        from nekotagger.taggers.oppai_oracle import OppaiOracleTagger

        return OppaiOracleTagger(variant=variant or "v1.1", device=device)
    if name == "camie-v2":
        if variant is not None:
            raise InputError("--variant is only supported by tagger oppai-oracle")
        from nekotagger.taggers.camie_v2 import CamieV2Tagger

        return CamieV2Tagger(device=device)
    raise InputError(f"unknown tagger: {name}")
