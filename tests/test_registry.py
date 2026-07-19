from __future__ import annotations

import pytest

from nekotagger.errors import InputError
from nekotagger.taggers.camie_v2 import CamieV2Tagger
from nekotagger.taggers.oppai_oracle import OppaiOracleTagger
from nekotagger.taggers.registry import create_tagger, tagger_names


def test_registry_lists_and_constructs_supported_taggers() -> None:
    assert tagger_names() == ("oppai-oracle", "camie-v2")
    assert isinstance(
        create_tagger("oppai-oracle", variant=None, device="cpu"),
        OppaiOracleTagger,
    )
    assert isinstance(
        create_tagger("camie-v2", variant=None, device="cpu"),
        CamieV2Tagger,
    )


def test_camie_rejects_oppai_oracle_variant() -> None:
    with pytest.raises(InputError, match="only supported by tagger oppai-oracle"):
        create_tagger("camie-v2", variant="v1", device="cpu")
