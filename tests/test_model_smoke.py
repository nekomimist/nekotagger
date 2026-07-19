from __future__ import annotations

import os

import pytest
from PIL import Image

from nekotagger.taggers.oppai_oracle import OppaiOracleTagger


@pytest.mark.model
@pytest.mark.parametrize("variant", ["v1.1", "v1"])
def test_real_model_smoke(variant: str) -> None:
    device = os.environ.get("NEKOTAGGER_TEST_DEVICE", "cpu")
    tagger = OppaiOracleTagger(variant=variant, device=device)
    tagger.load()
    image = Image.new("RGB", (512, 384), (160, 120, 180))
    predictions = tagger.predict_batch([image], threshold=0.0, max_tags=3)
    assert len(predictions) == 1
    assert len(predictions[0]) == 3
    assert tagger.active_device.startswith(device.split(":", 1)[0])
