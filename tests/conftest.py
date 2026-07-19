from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-model",
        action="store_true",
        default=False,
        help="run tests that download the real OppaiOracle model",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if config.getoption("--run-model"):
        return
    skip_model = pytest.mark.skip(reason="requires --run-model and a large model download")
    for item in items:
        if "model" in item.keywords:
            item.add_marker(skip_model)
