"""Shared fixtures: bundled data paths and surfaces."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from localvol import DupireLocalVol, ImpliedVolSurface

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR


@pytest.fixture(scope="session")
def bundled_surface() -> ImpliedVolSurface:
    return ImpliedVolSurface.from_csv(DATA_DIR / "implied_surface.csv")


@pytest.fixture(scope="session")
def flat_surface() -> ImpliedVolSurface:
    return ImpliedVolSurface.from_csv(DATA_DIR / "flat_surface.csv")


@pytest.fixture(scope="session")
def bundled_localvol(bundled_surface: ImpliedVolSurface) -> DupireLocalVol:
    return DupireLocalVol(bundled_surface)


@pytest.fixture(autouse=True)
def _no_unexpected_warnings():
    """Keep intentional warnings visible in the code paths that expect them."""
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        yield
