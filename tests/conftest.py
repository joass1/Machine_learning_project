"""Shared fixtures: a small toy dataset in the competition schema with a learnable signal."""
import numpy as np
import pandas as pd
import pytest

from airsat.config import CATEGORY_LEVELS, SERVICE_RATINGS, TARGET


def make_toy_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Age": rng.integers(18, 80, n),
        "Flight Distance": rng.choice([300, 800, 1500, 2500, 4000], n),
        "Departure Delay in Minutes": rng.choice([0, 0, 0, 5, 30, 120], n),
        "Arrival Delay in Minutes": rng.choice([0.0, 0.0, 10.0, np.nan], n),
        **{col: rng.integers(0, 6, n) for col in SERVICE_RATINGS},
        **{col: rng.choice(levels, n) for col, levels in CATEGORY_LEVELS.items()},
    })
    logit = 1.2 * (df["Online boarding"] - 2.5) + 1.5 * (df["Type of Travel"] == "Business travel") - 0.5
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


@pytest.fixture
def toy() -> pd.DataFrame:
    return make_toy_frame()
