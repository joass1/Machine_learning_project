"""Load the synthetic (Kaggle S6E10) and real (original survey) datasets in one schema.

Synthetic: https://www.kaggle.com/competitions/playground-series-s6e10/data
Real:      https://www.kaggle.com/datasets/arseniyshutko/binary-aviation-satisfaction-129k
"""
from pathlib import Path

import pandas as pd

from airsat.config import CATEGORY_LEVELS, FEATURES, REAL_FILE, SYNTHETIC_DIR, TARGET


def _with_int_target(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(**{TARGET: df[TARGET].astype(int)})


def load_synthetic(directory: Path = SYNTHETIC_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test) of the synthetic competition data; train target as 0/1."""
    train = pd.read_csv(directory / "train.csv")
    test = pd.read_csv(directory / "test.csv")
    return _with_int_target(train), test


def load_real(path: Path = REAL_FILE) -> pd.DataFrame:
    """Return the real 129,880-row survey in the competition schema; target as 0/1."""
    real = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(real.columns)
    if missing:
        raise ValueError(f"real data is missing columns: {sorted(missing)}")
    return _with_int_target(real)[FEATURES + [TARGET]]


def to_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix with categoricals encoded against fixed vocabularies.

    Using fixed levels guarantees identical category codes for synthetic and real rows,
    so a model trained on one dataset can score the other.
    """
    frame = df[FEATURES].copy()
    for col, levels in CATEGORY_LEVELS.items():
        unknown = set(frame[col].dropna().unique()) - set(levels)
        if unknown:
            raise ValueError(f"unexpected levels in {col!r}: {sorted(unknown)}")
        frame[col] = pd.Categorical(frame[col], categories=levels)
    return frame
