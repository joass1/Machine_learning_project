"""Input helpers shared by several rungs: stratified training subsamples and numeric NN frames."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from airsat.config import CATEGORICAL, SEED


def stratified_subsample(X: pd.DataFrame, y: np.ndarray, n: int | None,
                         seed: int = SEED) -> tuple[pd.DataFrame, np.ndarray]:
    """At most `n` rows of (X, y), stratified on y, in their original order; unchanged if n is None."""
    y = np.asarray(y)
    if n is None or len(y) <= n:
        return X, y
    idx, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=seed)
    idx = np.sort(idx)
    return X.iloc[idx], y[idx]


def impute_arrival_delay(X: pd.DataFrame) -> pd.DataFrame:
    """Fill missing arrival delays with the departure delay (Pearson r = 0.965 real, 0.842 synthetic).

    Neural nets in pytabkit reject NaN; this rule needs no fitted statistics, so it is fold-safe.
    """
    if "Arrival Delay in Minutes" not in X or "Departure Delay in Minutes" not in X:
        return X
    filled = X["Arrival Delay in Minutes"].fillna(X["Departure Delay in Minutes"])
    return X.assign(**{"Arrival Delay in Minutes": filled})


def numeric_frame(X: pd.DataFrame, categorical: list[str] = CATEGORICAL) -> pd.DataFrame:
    """float32 numerics; categoricals stay pandas 'category' (fixed levels from to_model_frame)."""
    X = impute_arrival_delay(X)
    numeric = [c for c in X.columns if c not in categorical]
    return X.astype({c: np.float32 for c in numeric})


def category_codes(X: pd.DataFrame, categorical: list[str] = CATEGORICAL) -> pd.DataFrame:
    """All-numeric float32 frame: categoricals as their fixed integer codes (NaN kept as NaN)."""
    codes = {c: X[c].cat.codes.replace(-1, np.nan) for c in categorical if c in X}
    return X.assign(**codes).astype(np.float32)
