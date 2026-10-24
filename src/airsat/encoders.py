"""Fold-safe exact-value target encoding and count encoding.

Target encoding (Micci-Barreca, 2001) maps each exact value of a column to a smoothed share of
positives. To avoid leakage, training rows receive *out-of-fold* encodings computed from inner
folds, while validation/apply rows use statistics from the whole training fold.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

MISSING_KEY = -1.0


def _keys(series: pd.Series) -> pd.Series:
    return series.astype(float).fillna(MISSING_KEY)


def smoothed_means(keys: pd.Series, y: pd.Series, prior: float, smoothing: float) -> pd.Series:
    stats = pd.DataFrame({"k": keys.to_numpy(), "y": np.asarray(y)}).groupby("k")["y"].agg(["sum", "count"])
    return (stats["sum"] + prior * smoothing) / (stats["count"] + smoothing)


@dataclass
class ExactValueEncoder:
    """Adds `te_<col>` (target encoding) and `cnt_<col>` (frequency) columns for `columns`."""

    columns: list[str]
    smoothing: float = 20.0
    n_inner: int = 5
    seed: int = 42
    prior_: float = field(default=np.nan, init=False)
    means_: dict = field(default_factory=dict, init=False)
    counts_: dict = field(default_factory=dict, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series, unlabeled: pd.DataFrame | None = None) -> "ExactValueEncoder":
        """Fit on training rows; `unlabeled` rows (no target used) only add to the counts."""
        y = pd.Series(np.asarray(y), index=X.index)
        self.prior_ = float(y.mean())
        pool = X if unlabeled is None else pd.concat([X[self.columns], unlabeled[self.columns]])
        for col in self.columns:
            self.means_[col] = smoothed_means(_keys(X[col]), y, self.prior_, self.smoothing)
            self.counts_[col] = _keys(pool[col]).value_counts()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode rows not used for fitting (validation, test, other datasets)."""
        out = X.copy()
        for col in self.columns:
            keys = _keys(X[col])
            out[f"te_{col}"] = keys.map(self.means_[col]).fillna(self.prior_).to_numpy()
            out[f"cnt_{col}"] = keys.map(self.counts_[col]).fillna(0).to_numpy()
        return out

    def fit_transform_oof(self, X: pd.DataFrame, y: pd.Series,
                          unlabeled: pd.DataFrame | None = None) -> pd.DataFrame:
        """Fit on all rows, but return training-row encodings computed out-of-fold."""
        self.fit(X, y, unlabeled)
        out = self.transform(X)
        y_arr = np.asarray(y)
        for col in self.columns:
            out[f"te_{col}"] = self._oof_column(_keys(X[col]), y_arr)
        return out

    def _oof_column(self, keys: pd.Series, y: np.ndarray) -> np.ndarray:
        encoded = np.full(len(keys), self.prior_)
        splitter = KFold(self.n_inner, shuffle=True, random_state=self.seed)
        for inner_tr, inner_va in splitter.split(keys):
            prior = float(y[inner_tr].mean())
            means = smoothed_means(keys.iloc[inner_tr], pd.Series(y[inner_tr]), prior, self.smoothing)
            encoded[inner_va] = keys.iloc[inner_va].map(means).fillna(prior).to_numpy()
        return encoded
