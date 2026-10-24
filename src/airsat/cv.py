"""Cross-validation runner: one model spec -> out-of-fold predictions + averaged predictions
on other datasets ("apply sets"), e.g. a synthetic-trained model scoring real passengers."""
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from airsat.folds import iter_folds
from airsat.metrics import auc

log = logging.getLogger(__name__)


class Estimator(Protocol):
    def fit(self, X: pd.DataFrame, y: np.ndarray, X_valid: pd.DataFrame | None = None,
            y_valid: np.ndarray | None = None) -> "Estimator": ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(satisfied) for each row."""
        ...


class FeaturePipeline(Protocol):
    def fit_transform(self, X: pd.DataFrame, y: np.ndarray, unlabeled: pd.DataFrame | None) -> pd.DataFrame: ...

    def transform(self, X: pd.DataFrame) -> pd.DataFrame: ...


class IdentityFeatures:
    """Feature pipeline that passes the frame through unchanged."""

    def fit_transform(self, X, y, unlabeled=None):
        return X

    def transform(self, X):
        return X


@dataclass(frozen=True)
class ModelSpec:
    """A named model + feature-set combination; factories give fresh objects per fold."""

    name: str
    feature_set: str
    make_model: Callable[[], Estimator]
    make_features: Callable[[], FeaturePipeline] = IdentityFeatures
    params: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.name}__{self.feature_set}"


@dataclass
class ApplySet:
    X: pd.DataFrame
    y: np.ndarray | None = None


@dataclass
class CVResult:
    key: str
    oof: np.ndarray
    fold_aucs: list[float]
    cv_auc: float
    applied: dict[str, np.ndarray]
    applied_auc: dict[str, float]
    seconds: float


def run_cv(spec: ModelSpec, X: pd.DataFrame, y: np.ndarray, fold_ids: np.ndarray,
           apply_sets: dict[str, ApplySet] | None = None,
           unlabeled: pd.DataFrame | None = None) -> CVResult:
    """Train one model per fold; average each fold model's predictions on every apply set."""
    y = np.asarray(y)
    apply_sets = apply_sets or {}
    oof = np.zeros(len(y))
    applied = {name: np.zeros(len(s.X)) for name, s in apply_sets.items()}
    fold_scores, n_folds, start = [], len(np.unique(fold_ids)), time.time()
    for fold, tr_idx, va_idx in iter_folds(fold_ids):
        features = spec.make_features()
        X_tr = features.fit_transform(X.iloc[tr_idx], y[tr_idx], unlabeled)
        X_va = features.transform(X.iloc[va_idx])
        model = spec.make_model().fit(X_tr, y[tr_idx], X_va, y[va_idx])
        oof[va_idx] = model.predict(X_va)
        fold_scores.append(auc(y[va_idx], oof[va_idx]))
        for name, s in apply_sets.items():
            applied[name] += model.predict(features.transform(s.X)) / n_folds
        log.info("%s fold %d auc=%.5f (%.0fs)", spec.key, fold, fold_scores[-1], time.time() - start)
    applied_auc = {name: auc(s.y, applied[name]) for name, s in apply_sets.items() if s.y is not None}
    return CVResult(spec.key, oof, fold_scores, auc(y, oof), applied, applied_auc,
                    round(time.time() - start, 1))
