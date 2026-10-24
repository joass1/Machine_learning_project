"""Evaluation metrics: AUC, per-fold statistics and bootstrap confidence intervals."""
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

N_BOOTSTRAP = 1000


def auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(np.asarray(y), np.asarray(p)))


def fold_aucs(y: np.ndarray, p: np.ndarray, fold_ids: np.ndarray) -> list[float]:
    y, p = np.asarray(y), np.asarray(p)
    return [auc(y[fold_ids == f], p[fold_ids == f]) for f in np.unique(fold_ids)]


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float


def _bootstrap_indices(n: int, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield rng.integers(0, n, n)


def bootstrap_auc(y: np.ndarray, p: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = 0) -> Interval:
    """Percentile 95% CI for AUC by resampling rows."""
    y, p = np.asarray(y), np.asarray(p)
    stats = [auc(y[idx], p[idx]) for idx in _bootstrap_indices(len(y), n_boot, seed) if 0 < y[idx].mean() < 1]
    return Interval(auc(y, p), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


def paired_bootstrap_delta(y: np.ndarray, p_new: np.ndarray, p_base: np.ndarray,
                           n_boot: int = N_BOOTSTRAP, seed: int = 0) -> Interval:
    """CI for AUC(p_new) - AUC(p_base) on the same resampled rows."""
    y, p_new, p_base = np.asarray(y), np.asarray(p_new), np.asarray(p_base)
    deltas = [auc(y[idx], p_new[idx]) - auc(y[idx], p_base[idx])
              for idx in _bootstrap_indices(len(y), n_boot, seed) if 0 < y[idx].mean() < 1]
    return Interval(auc(y, p_new) - auc(y, p_base), float(np.percentile(deltas, 2.5)),
                    float(np.percentile(deltas, 97.5)))
