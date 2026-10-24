"""Combine saved model predictions: rank averaging, logistic stacking on logits (Wolpert, 1992)
and greedy ensemble selection / hill climbing (Caruana et al., 2004).

All in-dataset scores are nested: the combiner for outer fold k is fitted only on the other
folds' out-of-fold predictions, so it never sees the rows it is scored on.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegressionCV

from airsat.metrics import auc

EPS = 1e-6
LOGISTIC_CS = (0.001, 0.01, 0.1, 1.0, 10.0)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


@dataclass
class Combiner:
    """A fitted combiner: weights over models plus a function producing scores."""

    name: str
    predict: Callable[[np.ndarray], np.ndarray]
    weights: np.ndarray


def fit_rank_mean(Z: np.ndarray, y: np.ndarray) -> Combiner:
    m = Z.shape[1]
    return Combiner("rank_mean", lambda P: rankdata(P, axis=0).mean(axis=1) / len(P), np.full(m, 1 / m))


def fit_logistic(Z: np.ndarray, y: np.ndarray, seed: int = 0) -> Combiner:
    """L2 logistic regression on logits; C chosen by inner 3-fold CV on AUC."""
    model = LogisticRegressionCV(Cs=list(LOGISTIC_CS), l1_ratios=(0.0,), cv=3, scoring="roc_auc",
                                 max_iter=2000, random_state=seed, use_legacy_attributes=False).fit(logit(Z), y)
    return Combiner("logistic", lambda P: model.predict_proba(logit(P))[:, 1], model.coef_.ravel())


def hill_climb_weights(Z: np.ndarray, y: np.ndarray, max_steps: int = 50, tol: float = 1e-7) -> np.ndarray:
    """Greedy forward selection with replacement on logits, starting from the best single model."""
    L = logit(Z)
    counts = np.zeros(Z.shape[1])
    best_single = int(np.argmax([auc(y, L[:, j]) for j in range(Z.shape[1])]))
    counts[best_single], blend, best = 1, L[:, best_single].copy(), auc(y, L[:, best_single])
    for step in range(2, max_steps + 1):
        trials = [auc(y, (blend * (step - 1) + L[:, j]) / step) for j in range(Z.shape[1])]
        j = int(np.argmax(trials))
        if trials[j] <= best + tol:
            break
        counts[j] += 1
        blend, best = (blend * (step - 1) + L[:, j]) / step, trials[j]
    return counts / counts.sum()


def fit_hill_climb(Z: np.ndarray, y: np.ndarray) -> Combiner:
    w = hill_climb_weights(Z, y)
    return Combiner("hill_climb", lambda P: logit(P) @ w, w)


FITTERS: dict[str, Callable[[np.ndarray, np.ndarray], Combiner]] = {
    "rank_mean": fit_rank_mean, "logistic": fit_logistic, "hill_climb": fit_hill_climb,
}


def nested_oof(Z: np.ndarray, y: np.ndarray, fold_ids: np.ndarray, method: str) -> np.ndarray:
    """Outer-fold predictions of a combiner fitted on the remaining folds only."""
    out = np.zeros(len(y))
    for fold in np.unique(fold_ids):
        train, valid = fold_ids != fold, fold_ids == fold
        out[valid] = FITTERS[method](Z[train], y[train]).predict(Z[valid])
    return out


@dataclass
class StackResult:
    method: str
    nested_auc: float
    apply_auc: float | None
    weights: np.ndarray


def evaluate_stack(Z: np.ndarray, y: np.ndarray, fold_ids: np.ndarray, method: str,
                   Z_apply: np.ndarray | None = None, y_apply: np.ndarray | None = None) -> StackResult:
    """Nested in-dataset AUC, plus AUC on another dataset for a combiner fitted on all rows."""
    nested = auc(y, nested_oof(Z, y, fold_ids, method))
    full = FITTERS[method](Z, y)
    apply_score = None
    if Z_apply is not None and y_apply is not None:
        apply_score = auc(y_apply, full.predict(Z_apply))
    return StackResult(method, nested, apply_score, full.weights)
