"""Tests for the stacking / ensemble-selection combiners."""
import numpy as np
import pytest

from airsat.folds import make_folds
from airsat.metrics import auc
from airsat.stacking import (FITTERS, evaluate_stack, hill_climb_weights, logit, nested_oof)


def _matrix(n: int = 2000, seed: int = 0):
    """Three 'models': two informative (different noise), one pure noise."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    signal = y + rng.normal(0, 0.8, n)
    Z = np.column_stack([
        1 / (1 + np.exp(-(signal + rng.normal(0, 0.6, n)))),
        1 / (1 + np.exp(-(signal + rng.normal(0, 0.6, n)))),
        rng.random(n),
    ])
    return Z, y


def test_logit_is_finite_at_the_edges():
    out = logit(np.array([0.0, 0.5, 1.0]))
    assert np.isfinite(out).all() and out[1] == pytest.approx(0.0)


def test_hill_climb_ignores_noise_and_blends_informative_models():
    Z, y = _matrix()
    w = hill_climb_weights(Z, y)
    assert w.sum() == pytest.approx(1.0) and w[2] == 0.0
    assert w[0] > 0 and w[1] > 0


@pytest.mark.parametrize("method", sorted(FITTERS))
def test_every_combiner_beats_the_noise_model(method):
    Z, y = _matrix()
    combiner = FITTERS[method](Z, y)
    assert len(combiner.weights) == 3
    assert auc(y, combiner.predict(Z)) > auc(y, Z[:, 2]) + 0.2


def test_nested_oof_and_apply_scores():
    Z, y = _matrix(seed=1)
    Z_apply, y_apply = _matrix(seed=2)
    folds = make_folds(y)
    oof = nested_oof(Z, y, folds, "logistic")
    assert oof.shape == y.shape
    result = evaluate_stack(Z, y, folds, "hill_climb", Z_apply, y_apply)
    best_single = max(auc(y, Z[:, j]) for j in range(3))
    assert result.nested_auc >= best_single - 0.01
    assert result.apply_auc is not None and result.apply_auc > 0.7
    assert evaluate_stack(Z, y, folds, "rank_mean").apply_auc is None
