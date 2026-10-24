"""Run a model spec in both training directions and save every setting to the prediction store.

- train on synthetic folds -> syn_cv (OOF), syn_to_real (apply "real"), syn_test (apply "syn_test")
- train on real folds      -> real_cv (OOF), real_to_syn (apply "synthetic")
`sample` (rows per dataset) gives a fast development run stored under artifacts/oof/dev.
"""
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from airsat.config import OOF_DIR, SEED, TARGET
from airsat.cv import ApplySet, CVResult, ModelSpec, run_cv
from airsat.data import load_real, load_synthetic, to_model_frame
from airsat.folds import get_folds, make_folds
from airsat.oof import save_run

log = logging.getLogger(__name__)
DIRECTIONS = ("synthetic", "real")


@dataclass
class Datasets:
    X_syn: pd.DataFrame
    y_syn: np.ndarray
    X_test: pd.DataFrame
    X_real: pd.DataFrame
    y_real: np.ndarray


def _subsample(df: pd.DataFrame, n: int | None) -> pd.DataFrame:
    return df if n is None or n >= len(df) else df.sample(n, random_state=SEED).reset_index(drop=True)


def load_datasets(sample: int | None = None) -> Datasets:
    train, test = load_synthetic()
    real = load_real()
    train, test, real = _subsample(train, sample), _subsample(test, sample), _subsample(real, sample)
    return Datasets(to_model_frame(train), train[TARGET].to_numpy(), to_model_frame(test),
                    to_model_frame(real), real[TARGET].to_numpy())


def folds_for(direction: str, y: np.ndarray, sample: int | None) -> np.ndarray:
    return make_folds(y) if sample else get_folds(direction, y)


def run_direction(spec: ModelSpec, data: Datasets, direction: str, sample: int | None = None) -> CVResult:
    if direction == "synthetic":
        apply_sets = {"real": ApplySet(data.X_real, data.y_real), "syn_test": ApplySet(data.X_test)}
        return run_cv(spec, data.X_syn, data.y_syn, folds_for(direction, data.y_syn, sample),
                      apply_sets, unlabeled=data.X_test)
    if direction == "real":
        apply_sets = {"synthetic": ApplySet(data.X_syn, data.y_syn)}
        return run_cv(spec, data.X_real, data.y_real, folds_for(direction, data.y_real, sample), apply_sets)
    raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")


def run_and_save(spec: ModelSpec, data: Datasets, directions=DIRECTIONS, sample: int | None = None) -> dict:
    """Run `spec` in each direction, save to the store, return {direction: CVResult}."""
    root = OOF_DIR / "dev" if sample else OOF_DIR
    results = {}
    for direction in directions:
        result = run_direction(spec, data, direction, sample)
        save_run(direction, result, {"model": spec.name, "feature_set": spec.feature_set,
                                     "params": {k: str(v) for k, v in spec.params.items()},
                                     "sample": sample}, root=root)
        log.info("%s [%s] cv=%.5f applied=%s (%ss)", spec.key, direction, result.cv_auc,
                 {k: round(v, 5) for k, v in result.applied_auc.items()}, result.seconds)
        results[direction] = result
    return results
