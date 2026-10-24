"""Fixed stratified folds shared by every model (one file per dataset)."""
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from airsat.config import FOLDS_DIR, N_FOLDS, SEED


def make_folds(y: np.ndarray, n_splits: int = N_FOLDS, seed: int = SEED) -> np.ndarray:
    """Fold id (0..n_splits-1) for every row, stratified on y."""
    y = np.asarray(y)
    fold_ids = np.full(len(y), -1, dtype=np.int8)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, valid_idx) in enumerate(splitter.split(np.zeros(len(y)), y)):
        fold_ids[valid_idx] = fold
    return fold_ids


def get_folds(name: str, y: np.ndarray, directory: Path = FOLDS_DIR) -> np.ndarray:
    """Load saved folds for `name`, or create and save them. Raises if a saved file mismatches y."""
    path = directory / f"{name}.npy"
    if path.exists():
        fold_ids = np.load(path)
        if len(fold_ids) != len(y):
            raise ValueError(f"saved folds {path} have {len(fold_ids)} rows, data has {len(y)}")
        return fold_ids
    fold_ids = make_folds(y)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(path, fold_ids)
    return fold_ids


def iter_folds(fold_ids: np.ndarray):
    """Yield (fold, train_idx, valid_idx) for each fold id in ascending order."""
    for fold in np.unique(fold_ids):
        yield int(fold), np.flatnonzero(fold_ids != fold), np.flatnonzero(fold_ids == fold)
