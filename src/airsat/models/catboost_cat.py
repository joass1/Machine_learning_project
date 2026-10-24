"""CatBoost with every column treated as categorical (forum S6E10 trick: exact values as levels)."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from catboost.utils import get_gpu_device_count

from airsat.config import SEED
from airsat.cv import ModelSpec

MISSING_LEVEL = -1  # NaN becomes its own level (the integer analogue of a "nan" string)
CATBOOST_PARAMS = dict(iterations=5000, learning_rate=0.08, depth=6, l2_leaf_reg=3.0,
                       eval_metric="AUC", early_stopping_rounds=200, random_seed=SEED,
                       thread_count=6, verbose=0, allow_writing_files=False)


def task_type() -> str:
    """'GPU' when CatBoost sees a CUDA device, else 'CPU'."""
    try:
        return "GPU" if get_gpu_device_count() > 0 else "CPU"
    except Exception:  # noqa: BLE001 - CatBoost raises generic errors when CUDA is unusable
        return "CPU"


def all_categorical(X: pd.DataFrame) -> pd.DataFrame:
    """Integer level per column: category codes for categoricals, exact integer values for numerics.

    Raises if a numeric column has fractional values (they would be merged by the cast).
    """
    out = {}
    for col in X.columns:
        series = X[col]
        if isinstance(series.dtype, pd.CategoricalDtype):
            out[col] = series.cat.codes.astype(np.int64)  # NaN -> -1
            continue
        present = series.dropna()
        if not np.all(np.mod(present.to_numpy(dtype=float), 1) == 0):
            raise ValueError(f"column {col!r} has non-integer values; cannot use as exact levels")
        out[col] = series.fillna(MISSING_LEVEL).astype(np.int64)
    return pd.DataFrame(out, index=X.index)


class CatBoostAllCategorical:
    def __init__(self, **params):
        self.params = params
        self.model = None

    def _pool(self, X: pd.DataFrame, y: np.ndarray | None = None) -> Pool:
        frame = all_categorical(X)
        return Pool(frame, label=y, cat_features=list(range(frame.shape[1])))

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_valid=None, y_valid=None) -> "CatBoostAllCategorical":
        params = dict(self.params)
        params.setdefault("task_type", task_type())
        if X_valid is None:
            params.pop("early_stopping_rounds", None)
            eval_set = None
        else:
            eval_set = self._pool(X_valid, y_valid)
        self.model = CatBoostClassifier(**params)
        self.model.fit(self._pool(X, y), eval_set=eval_set, use_best_model=eval_set is not None)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._pool(X))[:, 1]


def catboost_all_categorical() -> ModelSpec:
    params = {**CATBOOST_PARAMS, "task_type": task_type()}
    return ModelSpec(name="catboost_cat", feature_set="all_categorical",
                     make_model=lambda: CatBoostAllCategorical(**params), params=params)
