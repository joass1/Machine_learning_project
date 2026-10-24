"""Tabular neural nets from pytabkit (Holzmueller et al.): RealMLP-TD and TabM-D, on GPU when available.

pytabkit is imported lazily so the rest of the ladder works without it installed.
Best-epoch selection uses the CV validation fold (as GBDT early stopping does), scored by 1 - AUC.
"""
import logging

import numpy as np
import pandas as pd

from airsat.config import CATEGORICAL, SEED
from airsat.cv import ModelSpec
from airsat.models.inputs import numeric_frame, stratified_subsample

log = logging.getLogger(__name__)
COMMON = dict(val_metric_name="1-auc_ovr", n_threads=6, verbosity=0)
# RealMLP-TD defaults except: no label smoothing (pytabkit's advice for AUC / log-loss), and a
# shorter schedule with bigger batches (TD default 256 epochs x batch 256 is ~hours on 560k rows).
REALMLP_PARAMS = dict(COMMON, n_epochs=16, batch_size=1024, use_ls=False)
# TabM-D defaults except batch size (TabM paper uses 1024 on its largest datasets) and an epoch cap.
TABM_PARAMS = dict(COMMON, batch_size=1024, n_epochs=40, patience=8)
MAX_TRAIN_ROWS: int | None = None  # set per spec if a full training fold is too slow


def _device() -> str:
    import torch  # local: torch is only needed for these rungs
    return "cuda" if torch.cuda.is_available() else "cpu"


def _estimator_class(kind: str):
    from pytabkit import RealMLP_TD_Classifier, TabM_D_Classifier
    classes = {"realmlp": RealMLP_TD_Classifier, "tabm": TabM_D_Classifier}
    if kind not in classes:
        raise ValueError(f"kind must be one of {sorted(classes)}, got {kind!r}")
    return classes[kind]


class PyTabKitModel:
    """Estimator-protocol adapter for a pytabkit classifier; optional stratified training subsample."""

    def __init__(self, kind: str, params: dict, max_train_rows: int | None = None, seed: int = SEED):
        self.kind, self.params, self.max_train_rows, self.seed = kind, params, max_train_rows, seed
        self.model = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_valid=None, y_valid=None) -> "PyTabKitModel":
        X_fit, y_fit = stratified_subsample(numeric_frame(X), y, self.max_train_rows, self.seed)
        estimator = _estimator_class(self.kind)(device=_device(), random_state=self.seed, **self.params)
        cats = [c for c in CATEGORICAL if c in X_fit]
        if X_valid is None:
            estimator.fit(X_fit, np.asarray(y_fit), cat_col_names=cats)
        else:
            estimator.fit(X_fit, np.asarray(y_fit), numeric_frame(X_valid), np.asarray(y_valid),
                          cat_col_names=cats)
        self.model = estimator
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(numeric_frame(X))[:, 1]


def _spec(kind: str, params: dict, max_train_rows: int | None) -> ModelSpec:
    return ModelSpec(name=kind, feature_set="raw",
                     make_model=lambda: PyTabKitModel(kind, params, max_train_rows),
                     params={**params, "max_train_rows": max_train_rows,
                             "arrival_delay_nan": "filled with departure delay"})


def realmlp_raw() -> ModelSpec:
    return _spec("realmlp", REALMLP_PARAMS, MAX_TRAIN_ROWS)


def tabm_raw() -> ModelSpec:
    return _spec("tabm", TABM_PARAMS, MAX_TRAIN_ROWS)
