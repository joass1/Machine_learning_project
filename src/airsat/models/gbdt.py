"""Gradient-boosted tree models: LightGBM (CPU) and XGBoost (CUDA) with native categoricals."""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from airsat.config import SEED
from airsat.cv import ModelSpec
from airsat.features import EXACT_VALUE_PARAMS, ExactValueFeatures

LGBM_PARAMS = dict(n_estimators=4000, learning_rate=0.05, num_leaves=63, min_child_samples=40,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                   random_state=SEED, verbose=-1, n_jobs=6)
EARLY_STOPPING_ROUNDS = 200

# Mirrors the LightGBM settings (learning rate, row/column sampling, L2) at a comparable depth.
XGB_PARAMS = dict(n_estimators=5000, learning_rate=0.05, max_depth=6, min_child_weight=1.0,
                  subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, tree_method="hist",
                  device="cuda", enable_categorical=True, max_bin=256, eval_metric="auc",
                  early_stopping_rounds=EARLY_STOPPING_ROUNDS, random_state=SEED, n_jobs=6)
FINE_BINS = 1024  # "Flight Distance as route ID": finer histograms keep near-exact values apart


class LightGBMModel:
    def __init__(self, **params):
        self.params = params
        self.model = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_valid=None, y_valid=None) -> "LightGBMModel":
        self.model = lgb.LGBMClassifier(**self.params)
        if X_valid is None:
            self.model.fit(X, y)
        else:
            self.model.fit(X, y, eval_X=(X_valid,), eval_y=(y_valid,), eval_metric="auc",
                           callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class XGBoostModel:
    """XGBoost with native categorical splits; early stopping on the validation fold when given."""

    def __init__(self, **params):
        self.params = params
        self.model = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_valid=None, y_valid=None) -> "XGBoostModel":
        params = dict(self.params)
        if X_valid is None:
            params.pop("early_stopping_rounds", None)
            self.model = xgb.XGBClassifier(**params).fit(X, y, verbose=False)
        else:
            self.model = xgb.XGBClassifier(**params).fit(X, y, eval_set=[(X_valid, y_valid)], verbose=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        with warnings.catch_warnings():  # host-memory input on a CUDA booster: harmless fallback warning
            warnings.filterwarnings("ignore", message=".*mismatched devices.*")
            return self.model.predict_proba(X)[:, 1]


def lightgbm_raw() -> ModelSpec:
    return ModelSpec(name="lgbm", feature_set="raw", make_model=lambda: LightGBMModel(**LGBM_PARAMS),
                     params=LGBM_PARAMS)


def xgboost_raw() -> ModelSpec:
    return ModelSpec(name="xgb", feature_set="raw", make_model=lambda: XGBoostModel(**XGB_PARAMS),
                     params=XGB_PARAMS)


def xgboost_bins() -> ModelSpec:
    params = {**XGB_PARAMS, "max_bin": FINE_BINS}
    return ModelSpec(name="xgb_bins", feature_set="raw", make_model=lambda: XGBoostModel(**params),
                     params=params)


def xgboost_exact_value() -> ModelSpec:
    """xgb_bins + te_/cnt_ columns for EXACT_VALUE_COLUMNS (ExactValueFeatures defaults)."""
    params = {**XGB_PARAMS, "max_bin": FINE_BINS}
    return ModelSpec(name="xgb_ev", feature_set="exact_value", make_model=lambda: XGBoostModel(**params),
                     make_features=ExactValueFeatures, params={**params, **EXACT_VALUE_PARAMS})
