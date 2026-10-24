"""E6: does fold-safe target encoding (TE) of exact numeric values reproduce the forum's
reported +0.0013 AUC? (S6E10 discussion 745908; github.com/ferreret/kaggle-playground-s6e10)

Compares XGBoost (GPU) on raw features vs raw + nested-fold TE + count encoding, same 5 folds.
Run: python src/prelim_te.py
"""
import json
import logging

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from airsat.config import RESULTS_DIR, TARGET
from airsat.data import load_synthetic as load_competition
from airsat.data import to_model_frame
from prelim_experiments import N_FOLDS, SEED, XGB_PARAMS

log = logging.getLogger("prelim_te")
TE_COLS = ["Flight Distance", "Age", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
SMOOTHING = 20.0
N_INNER = 5


def smoothed_means(keys: pd.Series, y: pd.Series, prior: float) -> pd.Series:
    stats = y.groupby(keys).agg(["sum", "count"])
    return (stats["sum"] + prior * SMOOTHING) / (stats["count"] + SMOOTHING)


def target_encode(train_keys: pd.Series, y_train: pd.Series, valid_keys: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Train rows get inner-fold (out-of-fold) encodings; valid rows use the whole training fold."""
    prior = y_train.mean()
    train_enc = np.full(len(train_keys), prior)
    for in_tr, in_va in KFold(N_INNER, shuffle=True, random_state=SEED).split(train_keys):
        means = smoothed_means(train_keys.iloc[in_tr], y_train.iloc[in_tr], prior)
        train_enc[in_va] = train_keys.iloc[in_va].map(means).fillna(prior).to_numpy()
    valid_enc = valid_keys.map(smoothed_means(train_keys, y_train, prior)).fillna(prior).to_numpy()
    return train_enc, valid_enc


def add_encodings(X_tr: pd.DataFrame, y_tr: pd.Series, X_va: pd.DataFrame, counts: dict) -> tuple:
    X_tr, X_va = X_tr.copy(), X_va.copy()
    for col in TE_COLS:
        keys_tr, keys_va = X_tr[col].fillna(-1), X_va[col].fillna(-1)
        X_tr[f"te_{col}"], X_va[f"te_{col}"] = target_encode(keys_tr, y_tr, keys_va)
        X_tr[f"cnt_{col}"], X_va[f"cnt_{col}"] = keys_tr.map(counts[col]), keys_va.map(counts[col])
    return X_tr, X_va


def run(X: pd.DataFrame, y: pd.Series, counts: dict | None) -> dict:
    params = {**XGB_PARAMS, "max_bin": 1024}
    oof = np.zeros(len(y))
    for tr_idx, va_idx in StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y):
        X_tr, y_tr, X_va, y_va = X.iloc[tr_idx], y.iloc[tr_idx], X.iloc[va_idx], y.iloc[va_idx]
        if counts is not None:
            X_tr, X_va = add_encodings(X_tr, y_tr, X_va, counts)
        model = xgb.XGBClassifier(**params).fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof[va_idx] = model.predict_proba(X_va)[:, 1]
    return {"cv_auc": round(roc_auc_score(y, oof), 6)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    train, test = load_competition()
    X, y = to_model_frame(train), train[TARGET]
    full = pd.concat([train, test], ignore_index=True)  # unsupervised counts over train+test
    counts = {c: full[c].fillna(-1).value_counts() for c in TE_COLS}
    results = {"xgb_raw_maxbin1024": run(X, y, None), "xgb_te_count": run(X, y, counts)}
    results["delta"] = round(results["xgb_te_count"]["cv_auc"] - results["xgb_raw_maxbin1024"]["cv_auc"], 6)
    log.info(results)
    (RESULTS_DIR / "prelim_te.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
