"""Preliminary viability experiments for the S6E10 plan.

E1  Model ladder (5-fold stratified CV ROC AUC on competition train):
      logistic regression -> LightGBM -> XGBoost (GPU)
E2  Does appending the real survey data to each training fold help?
E3  Synthetic <-> real transfer (train on one, score AUC on the other)
E4  Adversarial validation (can a model tell train from test? train from real?)
E5  Global feature attribution with TreeSHAP (LightGBM pred_contrib)

Run: python src/prelim_experiments.py
"""
import json
import logging
import time

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from airsat import plotstyle as ps
from airsat.config import (CATEGORICAL, DISPLAY_NAMES, FIG_DIR, NUMERIC, RESULTS_DIR,
                           SERVICE_RATINGS, TARGET)
from airsat.data import load_real as load_original
from airsat.data import load_synthetic as load_competition
from airsat.data import to_model_frame

log = logging.getLogger("prelim")
SEED = 42
N_FOLDS = 5
LGB_PARAMS = dict(n_estimators=4000, learning_rate=0.05, num_leaves=63, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                  random_state=SEED, verbose=-1, n_jobs=-1)
XGB_PARAMS = dict(n_estimators=4000, learning_rate=0.05, max_depth=7, subsample=0.8,
                  colsample_bytree=0.7, tree_method="hist", device="cuda", enable_categorical=True,
                  eval_metric="auc", early_stopping_rounds=200, random_state=SEED)
EARLY_STOP = 200


def make_logreg():
    pre = make_column_transformer(
        (make_pipeline(StandardScaler()), NUMERIC),
        (OneHotEncoder(handle_unknown="ignore"), SERVICE_RATINGS + CATEGORICAL),
    )
    return make_pipeline(pre, LogisticRegression(max_iter=2000, C=1.0))


def fit_predict(kind: str, X_tr, y_tr, X_va, y_va) -> np.ndarray:
    if kind == "logreg":
        model = make_logreg().fit(X_tr.fillna(0), y_tr)
        return model.predict_proba(X_va.fillna(0))[:, 1]
    if kind == "lgbm":
        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        return model.predict_proba(X_va)[:, 1]
    if kind == "xgb":
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        return model.predict_proba(X_va)[:, 1]
    raise ValueError(f"unknown model kind: {kind}")


def run_cv(kind: str, X: pd.DataFrame, y: pd.Series, extra: tuple | None = None) -> dict:
    """Stratified K-fold CV; `extra` = (X_extra, y_extra) appended to every training fold only."""
    oof = np.zeros(len(y))
    fold_aucs = []
    start = time.time()
    for tr_idx, va_idx in StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        if extra is not None:
            X_tr, y_tr = pd.concat([X_tr, extra[0]]), pd.concat([y_tr, extra[1]])
        oof[va_idx] = fit_predict(kind, X_tr, y_tr, X.iloc[va_idx], y.iloc[va_idx])
        fold_aucs.append(roc_auc_score(y.iloc[va_idx], oof[va_idx]))
    result = {"cv_auc": round(roc_auc_score(y, oof), 6), "fold_std": round(float(np.std(fold_aucs)), 6),
              "fold_aucs": [round(a, 6) for a in fold_aucs], "seconds": round(time.time() - start, 1)}
    log.info("%s%s -> %s", kind, " +orig" if extra is not None else "", result)
    return result


def transfer(X_a, y_a, X_b, y_b) -> float:
    model = lgb.LGBMClassifier(**{**LGB_PARAMS, "n_estimators": 800}).fit(X_a, y_a)
    return round(roc_auc_score(y_b, model.predict_proba(X_b)[:, 1]), 6)


def adversarial_auc(X_a: pd.DataFrame, X_b: pd.DataFrame, n: int = 100_000) -> float:
    """CV AUC of a classifier separating sample A from sample B (0.5 = indistinguishable)."""
    a = X_a.sample(min(n, len(X_a)), random_state=SEED)
    b = X_b.sample(min(n, len(X_b)), random_state=SEED)
    X = pd.concat([a, b], ignore_index=True)
    y = pd.Series(np.r_[np.zeros(len(a)), np.ones(len(b))])
    params = {**LGB_PARAMS, "n_estimators": 300}
    oof = np.zeros(len(y))
    for tr_idx, va_idx in StratifiedKFold(3, shuffle=True, random_state=SEED).split(X, y):
        model = lgb.LGBMClassifier(**params).fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
    return round(roc_auc_score(y, oof), 4)


def shap_importance(X: pd.DataFrame, y: pd.Series, n_explain: int = 50_000) -> pd.Series:
    model = lgb.LGBMClassifier(**{**LGB_PARAMS, "n_estimators": 800}).fit(X, y)
    sample = X.sample(n_explain, random_state=SEED)
    contrib = model.predict(sample, pred_contrib=True)[:, :-1]  # drop bias column
    return pd.Series(np.abs(contrib).mean(axis=0), index=X.columns).sort_values(ascending=False)


SHAP_SLOT_PX = 768  # width of the figure's slot on the slide


def fig_shap(importance: pd.Series) -> None:
    top = importance.head(10)[::-1]
    fig, ax = plt.subplots(figsize=ps.figsize(SHAP_SLOT_PX, 2.1))
    ax.barh([DISPLAY_NAMES[c] for c in top.index], top.values, height=0.62, color=ps.SERIES[0])
    for yi, v in enumerate(top.values):
        ax.text(v + top.max() * 0.015, yi, f"{v:.2f}", va="center", fontsize=12, color=ps.INK_2)
    ax.set_xlim(0, top.max() * 1.18)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean |SHAP| (log-odds)")
    ps.save(fig, FIG_DIR / "fig_shap_importance.png")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    ps.apply()
    train, test = load_competition()
    orig = load_original()
    X, y = to_model_frame(train), train[TARGET]
    X_orig, y_orig = to_model_frame(orig), orig[TARGET]
    X_test = to_model_frame(test)

    results = {"E1_model_ladder": {k: run_cv(k, X, y) for k in ("logreg", "lgbm", "xgb")}}
    results["E2_add_original"] = {"lgbm_plus_orig": run_cv("lgbm", X, y, extra=(X_orig, y_orig))}
    results["E3_transfer"] = {"synthetic_to_real": transfer(X, y, X_orig, y_orig),
                              "real_to_synthetic": transfer(X_orig, y_orig, X, y)}
    log.info("E3 %s", results["E3_transfer"])
    results["E4_adversarial"] = {"train_vs_test": adversarial_auc(X, X_test),
                                 "train_vs_original": adversarial_auc(X, X_orig)}
    log.info("E4 %s", results["E4_adversarial"])
    importance = shap_importance(X, y)
    results["E5_shap_mean_abs"] = importance.round(4).to_dict()
    fig_shap(importance)
    (RESULTS_DIR / "prelim_results.json").write_text(json.dumps(results, indent=2))
    log.info("saved %s", RESULTS_DIR / "prelim_results.json")


if __name__ == "__main__":
    main()
