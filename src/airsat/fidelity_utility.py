"""RQ2 synthetic-data utility and the real-trained reference model.

- `utility_curve`: learning curve of LightGBM trained on n real rows, scored on one held-out real
  fold, compared with TSTR (LightGBM trained on all synthetic rows, scored on the same fold);
  `equivalent_n` reads off how many real rows the synthetic set is worth.
- `real_reference_predictions`: 5-fold LightGBM on real data, applied to synthetic rows
  (input to the label-noise estimate), unless the prediction store already holds a full run.
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from airsat.config import OOF_DIR, SEED
from airsat.cv import ApplySet, CVResult, ModelSpec, run_cv
from airsat.metrics import Interval, auc, bootstrap_auc, paired_bootstrap_delta
from airsat.models.gbdt import LightGBMModel

log = logging.getLogger(__name__)

# Same tree settings as the WP1 LightGBM rung, but at most 4 CPU threads (shared machine).
FIDELITY_LGBM = dict(n_estimators=5000, learning_rate=0.05, num_leaves=63, min_child_samples=40,
                     subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                     verbose=-1, n_jobs=4)
UTILITY_SIZES = (1_000, 2_000, 5_000, 10_000, 20_000, 50_000, None)  # None = every remaining row
UTILITY_SEEDS = (0, 1, 2)
MULTI_SEED_MAX_N = 20_000
VALID_SHARE = 0.1
EARLY_STOPPING_ROUNDS = 100


def stratified_subsample(y, n: int | None, seed: int) -> np.ndarray:
    """Sorted positions of a stratified sample of n rows (all rows if n is None or >= len(y))."""
    idx = np.arange(len(y))
    if n is None or n >= len(y):
        return idx
    chosen, _ = train_test_split(idx, train_size=n, stratify=np.asarray(y), random_state=seed)
    return np.sort(chosen)


def fit_with_holdout(X: pd.DataFrame, y, seed: int, params: dict | None = None,
                     valid_share: float = VALID_SHARE) -> lgb.LGBMClassifier:
    """Fit LightGBM on (1 - valid_share) of the rows, early-stopping on the remaining stratified share.

    The early-stopping rows come from the training budget itself, never from the test set.
    """
    y = np.asarray(y)
    tr, va = train_test_split(np.arange(len(y)), test_size=valid_share, stratify=y, random_state=seed)
    model = lgb.LGBMClassifier(**{**FIDELITY_LGBM, **(params or {}), "random_state": seed})
    model.fit(X.iloc[tr], y[tr], eval_X=(X.iloc[va],), eval_y=(y[va],), eval_metric="auc",
              callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
    return model


def _effective_sizes(sizes, n_pool: int) -> list[int]:
    return list(dict.fromkeys(n_pool if n is None else min(n, n_pool) for n in sizes))


def real_learning_curve(X_pool: pd.DataFrame, y_pool, X_test: pd.DataFrame, y_test,
                        sizes=UTILITY_SIZES, seeds=UTILITY_SEEDS, multi_seed_max_n: int = MULTI_SEED_MAX_N,
                        params: dict | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """AUC on the test rows of LightGBM trained on n stratified pool rows, for each n (and seed).

    Returns the per-fit table and the test predictions of the model trained on the largest n.
    """
    y_pool, rows, p_largest = np.asarray(y_pool), [], None
    for n in _effective_sizes(sizes, len(y_pool)):
        for seed in seeds if n <= multi_seed_max_n else seeds[:1]:
            idx = stratified_subsample(y_pool, n, seed)
            model = fit_with_holdout(X_pool.iloc[idx], y_pool[idx], seed, params)
            p = model.predict_proba(X_test)[:, 1]
            rows.append({"n": n, "seed": seed, "auc": auc(y_test, p), "best_iteration": model.best_iteration_})
            log.info("learning curve n=%d seed=%d auc=%.5f trees=%d", n, seed, rows[-1]["auc"],
                     model.best_iteration_)
            p_largest = p
    return pd.DataFrame(rows), p_largest


def equivalent_n(ns, aucs, target: float) -> float | None:
    """Smallest n where the curve reaches `target`, linearly interpolated on log(n).

    Returns ns[0] if the curve already reaches the target at the smallest n,
    and None if it never does (the synthetic set is worth more than every real size tried).
    """
    ns, aucs = np.asarray(ns, dtype=float), np.asarray(aucs, dtype=float)
    order = np.argsort(ns)
    ns, aucs = ns[order], aucs[order]
    if aucs[0] >= target:
        return float(ns[0])
    for k in range(1, len(ns)):
        if aucs[k] >= target:
            frac = (target - aucs[k - 1]) / (aucs[k] - aucs[k - 1])
            return float(np.exp(np.log(ns[k - 1]) + frac * (np.log(ns[k]) - np.log(ns[k - 1]))))
    return None


@dataclass(frozen=True)
class UtilityResult:
    curve: pd.DataFrame        # one row per fit: n, seed, auc, best_iteration
    summary: pd.DataFrame      # per n: auc_mean, auc_std, n_seeds
    tstr: Interval             # AUC of the synthetic-trained model on the held-out real fold
    tstr_minus_trtr: Interval  # paired bootstrap: TSTR - real model trained on the largest n
    equivalent_n: float | None
    equivalent_n_ci: tuple[float | None, float | None]  # at the TSTR CI bounds (low, high)
    n_test: int
    n_pool: int
    n_synthetic: int
    tstr_trees: int


def utility_curve(X_real: pd.DataFrame, y_real, real_folds: np.ndarray, X_syn: pd.DataFrame, y_syn,
                  test_fold: int = 0, sizes=UTILITY_SIZES, seeds=UTILITY_SEEDS,
                  multi_seed_max_n: int = MULTI_SEED_MAX_N, params: dict | None = None,
                  n_boot: int = 1000) -> UtilityResult:
    """Real-data learning curve vs TSTR on the same held-out real fold."""
    y_real, test = np.asarray(y_real), np.asarray(real_folds) == test_fold
    X_test, y_test = X_real.iloc[np.flatnonzero(test)], y_real[test]
    X_pool, y_pool = X_real.iloc[np.flatnonzero(~test)], y_real[~test]
    curve, p_trtr = real_learning_curve(X_pool, y_pool, X_test, y_test, sizes, seeds, multi_seed_max_n, params)
    summary = curve.groupby("n")["auc"].agg(auc_mean="mean", auc_std="std", n_seeds="count").reset_index()
    tstr_model = fit_with_holdout(X_syn, y_syn, SEED, params)
    p_tstr = tstr_model.predict_proba(X_test)[:, 1]
    tstr = bootstrap_auc(y_test, p_tstr, n_boot=n_boot)
    log.info("TSTR auc=%.5f [%.5f, %.5f] trees=%d", tstr.estimate, tstr.low, tstr.high,
             tstr_model.best_iteration_)
    ns, means = summary["n"], summary["auc_mean"]
    return UtilityResult(
        curve, summary, tstr, paired_bootstrap_delta(y_test, p_tstr, p_trtr, n_boot=n_boot),
        equivalent_n(ns, means, tstr.estimate),
        (equivalent_n(ns, means, tstr.low), equivalent_n(ns, means, tstr.high)),
        int(test.sum()), int((~test).sum()), len(X_syn), int(tstr_model.best_iteration_))


# ---------- real-trained reference model (label noise) ----------
def real_reference_predictions(X_real: pd.DataFrame, y_real, real_folds: np.ndarray, X_syn: pd.DataFrame,
                               y_syn, params: dict | None = None) -> CVResult:
    """5-fold LightGBM on real rows: OOF predictions on real + fold-averaged predictions on synthetic."""
    lgb_params = {**FIDELITY_LGBM, **(params or {}), "random_state": SEED}
    spec = ModelSpec(name="lgbm_rq2", feature_set="raw", make_model=lambda: LightGBMModel(**lgb_params),
                     params=lgb_params)
    return run_cv(spec, X_real, np.asarray(y_real), real_folds, {"synthetic": ApplySet(X_syn, np.asarray(y_syn))})


def stored_real_predictions(n_real: int, n_syn: int, key: str = "lgbm__raw",
                            root: Path = OOF_DIR) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """(real OOF, real-model predictions on synthetic, metadata) from a full store run, or None.

    Returns None when the run is missing, was a sampled dev run, or its lengths do not match.
    """
    npz_path, json_path = root / "real" / f"{key}.npz", root / "real" / f"{key}.json"
    if not (npz_path.exists() and json_path.exists()):
        return None
    meta = json.loads(json_path.read_text())
    with np.load(npz_path) as data:
        if meta.get("sample") or "apply__synthetic" not in data.files:
            return None
        oof, p_syn = data["oof"], data["apply__synthetic"]
    if len(oof) != n_real or len(p_syn) != n_syn:
        log.warning("stored %s has %d/%d rows, expected %d/%d; ignoring", key, len(oof), len(p_syn), n_real, n_syn)
        return None
    return oof, p_syn, meta
