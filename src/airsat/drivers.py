"""RQ3 driver analyses: which services move satisfaction, for whom, and do synthetic and real agree.

All functions are pure (no file I/O) and work on frames from `airsat.data.to_model_frame`.
Every quantity here describes how a fitted model's predictions respond to the ratings. The
results are associational: they do not say what would happen if an airline improved a service.
"""
import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from airsat.config import SEED, SEGMENT_COLUMNS, SERVICE_RATINGS
from airsat.metrics import auc
from airsat.models.gbdt import LGBM_PARAMS, LightGBMModel

log = logging.getLogger(__name__)

DRIVER_N_JOBS = 4             # memory/CPU budget: other jobs share the machine
EARLY_STOP_FRACTION = 0.1     # share of training rows held out for early stopping
MAX_SHAP_ROWS = 20_000        # TreeSHAP rows per segment
N_TOP1_BOOTSTRAP = 200        # row-bootstrap draws for the stability of the #1 service
RATING_MIN, RATING_MAX = 1, 5  # 0 = "not applicable"
WHAT_IF_MAX_START = 3         # what-if raises only passengers who rated the service 1-3
TOP_K = 3
SEGMENT_SEP = " | "


class Predictor(Protocol):
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(satisfied) for each row."""
        ...


@dataclass(frozen=True)
class DriverModel:
    """A LightGBM fitted on every fold except `holdout_fold`; analyses use the holdout rows."""

    model: LightGBMModel
    train_idx: np.ndarray
    holdout_idx: np.ndarray
    holdout_auc: float
    best_iteration: int


def driver_params(**overrides) -> dict:
    """The reference LightGBM parameters with the thread count capped at DRIVER_N_JOBS."""
    params = {**LGBM_PARAMS, **overrides}
    n_jobs = params.get("n_jobs", DRIVER_N_JOBS)
    params["n_jobs"] = n_jobs if 0 < n_jobs <= DRIVER_N_JOBS else DRIVER_N_JOBS
    return params


def train_driver_model(X: pd.DataFrame, y: np.ndarray, fold_ids: np.ndarray, holdout_fold: int = 0,
                       params: dict | None = None, seed: int = SEED) -> DriverModel:
    """Fit LightGBM on all folds except the holdout; early stopping uses a split of the training rows,
    so the holdout stays untouched for permutation importance, SHAP and what-if analyses."""
    y, fold_ids = np.asarray(y), np.asarray(fold_ids)
    if len(y) != len(X) or len(fold_ids) != len(X):
        raise ValueError(f"X, y and fold_ids lengths differ: {len(X)}, {len(y)}, {len(fold_ids)}")
    holdout_idx = np.flatnonzero(fold_ids == holdout_fold)
    train_idx = np.flatnonzero(fold_ids != holdout_fold)
    if len(holdout_idx) == 0 or len(train_idx) == 0:
        raise ValueError(f"holdout fold {holdout_fold} leaves an empty train or holdout set")
    fit_idx, stop_idx = train_test_split(train_idx, test_size=EARLY_STOP_FRACTION,
                                         stratify=y[train_idx], random_state=seed)
    model = LightGBMModel(**driver_params(**(params or {})))
    model.fit(X.iloc[fit_idx], y[fit_idx], X.iloc[stop_idx], y[stop_idx])
    holdout_auc = auc(y[holdout_idx], model.predict(X.iloc[holdout_idx]))
    best = int(model.model.best_iteration_ or model.model.n_estimators)
    log.info("driver model: %d train rows, %d holdout rows, best iteration %d, holdout AUC %.5f",
             len(train_idx), len(holdout_idx), best, holdout_auc)
    return DriverModel(model, train_idx, holdout_idx, holdout_auc, best)


# --- permutation importance -------------------------------------------------------------------

def _permuted(X: pd.DataFrame, columns: list[str], rng: np.random.Generator) -> pd.DataFrame:
    """Copy of X with `columns` row-shuffled together (one shared order keeps their joint distribution)."""
    order = rng.permutation(len(X))
    return X.assign(**{col: X[col].iloc[order].set_axis(X.index) for col in columns})


def permutation_importance_grouped(model: Predictor, X: pd.DataFrame, y: np.ndarray,
                                   groups: dict[str, list[str]], n_repeats: int = 3,
                                   seed: int = SEED) -> pd.DataFrame:
    """AUC drop when all columns of each group are jointly permuted; one row per group."""
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")
    missing = {col for cols in groups.values() for col in cols} - set(X.columns)
    if missing:
        raise ValueError(f"group columns not in X: {sorted(missing)}")
    y = np.asarray(y)
    base = auc(y, model.predict(X))
    rng = np.random.default_rng(seed)
    rows = []
    for name, columns in groups.items():
        drops = [base - auc(y, model.predict(_permuted(X, columns, rng))) for _ in range(n_repeats)]
        rows.append({"name": name, "n_columns": len(columns), "columns": ";".join(columns),
                     "auc_base": base, "auc_drop": float(np.mean(drops)),
                     "auc_drop_std": float(np.std(drops, ddof=1)) if n_repeats > 1 else 0.0})
    return pd.DataFrame(rows).sort_values("auc_drop", ascending=False, ignore_index=True)


def permutation_importance_single(model: Predictor, X: pd.DataFrame, y: np.ndarray,
                                  features: list[str] | None = None, n_repeats: int = 3,
                                  seed: int = SEED) -> pd.DataFrame:
    """AUC drop when each feature is permuted on its own."""
    features = list(X.columns) if features is None else list(features)
    return permutation_importance_grouped(model, X, y, {f: [f] for f in features}, n_repeats, seed)


def single_vs_group(grouped: pd.DataFrame, single: pd.DataFrame,
                    groups: dict[str, list[str]]) -> pd.DataFrame:
    """Joint group drop vs the sum of its members' single-feature drops.

    ratio_group_to_sum > 1 means permuting the members one at a time understates the group:
    the model still reads part of the signal from the overlapping ratings left intact.
    """
    group_drop = grouped.set_index("name")["auc_drop"]
    single_drop = single.set_index("name")["auc_drop"]
    rows = []
    for name, columns in groups.items():
        missing = set(columns) - set(single_drop.index)
        if missing or name not in group_drop.index:
            raise ValueError(f"group {name!r} lacks importances for {sorted(missing) or name}")
        members = single_drop.loc[columns]
        total = float(members.sum())
        rows.append({"group": name, "n_columns": len(columns), "group_drop": float(group_drop[name]),
                     "sum_single_drop": total, "max_single_drop": float(members.max()),
                     "top_single": members.idxmax(),
                     "ratio_group_to_sum": float(group_drop[name]) / total if total > 0 else np.nan})
    return pd.DataFrame(rows).sort_values("group_drop", ascending=False, ignore_index=True)


# --- segment TreeSHAP ---------------------------------------------------------------------------

def segment_labels(X: pd.DataFrame, segment_cols=SEGMENT_COLUMNS) -> pd.Series:
    """'Business | Business travel'-style label per row."""
    parts = [X[col].astype(str) for col in segment_cols]
    return parts[0].str.cat(parts[1:], sep=SEGMENT_SEP).rename("segment")


def _contributions(model, X: pd.DataFrame) -> np.ndarray:
    """TreeSHAP values (log-odds) per row and feature from LightGBM; the bias column is dropped."""
    estimator = getattr(model, "model", model)
    return np.asarray(estimator.predict(X, pred_contrib=True))[:, : X.shape[1]]


def _top1_share(abs_shap: np.ndarray, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    """Share of row-bootstrap resamples in which each column has the largest mean |SHAP|."""
    if n_boot <= 0:
        return np.full(abs_shap.shape[1], np.nan)
    n = len(abs_shap)
    weights = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot)
    winners = np.argmax(weights @ abs_shap, axis=1)
    return np.bincount(winners, minlength=abs_shap.shape[1]) / n_boot


def segment_shap(model, X: pd.DataFrame, segment_cols=SEGMENT_COLUMNS,
                 services: list[str] = SERVICE_RATINGS, max_rows: int = MAX_SHAP_ROWS,
                 n_boot: int = N_TOP1_BOOTSTRAP, seed: int = SEED) -> pd.DataFrame:
    """Mean |TreeSHAP| of each service per segment (≤ max_rows rows each), ranked within segment.

    `top1_share` is the share of row-bootstrap resamples in which the service ranks #1: a measure
    of how stable the "fix-first" answer is given the model (it ignores refitting noise).
    """
    rng = np.random.default_rng(seed)
    positions = [X.columns.get_loc(s) for s in services]
    labels = segment_labels(X, segment_cols)
    frames = []
    for segment, rows in sorted(labels.groupby(labels).indices.items()):
        if len(rows) > max_rows:
            rows = np.sort(rng.choice(rows, max_rows, replace=False))
        abs_shap = np.abs(_contributions(model, X.iloc[rows])[:, positions])
        frame = pd.DataFrame({"segment": segment, "feature": services,
                              "mean_abs_shap": abs_shap.mean(axis=0), "n_rows": len(rows)})
        frame["rank"] = frame["mean_abs_shap"].rank(ascending=False, method="min").astype(int)
        frame["top1_share"] = _top1_share(abs_shap, rng, n_boot)
        frames.append(frame)
        del abs_shap
    return pd.concat(frames, ignore_index=True)


# --- agreement of rankings --------------------------------------------------------------------

def _top(scores: pd.Series, k: int) -> list[str]:
    return list(scores.sort_values(ascending=False, kind="stable").index[:k])


def compare_rankings(a: pd.Series, b: pd.Series, k: int = TOP_K, names=("syn", "real")) -> dict:
    """Spearman rho, top-k overlap and #1 match for two importance scores over the same features."""
    if set(a.index) != set(b.index):
        raise ValueError(f"rankings cover different features: {sorted(set(a.index) ^ set(b.index))}")
    b = b.loc[a.index]
    top_a, top_b = _top(a, k), _top(b, k)
    first, second = names
    return {"spearman": float(spearmanr(a.to_numpy(), b.to_numpy()).statistic),
            f"top{k}_overlap": len(set(top_a) & set(top_b)),
            f"top1_{first}": top_a[0], f"top1_{second}": top_b[0], "top1_match": top_a[0] == top_b[0],
            f"top{k}_{first}": ";".join(top_a), f"top{k}_{second}": ";".join(top_b)}


def ranking_agreement(syn_rank: pd.DataFrame, real_rank: pd.DataFrame, score: str = "mean_abs_shap",
                      k: int = TOP_K) -> pd.DataFrame:
    """Per segment agreement between two `segment_shap` frames (synthetic vs real model)."""
    syn = syn_rank.pivot(index="feature", columns="segment", values=score)
    real = real_rank.pivot(index="feature", columns="segment", values=score)
    segments = sorted(set(syn.columns) & set(real.columns))
    if not segments:
        raise ValueError("the two rankings share no segment")
    rows = [{"segment": s, **compare_rankings(syn[s], real[s], k)} for s in segments]
    frame = pd.DataFrame(rows)
    if "top1_share" in syn_rank and "top1_share" in real_rank:
        frame["top1_share_syn"] = _share_of(syn_rank, frame["segment"], frame["top1_syn"])
        frame["top1_share_real"] = _share_of(real_rank, frame["segment"], frame["top1_real"])
    return frame


def _share_of(rank: pd.DataFrame, segments: pd.Series, features: pd.Series) -> list[float]:
    shares = rank.set_index(["segment", "feature"])["top1_share"]
    return [float(shares[(s, f)]) for s, f in zip(segments, features)]


# --- what-if uplift -----------------------------------------------------------------------------

def what_if_uplift(model: Predictor, X: pd.DataFrame, service: str, delta: int = 1,
                   segment_cols=SEGMENT_COLUMNS, base_pred: np.ndarray | None = None) -> pd.DataFrame:
    """Mean change in predicted P(satisfied) per segment when `service` is raised by `delta`
    (capped at 5) for passengers who rated it 1-3; 0 = "not applicable" is left unchanged.

    `mean_uplift` averages over those eligible passengers; `segment_uplift` over everyone in the
    segment. Model-based and associational: it is not the causal effect of improving the service.
    """
    if delta < 1:
        raise ValueError("delta must be a positive number of rating points")
    ratings = X[service].to_numpy()
    eligible = (ratings >= RATING_MIN) & (ratings <= WHAT_IF_MAX_START)
    p_before = model.predict(X) if base_pred is None else np.asarray(base_pred)
    rows = X.iloc[np.flatnonzero(eligible)]
    raised = rows.assign(**{service: np.minimum(rows[service] + delta, RATING_MAX)})
    change = np.zeros(len(X))
    change[eligible] = model.predict(raised) - p_before[eligible]
    frame = pd.DataFrame({"segment": segment_labels(X, segment_cols).to_numpy(), "eligible": eligible,
                          "change": change, "p_before": p_before})
    by_segment = frame.groupby("segment")
    on_eligible = frame[frame["eligible"]].groupby("segment")
    out = pd.DataFrame({"n_rows": by_segment.size(), "n_eligible": by_segment["eligible"].sum(),
                        "p_before_eligible": on_eligible["p_before"].mean(),
                        "mean_uplift": on_eligible["change"].mean(),
                        "segment_uplift": by_segment["change"].mean()})
    out["share_eligible"] = out["n_eligible"] / out["n_rows"]
    return out.reset_index().assign(service=service, delta=delta)


def what_if_table(model: Predictor, X: pd.DataFrame, services: list[str] = SERVICE_RATINGS,
                  delta: int = 1, segment_cols=SEGMENT_COLUMNS) -> pd.DataFrame:
    """`what_if_uplift` for every service, reusing one baseline prediction."""
    base = model.predict(X)
    return pd.concat([what_if_uplift(model, X, s, delta, segment_cols, base) for s in services],
                     ignore_index=True)
