"""RQ2 synthetic-data fidelity: what did the generator distort, and does it matter?

Pure functions (data in, numbers/frames out):
- marginal distortion per feature (KS for numerics, total variation distance for discrete columns),
  delay summaries and the Spearman-correlation difference between the two datasets;
- detectability: adversarial AUC of a LightGBM model separating synthetic from real rows,
  single-feature adversarial AUCs and gain importance;
- target dependence: univariate AUC of an out-of-fold exact-value target-mean encoding;
- label noise: rows where a confident real-trained model disagrees with the label;
- model-selection fidelity: rank agreement of model AUCs across settings.
The utility learning curve lives in `airsat.fidelity_utility`.
"""
import json
import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold

from airsat.config import DELAYS, FEATURES, NUMERIC, OOF_DIR, SEED, SERVICE_RATINGS, TARGET
from airsat.encoders import MISSING_KEY, smoothed_means
from airsat.metrics import auc

log = logging.getLogger(__name__)

ADVERSARIAL_N = 100_000
ADVERSARIAL_PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=63, min_child_samples=40,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                          random_state=SEED, verbose=-1, n_jobs=4)
TE_SMOOTHING = 20.0
DELAY_EDGES = (-np.inf, 0, 15, 60, 180, np.inf)
DELAY_LABELS = ("0", "1-15", "16-60", "61-180", ">180")
NOISE_THRESHOLDS = (0.9, 0.95, 0.99)
SETTINGS = ("S->S", "R->R", "S->R", "R->S")
SETTING_ALIASES = {"syn_cv": "S->S", "real_cv": "R->R", "syn_to_real": "S->R", "real_to_syn": "R->S"}
RANKING_COLUMNS = ["setting_a", "setting_b", "n_models", "kendall_tau", "kendall_p", "spearman_rho",
                   "spearman_p", "top_a", "top_b", "same_top", "models"]


# ---------- marginal distortion ----------
def feature_kind(col: str) -> str:
    if col in NUMERIC:
        return "numeric"
    return "rating" if col in SERVICE_RATINGS else "categorical"


def _finite(values) -> np.ndarray:
    arr = np.asarray(pd.to_numeric(pd.Series(values)), dtype=float)
    return np.sort(arr[~np.isnan(arr)])


def ks_gap(a, b) -> tuple[float, float, float, float]:
    """Two-sample KS statistic sup|F_a - F_b| and where it is reached: (stat, x, F_a(x), F_b(x))."""
    a, b = _finite(a), _finite(b)
    grid = np.union1d(a, b)
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    k = int(np.argmax(np.abs(cdf_a - cdf_b)))
    return float(abs(cdf_a[k] - cdf_b[k])), float(grid[k]), float(cdf_a[k]), float(cdf_b[k])


def _shares(values) -> pd.Series:
    counts = pd.Series(np.asarray(values, dtype=object)).value_counts(normalize=True, dropna=False)
    return counts.rename(index=lambda v: "missing" if pd.isna(v) else v)


def tvd_gap(a, b) -> tuple[float, object, float, float]:
    """Total variation distance 0.5 * sum|p_a - p_b| and the value with the largest share gap."""
    shares = pd.concat([_shares(a), _shares(b)], axis=1, keys=["a", "b"]).fillna(0.0)
    gap = (shares["a"] - shares["b"]).abs()
    top = gap.idxmax()
    return float(0.5 * gap.sum()), top, float(shares.at[top, "a"]), float(shares.at[top, "b"])


def marginal_distortion(syn: pd.DataFrame, real: pd.DataFrame, features=FEATURES) -> pd.DataFrame:
    """Per feature: KS (numerics) or TVD (ratings, categoricals), sorted by distortion.

    `gap_at` is where the gap is largest; `syn_at`/`real_at` are the CDF values there (KS)
    or the shares of that value (TVD).
    """
    rows = []
    for col in features:
        kind = feature_kind(col)
        stat, at, syn_at, real_at = (ks_gap if kind == "numeric" else tvd_gap)(syn[col], real[col])
        is_ordered = kind != "categorical"
        rows.append({"feature": col, "kind": kind, "metric": "KS" if kind == "numeric" else "TVD",
                     "distortion": stat, "gap_at": at, "syn_at": syn_at, "real_at": real_at,
                     "syn_mean": float(syn[col].mean()) if is_ordered else np.nan,
                     "real_mean": float(real[col].mean()) if is_ordered else np.nan})
    return pd.DataFrame(rows).sort_values("distortion", ascending=False, ignore_index=True)


def exact_copies(syn: pd.DataFrame, real: pd.DataFrame, features=FEATURES) -> dict:
    """How many synthetic rows reproduce a real row on every feature (a memorisation check)."""
    def row_hashes(df: pd.DataFrame) -> pd.Series:
        return pd.util.hash_pandas_object(df[list(features)].astype(str), index=False)

    copied = row_hashes(syn).isin(set(row_hashes(real)))
    return {"n_synthetic": int(len(syn)), "n_copies": int(copied.sum()), "share": float(copied.mean())}


def _delay_stats(values: pd.Series) -> dict:
    v = values.dropna()
    return {"mean": v.mean(), "median": v.median(), "p90": v.quantile(0.9), "p99": v.quantile(0.99),
            "max": v.max(), "share_zero": (v == 0).mean(), "share_delayed": (v > 0).mean(),
            "share_over_15": (v > 15).mean(), "share_missing": values.isna().mean()}


def delay_summary(syn: pd.DataFrame, real: pd.DataFrame, columns=DELAYS) -> pd.DataFrame:
    """Mean, median, tail quantiles and zero/delayed shares of each delay column, per dataset."""
    rows = [{"feature": col, "dataset": name, **_delay_stats(df[col])}
            for col in columns for name, df in (("synthetic", syn), ("real", real))]
    return pd.DataFrame(rows)


def delay_bins(syn: pd.DataFrame, real: pd.DataFrame, columns=DELAYS) -> pd.DataFrame:
    """Share of non-missing rows in each delay band (minutes), per column and dataset."""
    frames = []
    for col in columns:
        for name, df in (("synthetic", syn), ("real", real)):
            bands = pd.cut(df[col].dropna(), DELAY_EDGES, labels=DELAY_LABELS)
            shares = bands.value_counts(normalize=True, sort=False).reindex(DELAY_LABELS, fill_value=0.0)
            frames.append(pd.DataFrame({"feature": col, "dataset": name, "band": DELAY_LABELS,
                                        "share": shares.to_numpy()}))
    return pd.concat(frames, ignore_index=True)


def correlation_difference(syn: pd.DataFrame, real: pd.DataFrame,
                           columns=(*NUMERIC, *SERVICE_RATINGS, TARGET)) -> pd.DataFrame:
    """Spearman correlation of every column pair in each dataset, sorted by |synthetic - real|."""
    cols = [c for c in columns if c in syn and c in real]
    rho_syn = syn[cols].corr(method="spearman")
    rho_real = real[cols].corr(method="spearman")
    rows = [{"a": a, "b": b, "syn_rho": rho_syn.at[a, b], "real_rho": rho_real.at[a, b],
             "diff": rho_syn.at[a, b] - rho_real.at[a, b]} for a, b in combinations(cols, 2)]
    out = pd.DataFrame(rows)
    return out.reindex(out["diff"].abs().sort_values(ascending=False).index).reset_index(drop=True)


# ---------- out-of-fold exact-value means (shared by detectability and dependence) ----------
def value_keys(series: pd.Series) -> np.ndarray:
    """Numeric key per row for grouping exact values; missing values get MISSING_KEY."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        return series.cat.codes.to_numpy(dtype=float)
    if not pd.api.types.is_numeric_dtype(series):
        return pd.factorize(series)[0].astype(float)
    return series.astype(float).fillna(MISSING_KEY).to_numpy()


def oof_value_means(x: pd.Series, y, n_splits: int = 5, smoothing: float = TE_SMOOTHING,
                    seed: int = SEED) -> np.ndarray:
    """Out-of-fold smoothed mean of y for each exact value of x (unseen values get the fold prior)."""
    keys, y = pd.Series(value_keys(x)), np.asarray(y)
    encoded = np.empty(len(y))
    for tr, va in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(keys, y):
        prior = float(y[tr].mean())
        means = smoothed_means(keys.iloc[tr], pd.Series(y[tr]), prior, smoothing)
        encoded[va] = keys.iloc[va].map(means).fillna(prior).to_numpy()
    return encoded


def univariate_aucs(X: pd.DataFrame, y, features=FEATURES, n_splits: int = 5, seed: int = SEED) -> pd.Series:
    """AUC of each feature's out-of-fold exact-value mean encoding for predicting y."""
    y = np.asarray(y)
    return pd.Series({col: auc(y, oof_value_means(X[col], y, n_splits, seed=seed)) for col in features},
                     name="auc")


# ---------- detectability ----------
@dataclass(frozen=True)
class AdversarialResult:
    auc: float
    fold_aucs: list[float]
    importance: pd.Series  # share of total gain per feature, descending
    n_per_class: int


def balanced_sample(syn_X: pd.DataFrame, real_X: pd.DataFrame, n: int = ADVERSARIAL_N, seed: int = SEED,
                    features: list[str] | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """min(n, len) rows from each frame (columns `features`, default all of syn_X); label 1 = real."""
    k, cols = min(n, len(syn_X), len(real_X)), list(features or syn_X.columns)
    X = pd.concat([syn_X.sample(k, random_state=seed)[cols], real_X.sample(k, random_state=seed)[cols]],
                  ignore_index=True)
    return X, np.r_[np.zeros(k, dtype=int), np.ones(k, dtype=int)]


def adversarial_auc(syn_X: pd.DataFrame, real_X: pd.DataFrame, n: int = ADVERSARIAL_N, n_splits: int = 3,
                    seed: int = SEED, params: dict | None = None,
                    features: list[str] | None = None) -> AdversarialResult:
    """Out-of-fold AUC of LightGBM separating synthetic from real rows (0.5 = indistinguishable)."""
    X, y = balanced_sample(syn_X, real_X, n, seed, features)
    params = {**ADVERSARIAL_PARAMS, **(params or {})}
    oof, gains, fold_scores = np.zeros(len(y)), [], []
    for tr, va in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y):
        model = lgb.LGBMClassifier(**params).fit(X.iloc[tr], y[tr])
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        fold_scores.append(auc(y[va], oof[va]))
        gains.append(model.booster_.feature_importance(importance_type="gain"))
    gain = pd.Series(np.mean(gains, axis=0), index=X.columns)
    share = gain / gain.sum() if gain.sum() > 0 else gain
    log.info("adversarial auc=%.4f folds=%s", auc(y, oof), np.round(fold_scores, 4))
    return AdversarialResult(auc(y, oof), fold_scores, share.sort_values(ascending=False), len(y) // 2)


def single_feature_adversarial(syn_X: pd.DataFrame, real_X: pd.DataFrame, n: int = ADVERSARIAL_N,
                               features=FEATURES, seed: int = SEED) -> pd.DataFrame:
    """Adversarial AUC using one feature at a time (out-of-fold mean of 'is real' per exact value)."""
    X, y = balanced_sample(syn_X, real_X, n, seed)
    aucs = univariate_aucs(X, y, [c for c in features if c in X], n_splits=3, seed=seed)
    return aucs.rename_axis("feature").reset_index().sort_values("auc", ascending=False, ignore_index=True)


# ---------- target dependence ----------
def target_dependence(df: pd.DataFrame, features=FEATURES, n_splits: int = 5, seed: int = SEED) -> pd.Series:
    """Univariate AUC of each feature (out-of-fold target-mean encoding) for predicting satisfaction."""
    return univariate_aucs(df, df[TARGET].to_numpy(), features, n_splits, seed)


def compare_target_dependence(syn: pd.DataFrame, real: pd.DataFrame, n: int | None = None,
                              features=FEATURES, seed: int = SEED) -> pd.DataFrame:
    """Target dependence on equal-size samples of each dataset (default: the smaller size).

    Equal sizes matter: an out-of-fold encoding of a high-cardinality column gets more accurate
    with more rows, which would otherwise favour the larger (synthetic) dataset.
    """
    n = min(len(syn), len(real)) if n is None else min(n, len(syn), len(real))
    syn_auc = target_dependence(syn.sample(n, random_state=seed), features, seed=seed)
    real_auc = target_dependence(real.sample(n, random_state=seed), features, seed=seed)
    out = pd.DataFrame({"syn_auc": syn_auc, "real_auc": real_auc}).rename_axis("feature")
    out["diff"] = out["syn_auc"] - out["real_auc"]
    out["n_rows"] = n
    return out.reindex(out["diff"].abs().sort_values(ascending=False).index).reset_index()


# ---------- label noise ----------
def label_noise_estimate(y, p, threshold: float = 0.95) -> dict:
    """Share of rows where the model is confident (p > t or p < 1 - t) and disagrees with the label."""
    if not 0.5 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0.5, 1), got {threshold}")
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    if len(y) != len(p):
        raise ValueError(f"y has {len(y)} rows but p has {len(p)}")
    confident = (p > threshold) | (p < 1 - threshold)
    flagged = confident & ((p > 0.5).astype(int) != y)
    return {"threshold": threshold, "n": int(len(y)), "confident_share": float(confident.mean()),
            "flagged_share": float(flagged.mean()),
            "flagged_among_confident": float(flagged.sum() / confident.sum()) if confident.any() else np.nan}


def compare_label_noise(y_syn, p_syn, y_real, p_real, thresholds=NOISE_THRESHOLDS) -> pd.DataFrame:
    """Label-noise estimate on synthetic rows vs held-out real rows, and their ratio, per threshold."""
    rows = []
    for t in thresholds:
        syn, real = label_noise_estimate(y_syn, p_syn, t), label_noise_estimate(y_real, p_real, t)
        row = {"threshold": t}
        for name, est in (("syn", syn), ("real", real)):
            row.update({f"{name}_{k}": v for k, v in est.items() if k != "threshold"})
        row["ratio"] = syn["flagged_share"] / real["flagged_share"] if real["flagged_share"] > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ---------- model-selection fidelity ----------
def _setting_name(col) -> str:
    name = str(col).strip().replace("→", "->").replace(" ", "")
    return SETTING_ALIASES.get(name, name)


def _model_index(df: pd.DataFrame) -> pd.DataFrame:
    if "key" in df:
        return df.set_index("key")
    if "model" in df and "feature_set" in df:
        return df.assign(key=df["model"].astype(str) + "__" + df["feature_set"].astype(str)).set_index("key")
    for col in ("model", "name"):
        if col in df:
            return df.set_index(col)
    return df


def normalise_ladder(table: pd.DataFrame) -> pd.DataFrame:
    """Wide table: one row per model, one column per setting present (S->S, R->R, S->R, R->S).

    Accepts wide tables with arrow or store-style column names (syn_cv, real_cv, syn_to_real, ...)
    and long tables with `setting` and `auc` columns.
    """
    df = table.copy()
    if {"setting", "auc"} <= set(df.columns):
        df = _model_index(df.assign(setting=df["setting"].map(_setting_name)))
        df = df.groupby([df.index, "setting"])["auc"].mean().unstack("setting")
    else:
        df = _model_index(df.rename(columns=_setting_name))
    keep = [s for s in SETTINGS if s in df.columns]
    wide = df[keep].apply(pd.to_numeric, errors="coerce")
    return wide.groupby(level=0).mean().dropna(how="all")


def ranking_agreement(table: pd.DataFrame) -> pd.DataFrame:
    """Kendall tau and Spearman rho between the model rankings of every pair of settings.

    Each pair uses the models scored in both settings; fewer than 3 such models gives NaN.
    """
    wide = normalise_ladder(table)
    rows = []
    for a, b in combinations(wide.columns, 2):
        pair = wide[[a, b]].dropna()
        row = {"setting_a": a, "setting_b": b, "n_models": len(pair), "models": ";".join(map(str, pair.index))}
        if len(pair) >= 3:
            tau, rho = stats.kendalltau(pair[a], pair[b]), stats.spearmanr(pair[a], pair[b])
            row.update(kendall_tau=float(tau.statistic), kendall_p=float(tau.pvalue),
                       spearman_rho=float(rho.statistic), spearman_p=float(rho.pvalue),
                       top_a=pair[a].idxmax(), top_b=pair[b].idxmax(),
                       same_top=bool(pair[a].idxmax() == pair[b].idxmax()))
        rows.append(row)
    return pd.DataFrame(rows, columns=RANKING_COLUMNS)


def ladder_from_store(root: Path = OOF_DIR, include_sampled: bool = False) -> pd.DataFrame:
    """Model AUCs per setting from the prediction store's JSON metadata (full runs unless include_sampled)."""
    rows: dict[str, dict] = {}
    for train, own, other, applied in (("synthetic", "S->S", "S->R", "real"), ("real", "R->R", "R->S", "synthetic")):
        for path in sorted((root / train).glob("*.json")) if (root / train).exists() else []:
            meta = json.loads(path.read_text())
            if meta.get("sample") and not include_sampled:
                continue
            row = rows.setdefault(path.stem, {"key": path.stem})
            row[own] = meta.get("cv_auc")
            row[other] = meta.get("applied_auc", {}).get(applied)
    return pd.DataFrame(list(rows.values()), columns=["key", *SETTINGS])
