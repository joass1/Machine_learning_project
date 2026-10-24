"""Unit tests for the RQ3 driver analyses (airsat.drivers) on toy data with a known signal.

The toy target depends on "Online boarding" (strongly) and "Type of Travel", nothing else.
"""
import numpy as np
import pandas as pd
import pytest

from airsat import drivers
from airsat.config import SERVICE_GROUPS, SERVICE_RATINGS, TARGET
from airsat.data import to_model_frame
from airsat.folds import make_folds
from tests.conftest import make_toy_frame

FAST_PARAMS = dict(n_estimators=200, learning_rate=0.1, num_leaves=15, min_child_samples=20)


@pytest.fixture(scope="module")
def fitted():
    df = make_toy_frame(4000, seed=0)
    X, y = to_model_frame(df), df[TARGET].to_numpy()
    folds = make_folds(y)
    driver = drivers.train_driver_model(X, y, folds, holdout_fold=0, params=FAST_PARAMS)
    holdout = driver.holdout_idx
    return driver, folds, X.iloc[holdout].reset_index(drop=True), y[holdout]


class MeanOfColumns:
    """Fake model: the score is the mean of a few columns (lets us control redundancy exactly)."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.columns].to_numpy(float).mean(axis=1)


def test_driver_params_cap_threads():
    assert drivers.driver_params()["n_jobs"] <= drivers.DRIVER_N_JOBS
    assert drivers.driver_params(n_jobs=-1)["n_jobs"] == drivers.DRIVER_N_JOBS
    assert drivers.driver_params(n_jobs=2)["n_jobs"] == 2


def test_train_driver_model_holds_out_one_fold(fitted):
    driver, folds, _, _ = fitted
    np.testing.assert_array_equal(driver.holdout_idx, np.flatnonzero(folds == 0))
    assert not set(driver.train_idx) & set(driver.holdout_idx)
    assert len(driver.train_idx) + len(driver.holdout_idx) == len(folds)
    assert driver.holdout_auc > 0.75 and driver.best_iteration >= 1


def test_train_driver_model_rejects_bad_inputs():
    df = make_toy_frame(300)
    X, y = to_model_frame(df), df[TARGET].to_numpy()
    with pytest.raises(ValueError, match="lengths differ"):
        drivers.train_driver_model(X, y[:-1], make_folds(y))
    with pytest.raises(ValueError, match="empty"):
        drivers.train_driver_model(X, y, make_folds(y), holdout_fold=9)


def test_grouped_permutation_ranks_the_online_boarding_group_first(fitted):
    driver, _, X, y = fitted
    grouped = drivers.permutation_importance_grouped(driver.model, X, y, SERVICE_GROUPS, n_repeats=2)
    assert list(grouped["name"])[0] == "Digital"
    assert set(grouped["name"]) == set(SERVICE_GROUPS)
    assert grouped.set_index("name").at["Profile", "auc_drop"] > 0.01  # Type of Travel matters too
    assert np.allclose(grouped["auc_base"], driver.holdout_auc)


def test_single_permutation_ranks_online_boarding_first(fitted):
    driver, _, X, y = fitted
    single = drivers.permutation_importance_single(driver.model, X, y, n_repeats=1)
    assert single["name"].iloc[0] == "Online boarding"
    assert len(single) == X.shape[1] and (single["auc_drop_std"] == 0).all()


def test_permutation_validates_inputs(fitted):
    driver, _, X, y = fitted
    with pytest.raises(ValueError, match="n_repeats"):
        drivers.permutation_importance_grouped(driver.model, X, y, SERVICE_GROUPS, n_repeats=0)
    with pytest.raises(ValueError, match="not in X"):
        drivers.permutation_importance_grouped(driver.model, X, y, {"g": ["nope"]})


def test_permuted_shuffles_group_columns_together_without_mutating():
    X = to_model_frame(make_toy_frame(200))
    before = X.copy()
    out = drivers._permuted(X, ["Online boarding", "Class"], np.random.default_rng(0))
    pd.testing.assert_frame_equal(X, before)
    assert out["Class"].dtype == X["Class"].dtype
    pairs = lambda f: sorted(zip(f["Online boarding"], f["Class"].astype(str)))  # noqa: E731
    assert pairs(out) == pairs(X)  # same (rating, class) pairs, just reordered rows
    assert not out["Online boarding"].equals(X["Online boarding"])
    pd.testing.assert_series_equal(out["Seat comfort"], X["Seat comfort"])


def test_single_feature_drops_understate_a_redundant_group():
    df = make_toy_frame(3000, seed=1).assign(**{"Inflight wifi service": lambda d: d["Online boarding"]})
    X, y = to_model_frame(df), df[TARGET].to_numpy()
    model = MeanOfColumns(["Online boarding", "Inflight wifi service"])  # two copies of one signal
    groups = {"Digital": ["Online boarding", "Inflight wifi service"]}
    grouped = drivers.permutation_importance_grouped(model, X, y, groups)
    single = drivers.permutation_importance_single(model, X, y, groups["Digital"])
    summary = drivers.single_vs_group(grouped, single, groups).iloc[0]
    assert summary["group_drop"] > summary["sum_single_drop"] > 0
    assert summary["ratio_group_to_sum"] > 1


def test_single_vs_group_arithmetic_and_validation():
    grouped = pd.DataFrame({"name": ["A", "B"], "auc_drop": [0.10, 0.0]})
    single = pd.DataFrame({"name": ["a1", "a2", "b1"], "auc_drop": [0.02, 0.03, 0.0]})
    out = drivers.single_vs_group(grouped, single, {"A": ["a1", "a2"], "B": ["b1"]}).set_index("group")
    assert out.at["A", "sum_single_drop"] == pytest.approx(0.05)
    assert out.at["A", "ratio_group_to_sum"] == pytest.approx(2.0)
    assert out.at["A", "top_single"] == "a2" and np.isnan(out.at["B", "ratio_group_to_sum"])
    with pytest.raises(ValueError, match="lacks"):
        drivers.single_vs_group(grouped, single, {"A": ["a1", "zz"]})


def test_segment_labels_combine_class_and_purpose():
    X = to_model_frame(make_toy_frame(50))
    labels = drivers.segment_labels(X)
    assert labels.iloc[0] == f"{X['Class'].iloc[0]} | {X['Type of Travel'].iloc[0]}"


def test_segment_shap_ranks_online_boarding_first_in_every_segment(fitted):
    driver, _, X, _ = fitted
    shap = drivers.segment_shap(driver.model, X, max_rows=60, n_boot=50)
    assert shap["segment"].nunique() == 6
    assert (shap.groupby("segment").size() == len(SERVICE_RATINGS)).all()
    assert (shap["n_rows"] <= 60).all()
    top = shap[shap["rank"] == 1]
    assert (top["feature"] == "Online boarding").all() and (top["top1_share"] > 0.9).all()
    assert np.allclose(shap.groupby("segment")["top1_share"].sum(), 1.0)
    assert np.isnan(drivers._top1_share(np.ones((3, 2)), np.random.default_rng(0), n_boot=0)).all()


def test_compare_rankings_on_known_orders():
    features = [f"s{i}" for i in range(13)]
    a = pd.Series(np.arange(13, 0, -1.0), index=features)  # s0 most important
    same = drivers.compare_rankings(a, a.copy())
    assert same["spearman"] == pytest.approx(1.0) and same["top3_overlap"] == 3 and same["top1_match"]
    rev = drivers.compare_rankings(a, pd.Series(np.arange(1, 14.0), index=features))
    assert rev["spearman"] == pytest.approx(-1.0) and rev["top3_overlap"] == 0 and not rev["top1_match"]
    assert rev["top1_syn"] == "s0" and rev["top1_real"] == "s12"
    with pytest.raises(ValueError, match="different features"):
        drivers.compare_rankings(a, a.rename({"s0": "x"}))


def _rank_frame(scores: dict[str, list[float]], features: list[str]) -> pd.DataFrame:
    rows = [{"segment": seg, "feature": f, "mean_abs_shap": v, "top1_share": float(v == max(vals))}
            for seg, vals in scores.items() for f, v in zip(features, vals)]
    return pd.DataFrame(rows)


def test_ranking_agreement_per_segment():
    features = ["a", "b", "c", "d"]
    syn = _rank_frame({"S1": [4, 3, 2, 1], "S2": [4, 3, 2, 1]}, features)
    real = _rank_frame({"S1": [4, 3, 2, 1], "S2": [3, 4, 2, 1]}, features)
    out = drivers.ranking_agreement(syn, real).set_index("segment")
    assert out.at["S1", "top1_match"] and out.at["S1", "spearman"] == pytest.approx(1.0)
    assert not out.at["S2", "top1_match"] and out.at["S2", "top3_overlap"] == 3
    assert out.at["S2", "spearman"] == pytest.approx(0.8)
    assert out.at["S2", "top1_real"] == "b" and out.at["S2", "top1_share_real"] == 1.0
    with pytest.raises(ValueError, match="no segment"):
        drivers.ranking_agreement(syn, real[real["segment"] == "S1"].assign(segment="other"))


def test_what_if_is_exact_for_a_linear_model():
    X = to_model_frame(make_toy_frame(500, seed=3))
    model = MeanOfColumns(["Seat comfort"])  # P rises by exactly 1 per rating point
    out = drivers.what_if_uplift(model, X, "Seat comfort", delta=1)
    eligible = X["Seat comfort"].between(1, 3)
    assert out["n_eligible"].sum() == eligible.sum() and out["n_rows"].sum() == len(X)
    assert np.allclose(out["mean_uplift"], 1.0)
    assert np.allclose(out["segment_uplift"], out["share_eligible"])
    capped = drivers.what_if_uplift(model, X, "Seat comfort", delta=4)
    assert np.allclose(capped["p_before_eligible"] + capped["mean_uplift"], 5.0)  # capped at 5
    with pytest.raises(ValueError, match="delta"):
        drivers.what_if_uplift(model, X, "Seat comfort", delta=0)


def test_what_if_on_online_boarding_is_positive_in_every_segment(fitted):
    driver, _, X, _ = fitted
    table = drivers.what_if_table(driver.model, X, ["Online boarding", "Gate location"])
    boarding = table[table["service"] == "Online boarding"].set_index("segment")
    gate = table[table["service"] == "Gate location"].set_index("segment")
    assert len(boarding) == 6 and (boarding["mean_uplift"] > 0.02).all()
    assert (boarding["mean_uplift"] > gate["mean_uplift"].abs()).all()
