"""Smoke tests for the RQ3 figures, plus the model input helpers and the CatBoost adapter."""
import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from airsat import drivers, drivers_plots  # noqa: E402
from airsat.config import SERVICE_GROUPS, TARGET  # noqa: E402
from airsat.data import to_model_frame  # noqa: E402
from airsat.folds import make_folds  # noqa: E402
from airsat.metrics import auc  # noqa: E402
from airsat.models import catboost_cat, inputs  # noqa: E402
from tests.conftest import make_toy_frame  # noqa: E402

FAST_PARAMS = dict(n_estimators=100, learning_rate=0.1, num_leaves=15, min_child_samples=20)
SHAP_VIEWS = {"syn_on_syn": ("synthetic", "synthetic"), "real_on_real": ("real", "real"),
              "syn_on_real": ("synthetic", "real")}
COMPARISONS = {"shap_own_data": ("syn_on_syn", "real_on_real"),
               "shap_real_passengers": ("syn_on_real", "real_on_real")}


def _overlap(model, X, y) -> pd.DataFrame:
    grouped = drivers.permutation_importance_grouped(model, X, y, SERVICE_GROUPS, n_repeats=1)
    single = drivers.permutation_importance_single(model, X, y, n_repeats=1)
    return drivers.single_vs_group(grouped, single, SERVICE_GROUPS)


@pytest.fixture(scope="module")
def rq3_tables():
    """The four RQ3 tables, built as experiments/rq3_drivers.py builds them, on two toy datasets."""
    models, holdouts = {}, {}
    for name, seed in (("synthetic", 0), ("real", 1)):
        df = make_toy_frame(3000, seed)
        X, y = to_model_frame(df), df[TARGET].to_numpy()
        models[name] = drivers.train_driver_model(X, y, make_folds(y), 0, params=FAST_PARAMS).model
        idx = make_folds(y) == 0
        holdouts[name] = (X[idx].reset_index(drop=True), y[idx])
    overlap = pd.concat([_overlap(models[n], *holdouts[n]).assign(dataset=n) for n in models])
    shap = pd.concat([drivers.segment_shap(models[t], holdouts[p][0], max_rows=150, n_boot=20).assign(view=v)
                      for v, (t, p) in SHAP_VIEWS.items()])
    agreement = pd.concat([drivers.ranking_agreement(shap[shap["view"] == a], shap[shap["view"] == b])
                           .assign(comparison=c) for c, (a, b) in COMPARISONS.items()])
    whatif = pd.concat([drivers.what_if_table(models[m], holdouts["real"][0]).assign(model=m, passengers="real")
                        for m in models])
    return overlap, shap, agreement, whatif


def test_rq3_figures_are_written(rq3_tables, tmp_path):
    overlap, shap, agreement, whatif = rq3_tables
    drivers_plots.fig_permutation(overlap, tmp_path / "perm.png")
    drivers_plots.fig_shap_heatmap(shap, tmp_path / "shap.png")
    drivers_plots.fig_agreement(agreement, tmp_path / "agree.png")
    drivers_plots.fig_whatif(whatif, tmp_path / "whatif.png")
    assert all((tmp_path / f).stat().st_size > 0 for f in ("perm.png", "shap.png", "agree.png", "whatif.png"))
    assert drivers_plots.segment_display("Eco | Personal Travel", " · ").count("·") == 1


def test_stratified_subsample_keeps_order_and_class_balance(toy):
    X, y = toy.drop(columns=TARGET), toy[TARGET].to_numpy()
    X_sub, y_sub = inputs.stratified_subsample(X, y, 200)
    assert len(y_sub) == 200 and X_sub.index.is_monotonic_increasing
    assert abs(y_sub.mean() - y.mean()) < 0.01
    assert inputs.stratified_subsample(X, y, None)[0] is X


def test_numeric_frame_fills_arrival_delay_and_keeps_categories(toy):
    X = to_model_frame(toy)
    out = inputs.numeric_frame(X)
    assert out["Arrival Delay in Minutes"].isna().sum() == 0
    missing = X["Arrival Delay in Minutes"].isna()
    assert (out.loc[missing, "Arrival Delay in Minutes"] == X.loc[missing, "Departure Delay in Minutes"]).all()
    assert out["Age"].dtype == np.float32 and isinstance(out["Class"].dtype, pd.CategoricalDtype)
    assert X["Arrival Delay in Minutes"].isna().any()  # input left untouched


def test_category_codes_are_all_numeric(toy):
    codes = inputs.category_codes(to_model_frame(toy))
    assert (codes.dtypes == np.float32).all()
    assert set(codes["Class"].dropna().unique()) <= {0.0, 1.0, 2.0}


def test_all_categorical_rejects_fractional_values(toy):
    X = to_model_frame(toy)
    levels = catboost_cat.all_categorical(X)
    assert (levels.dtypes == np.int64).all()
    n_missing = X["Arrival Delay in Minutes"].isna().sum()
    assert (levels["Arrival Delay in Minutes"] == catboost_cat.MISSING_LEVEL).sum() == n_missing
    with pytest.raises(ValueError, match="non-integer"):
        catboost_cat.all_categorical(X.assign(Age=X["Age"] + 0.5))


def test_catboost_all_categorical_learns_on_cpu(toy):
    X, y = to_model_frame(toy), toy[TARGET].to_numpy()
    params = {**catboost_cat.CATBOOST_PARAMS, "iterations": 60, "task_type": "CPU", "thread_count": 2}
    model = catboost_cat.CatBoostAllCategorical(**params).fit(X[:450], y[:450], X[450:], y[450:])
    assert auc(y[450:], model.predict(X[450:])) > 0.7
    assert catboost_cat.task_type() in {"CPU", "GPU"}
