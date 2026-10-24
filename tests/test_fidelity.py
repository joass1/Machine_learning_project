"""Unit tests for the RQ2 fidelity analyses (toy data; known distortions and known values)."""
import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from scipy import stats  # noqa: E402

from airsat import fidelity as fd  # noqa: E402
from airsat import fidelity_plots as fp  # noqa: E402
from airsat import fidelity_utility as fu  # noqa: E402
from airsat.config import DELAYS, FEATURES, TARGET  # noqa: E402
from airsat.cv import CVResult  # noqa: E402
from airsat.data import to_model_frame  # noqa: E402
from airsat.folds import make_folds  # noqa: E402
from airsat.oof import save_run  # noqa: E402
from tests.conftest import make_toy_frame  # noqa: E402

FAST_ADV = {"n_estimators": 40, "num_leaves": 15, "min_child_samples": 10, "n_jobs": 2}
FAST_LGBM = {"n_estimators": 200, "num_leaves": 7, "min_child_samples": 10, "n_jobs": 2}


def _frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    df = make_toy_frame(n, seed)
    return to_model_frame(df).assign(**{TARGET: df[TARGET].to_numpy()})


def _no_delays(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(**{col: 0.0 for col in DELAYS})


# ---------- marginal distortion ----------
def test_ks_gap_known_values_and_matches_scipy():
    assert fd.ks_gap([0, 0, 0, 0], [1, 1, 1, 1])[0] == 1.0
    stat, at, cdf_a, cdf_b = fd.ks_gap([0] * 5 + [1] * 5, [0] * 8 + [1] * 2 + [np.nan])
    assert (stat, at, cdf_a, cdf_b) == pytest.approx((0.3, 0.0, 0.5, 0.8))
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=300), rng.normal(0.3, size=200)
    assert fd.ks_gap(a, b)[0] == pytest.approx(stats.ks_2samp(a, b).statistic)


def test_tvd_gap_known_value_and_missing_level():
    stat, top, share_a, share_b = fd.tvd_gap(["A", "A", "B", "B"], ["A", "A", "A", "B"])
    assert stat == pytest.approx(0.25) and share_a + share_b in (pytest.approx(1.25), pytest.approx(0.75))
    stat, top, share_a, share_b = fd.tvd_gap([1, 1, 2, 2], [1, 1, 1, 3])
    assert stat == pytest.approx(0.5) and top == 2 and (share_a, share_b) == (0.5, 0.0)
    assert fd.tvd_gap(["x", None], ["x", "x"])[0] == pytest.approx(0.5)


def test_marginal_distortion_detects_zeroed_delays():
    real = _frame()
    syn = _no_delays(real)
    out = fd.marginal_distortion(syn, real)
    assert len(out) == len(FEATURES) and set(out["metric"]) == {"KS", "TVD"}
    assert set(out.head(2)["feature"]) == set(DELAYS) and (out.iloc[2:]["distortion"] == 0).all()
    dep = out.set_index("feature").loc["Departure Delay in Minutes"]
    assert dep["distortion"] == pytest.approx(1 - (real["Departure Delay in Minutes"] == 0).mean())
    assert dep["gap_at"] == 0.0 and dep["syn_at"] == 1.0
    assert np.isnan(out.set_index("feature").loc["Class", "syn_mean"])
    assert (fd.marginal_distortion(real, real)["distortion"] == 0).all()


def test_delay_summary_and_bins():
    real = _frame()
    syn = _no_delays(real)
    summary = fd.delay_summary(syn, real).set_index(["feature", "dataset"])
    assert summary.loc[("Departure Delay in Minutes", "synthetic"), "share_zero"] == 1.0
    assert summary.loc[("Arrival Delay in Minutes", "real"), "share_missing"] > 0
    bins = fd.delay_bins(syn, real)
    assert bins.groupby(["feature", "dataset"])["share"].sum().round(9).eq(1).all()
    syn_dep = bins[(bins["dataset"] == "synthetic") & (bins["feature"] == "Departure Delay in Minutes")]
    assert syn_dep.set_index("band").loc["0", "share"] == 1.0 and list(syn_dep["band"]) == list(fd.DELAY_LABELS)


def test_correlation_difference_flags_broken_relationship():
    real = _frame(1500)
    syn = real.assign(**{"Online boarding": np.random.default_rng(1).permutation(real["Online boarding"])})
    out = fd.correlation_difference(syn, real)
    assert {out.iloc[0]["a"], out.iloc[0]["b"]} == {"Online boarding", TARGET}
    assert out.iloc[0]["real_rho"] > 0.3 and abs(out.iloc[0]["syn_rho"]) < 0.1
    assert (fd.correlation_difference(real, real)["diff"].abs() < 1e-12).all()


# ---------- exact-value means ----------
def test_value_keys_handles_categoricals_strings_and_missing():
    assert fd.value_keys(pd.Series(pd.Categorical(["b", None], categories=["a", "b"]))).tolist() == [1.0, -1.0]
    assert fd.value_keys(pd.Series(["x", "y", "x"])).tolist() == [0.0, 1.0, 0.0]
    assert fd.value_keys(pd.Series([3.0, np.nan])).tolist() == [3.0, -1.0]


def test_oof_value_means_never_sees_own_label():
    x = pd.Series(np.arange(200))                    # every value unique
    y = np.tile([0, 1], 100)
    encoded = fd.oof_value_means(x, y, smoothing=0.0)
    assert np.allclose(encoded, 0.5)                 # only fold priors (stratified -> 0.5)
    assert fd.auc(y, encoded) == 0.5


# ---------- detectability ----------
def test_single_feature_adversarial_finds_shifted_feature():
    syn, real = _frame(600, 0), _frame(600, 1)
    real = real.assign(Age=real["Age"] + 60)
    out = fd.single_feature_adversarial(syn[FEATURES], real[FEATURES], n=600)
    assert out.iloc[0]["feature"] == "Age" and out.iloc[0]["auc"] > 0.95
    assert out[out["feature"] != "Age"]["auc"].max() < 0.65


def test_adversarial_auc_near_half_for_same_distribution_and_high_for_shift():
    syn, real = _frame(600, 0)[FEATURES], _frame(600, 1)[FEATURES]
    same = fd.adversarial_auc(syn, real, n=600, params=FAST_ADV)
    assert abs(same.auc - 0.5) < 0.1 and len(same.fold_aucs) == 3 and same.n_per_class == 600
    shifted = fd.adversarial_auc(_no_delays(syn), real, n=400, params=FAST_ADV)
    assert shifted.auc > 0.75 and shifted.n_per_class == 400
    assert shifted.importance.index[0] in DELAYS and shifted.importance.sum() == pytest.approx(1.0)


def test_balanced_sample_caps_at_smaller_frame():
    X, y = fd.balanced_sample(_frame(300)[FEATURES], _frame(100, 1)[FEATURES], n=1000)
    assert len(X) == 200 and y.sum() == 100 and list(X.columns) == FEATURES
    X, _ = fd.balanced_sample(_frame(300)[FEATURES], _frame(100, 1)[FEATURES], features=["Age", "Class"])
    assert list(X.columns) == ["Age", "Class"]


# ---------- target dependence ----------
def test_target_dependence_recovers_signal_and_flags_lost_relationship():
    real = _frame(1500, 0)
    dep = fd.target_dependence(real)
    assert dep.idxmax() == "Online boarding" and dep["Online boarding"] > 0.75
    syn = real.assign(**{TARGET: np.random.default_rng(2).permutation(real[TARGET])})
    out = fd.compare_target_dependence(syn, real).set_index("feature")
    assert out.loc["Online boarding", "diff"] < -0.2 and out["n_rows"].iloc[0] == 1500
    assert out.index[0] == "Online boarding"
    assert fd.compare_target_dependence(syn, real, n=500)["n_rows"].iloc[0] == 500


# ---------- label noise ----------
def test_label_noise_estimate_known_values():
    est = fd.label_noise_estimate([1, 1, 0, 0], [0.99, 0.01, 0.02, 0.6], threshold=0.95)
    assert est["confident_share"] == 0.75 and est["flagged_share"] == 0.25
    assert est["flagged_among_confident"] == pytest.approx(1 / 3) and est["n"] == 4
    assert np.isnan(fd.label_noise_estimate([1, 0], [0.6, 0.4])["flagged_among_confident"])
    with pytest.raises(ValueError, match="threshold"):
        fd.label_noise_estimate([1], [0.9], threshold=0.4)
    with pytest.raises(ValueError, match="rows"):
        fd.label_noise_estimate([1, 0], [0.9])


def test_compare_label_noise_ratio():
    y = np.array([1, 0] * 50)
    p_clean = np.where(y == 1, 0.99, 0.01)
    p_noisy = p_clean.copy()
    p_noisy[:10] = 1 - p_noisy[:10]                  # 10% confidently wrong
    p_real = p_clean.copy()
    p_real[:2] = 1 - p_real[:2]                      # 2% confidently wrong
    out = fd.compare_label_noise(y, p_noisy, y, p_real, thresholds=(0.95,)).iloc[0]
    assert out["syn_flagged_share"] == pytest.approx(0.10) and out["real_flagged_share"] == pytest.approx(0.02)
    assert out["ratio"] == pytest.approx(5.0)
    assert np.isnan(fd.compare_label_noise(y, p_noisy, y, p_clean, thresholds=(0.9,)).iloc[0]["ratio"])


# ---------- ranking agreement ----------
def test_ranking_agreement_known_orderings():
    table = pd.DataFrame({"model": list("abcd"), "S->S": [0.90, 0.91, 0.92, 0.93],
                          "R->R": [0.80, 0.81, 0.82, 0.83], "S->R": [0.73, 0.72, 0.71, 0.70]})
    out = fd.ranking_agreement(table).set_index(["setting_a", "setting_b"])
    assert out.loc[("S->S", "R->R"), "kendall_tau"] == pytest.approx(1.0)
    assert out.loc[("S->S", "S->R"), "spearman_rho"] == pytest.approx(-1.0)
    assert bool(out.loc[("S->S", "R->R"), "same_top"]) and out.loc[("S->S", "S->R"), "top_b"] == "a"
    assert out.loc[("S->S", "R->R"), "models"] == "a;b;c;d"


def test_ranking_agreement_handles_partial_tables_and_aliases():
    wide = pd.DataFrame({"key": list("abcd"), "syn_cv": [0.9, 0.8, 0.7, 0.6],
                         "real_cv": [0.9, 0.8, np.nan, np.nan], "S→R": [0.5, 0.6, 0.7, 0.8]})
    out = fd.ranking_agreement(wide).set_index(["setting_a", "setting_b"])
    assert out.loc[("S->S", "R->R"), "n_models"] == 2 and np.isnan(out.loc[("S->S", "R->R"), "kendall_tau"])
    assert out.loc[("S->S", "S->R"), "kendall_tau"] == pytest.approx(-1.0)
    long = pd.DataFrame({"model": ["x", "x", "y", "y", "z", "z"], "feature_set": ["raw"] * 6,
                         "setting": ["syn_cv", "real_cv"] * 3, "auc": [0.9, 0.8, 0.85, 0.75, 0.8, 0.7]})
    assert fd.normalise_ladder(long).index.tolist() == ["x__raw", "y__raw", "z__raw"]
    assert fd.ranking_agreement(long).iloc[0]["kendall_tau"] == pytest.approx(1.0)
    assert fd.ranking_agreement(pd.DataFrame({"model": ["a"], "S->S": [0.9]})).empty
    indexed = pd.DataFrame({"S->S": [0.9, 0.8, 0.7], "R->R": [0.7, 0.8, 0.9]}, index=list("abc"))
    assert fd.ranking_agreement(indexed).iloc[0]["kendall_tau"] == pytest.approx(-1.0)


def _fake_run(key: str, cv: float, applied: dict) -> CVResult:
    return CVResult(key, np.zeros(4), [cv], cv, {k: np.zeros(4) for k in applied}, applied, 1.0)


def test_ladder_from_store_reads_full_runs_only(tmp_path):
    save_run("synthetic", _fake_run("m1__raw", 0.95, {"real": 0.99}), root=tmp_path)
    save_run("real", _fake_run("m1__raw", 0.996, {"synthetic": 0.95}), root=tmp_path)
    save_run("synthetic", _fake_run("m2__raw", 0.96, {"real": 0.98}), {"sample": 3000}, root=tmp_path)
    table = fd.ladder_from_store(tmp_path).set_index("key")
    assert table.index.tolist() == ["m1__raw"]
    assert table.loc["m1__raw"].to_dict() == {"S->S": 0.95, "R->R": 0.996, "S->R": 0.99, "R->S": 0.95}
    assert fd.ladder_from_store(tmp_path, include_sampled=True).shape[0] == 2
    assert fd.ladder_from_store(tmp_path / "empty").empty


# ---------- utility ----------
def test_equivalent_n_interpolates_on_log_scale():
    assert fu.equivalent_n([1000, 10000], [0.9, 1.0], 0.95) == pytest.approx(np.sqrt(1000 * 10000))
    assert fu.equivalent_n([10000, 1000], [1.0, 0.9], 0.9) == 1000
    assert fu.equivalent_n([1000, 2000, 5000], [0.90, 0.92, 0.93], 0.95) is None


def test_stratified_subsample_keeps_class_share():
    y = np.r_[np.ones(300), np.zeros(700)].astype(int)
    idx = fu.stratified_subsample(y, 100, seed=0)
    assert len(idx) == 100 and y[idx].mean() == pytest.approx(0.3) and np.all(np.diff(idx) > 0)
    assert len(fu.stratified_subsample(y, None, 0)) == 1000 and len(fu.stratified_subsample(y, 5000, 0)) == 1000


def test_utility_curve_on_toy_data():
    real, syn = _frame(1000, 0), _frame(800, 1)
    y_real = real[TARGET].to_numpy()
    result = fu.utility_curve(real[FEATURES], y_real, make_folds(y_real), syn[FEATURES], syn[TARGET],
                              sizes=(200, 400, None), seeds=(0, 1), multi_seed_max_n=300,
                              params=FAST_LGBM, n_boot=50)
    assert result.summary["n"].tolist() == [200, 400, 800] and result.summary["n_seeds"].tolist() == [2, 1, 1]
    assert result.n_test == 200 and result.n_pool == 800 and result.n_synthetic == 800
    assert result.tstr.low <= result.tstr.estimate <= result.tstr.high and result.tstr.estimate > 0.7
    assert result.equivalent_n is None or 200 <= result.equivalent_n <= 800
    assert len(result.curve) == 4 and result.tstr_trees >= 1


def test_real_reference_predictions_and_store_lookup(tmp_path):
    real, syn = _frame(500, 0), _frame(300, 1)
    y_real = real[TARGET].to_numpy()
    res = fu.real_reference_predictions(real[FEATURES], y_real, make_folds(y_real), syn[FEATURES], syn[TARGET],
                                        params=FAST_LGBM)
    assert len(res.oof) == 500 and len(res.applied["synthetic"]) == 300 and res.cv_auc > 0.7
    assert fu.stored_real_predictions(500, 300, root=tmp_path) is None
    stored = CVResult("lgbm__raw", res.oof, res.fold_aucs, res.cv_auc, res.applied, res.applied_auc, 1.0)
    save_run("real", stored, root=tmp_path)
    oof, p_syn, meta = fu.stored_real_predictions(500, 300, root=tmp_path)
    assert np.allclose(oof, res.oof) and np.allclose(p_syn, res.applied["synthetic"]) and meta["train"] == "real"
    assert fu.stored_real_predictions(501, 300, root=tmp_path) is None
    save_run("real", stored, {"sample": 500}, root=tmp_path)
    assert fu.stored_real_predictions(500, 300, root=tmp_path) is None


# ---------- figures (smoke tests) ----------
def test_figures_are_written(tmp_path):
    real = _frame(400)
    syn = _no_delays(real)
    fp.plot_distortion(fd.marginal_distortion(syn, real), tmp_path / "d.png")
    fp.plot_delays(fd.delay_bins(syn, real), tmp_path / "delays.png")
    summary = pd.DataFrame({"n": [1000, 2000, 5000], "auc_mean": [0.9, 0.93, 0.95], "auc_std": [0.01, 0.01, np.nan]})
    fp.plot_utility(summary, (0.94, 0.93, 0.95), 3000.0, 700_000, tmp_path / "u.png")
    fp.plot_utility(summary, (0.97, 0.96, 0.98), None, 700_000, tmp_path / "u2.png")
    fp.plot_target_dependence(fd.compare_target_dependence(real, real, n=300), tmp_path / "t.png")
    assert {p.name for p in tmp_path.iterdir()} == {"d.png", "delays.png", "u.png", "u2.png", "t.png"}


def test_exact_copies_counts_rows_reproduced_on_every_feature():
    real = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    syn = pd.DataFrame({"a": [1, 2, 9, 1], "b": ["x", "q", "z", "x"]})
    out = fd.exact_copies(syn, real, features=["a", "b"])
    assert out == {"n_synthetic": 4, "n_copies": 2, "share": 0.5}
