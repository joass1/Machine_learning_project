"""Unit tests for the WP0 infrastructure: data, folds, encoders, metrics, CV runner, store."""
import json

import numpy as np
import pandas as pd
import pytest

from airsat import data as data_mod
from airsat.config import FEATURES, TARGET
from airsat.cv import ApplySet, IdentityFeatures, ModelSpec, run_cv
from airsat.encoders import ExactValueEncoder
from airsat.folds import get_folds, iter_folds, make_folds
from airsat.metrics import auc, bootstrap_auc, fold_aucs, paired_bootstrap_delta
from airsat.oof import list_runs, load_run, save_run
from tests.conftest import make_toy_frame


# ---------- data ----------
def test_to_model_frame_uses_fixed_category_codes(toy):
    frame = data_mod.to_model_frame(toy)
    assert list(frame.columns) == FEATURES
    assert list(frame["Class"].cat.categories) == ["Business", "Eco", "Eco Plus"]
    one_level = data_mod.to_model_frame(toy[toy["Class"] == "Eco"])
    assert list(one_level["Class"].cat.categories) == ["Business", "Eco", "Eco Plus"]


def test_to_model_frame_rejects_unknown_levels(toy):
    with pytest.raises(ValueError, match="Class"):
        data_mod.to_model_frame(toy.assign(Class="First"))


def test_load_real_validates_and_casts_target(tmp_path, toy):
    path = tmp_path / "real.csv"
    toy.assign(**{TARGET: toy[TARGET].astype(bool)}).to_csv(path, index=False)
    real = data_mod.load_real(path)
    assert real[TARGET].dtype.kind == "i" and list(real.columns) == FEATURES + [TARGET]
    toy.drop(columns=["Age"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Age"):
        data_mod.load_real(path)


def test_load_synthetic_reads_train_and_test(tmp_path, toy):
    toy.assign(**{TARGET: toy[TARGET].astype(bool)}).to_csv(tmp_path / "train.csv", index=False)
    toy.drop(columns=[TARGET]).to_csv(tmp_path / "test.csv", index=False)
    train, test = data_mod.load_synthetic(tmp_path)
    assert set(train[TARGET].unique()) <= {0, 1} and TARGET not in test


# ---------- folds ----------
def test_make_folds_is_deterministic_and_stratified():
    y = np.r_[np.ones(300), np.zeros(700)].astype(int)
    a, b = make_folds(y), make_folds(y)
    assert np.array_equal(a, b) and set(np.unique(a)) == set(range(5))
    for fold in range(5):
        assert y[a == fold].mean() == pytest.approx(0.3, abs=0.01)


def test_get_folds_saves_reuses_and_checks_length(tmp_path):
    y = np.tile([0, 1], 50)
    first = get_folds("demo", y, tmp_path)
    assert (tmp_path / "demo.npy").exists() and np.array_equal(first, get_folds("demo", y, tmp_path))
    with pytest.raises(ValueError, match="rows"):
        get_folds("demo", np.tile([0, 1], 10), tmp_path)


def test_iter_folds_partitions_rows():
    fold_ids = make_folds(np.tile([0, 1], 50))
    seen = np.concatenate([va for _, _, va in iter_folds(fold_ids)])
    assert sorted(seen) == list(range(100))


# ---------- encoders ----------
def test_encoder_transform_maps_values_and_falls_back_to_prior():
    X = pd.DataFrame({"d": [1, 1, 2, 2, 2]})
    y = pd.Series([1, 1, 0, 0, 1])
    enc = ExactValueEncoder(["d"], smoothing=0.0).fit(X, y)
    out = enc.transform(pd.DataFrame({"d": [1, 2, 99]}))
    assert out["te_d"].tolist() == pytest.approx([1.0, 1 / 3, 0.6])
    assert out["cnt_d"].tolist() == [2, 3, 0]


def test_encoder_counts_include_unlabeled_rows():
    X = pd.DataFrame({"d": [1, 2]})
    enc = ExactValueEncoder(["d"]).fit(X, [0, 1], unlabeled=pd.DataFrame({"d": [1, 1, 1]}))
    assert enc.transform(X)["cnt_d"].tolist() == [4, 1]


def test_oof_encoding_never_sees_own_label():
    X = pd.DataFrame({"d": np.arange(200)})          # every value unique
    y = pd.Series(np.tile([0, 1], 100))
    out = ExactValueEncoder(["d"], smoothing=0.0).fit_transform_oof(X, y)
    # a unique value is unseen in every inner-training fold, so it can only get the inner prior
    assert out["te_d"].between(0.4, 0.6).all()
    assert out["te_d"].nunique() <= 5


def test_encoder_handles_missing_values():
    X = pd.DataFrame({"d": [np.nan, np.nan, 3.0]})
    out = ExactValueEncoder(["d"], smoothing=0.0).fit(X, [1, 1, 0]).transform(X)
    assert out["te_d"].tolist() == pytest.approx([1.0, 1.0, 0.0])


# ---------- metrics ----------
def test_auc_and_fold_aucs():
    y, p = np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])
    assert auc(y, p) == 1.0
    assert fold_aucs(np.tile(y, 2), np.tile(p, 2), np.repeat([0, 1], 4)) == [1.0, 1.0]


def test_bootstrap_intervals_bracket_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = y * 0.5 + rng.random(400)
    ci = bootstrap_auc(y, p, n_boot=200)
    assert ci.low <= ci.estimate <= ci.high
    same = paired_bootstrap_delta(y, p, p, n_boot=50)
    assert same.estimate == same.low == same.high == 0.0


# ---------- CV runner + store ----------
class ColumnModel:
    """Toy estimator: predicts a scaled copy of one column; records fit calls."""

    fits = 0

    def fit(self, X, y, X_valid=None, y_valid=None):
        ColumnModel.fits += 1
        return self

    def predict(self, X):
        return X["Online boarding"].to_numpy() / 5.0


def _toy_spec():
    return ModelSpec(name="col", feature_set="raw", make_model=ColumnModel, make_features=IdentityFeatures)


def test_run_cv_produces_oof_and_averaged_apply_predictions(toy):
    X, y = data_mod.to_model_frame(toy), toy[TARGET].to_numpy()
    other = make_toy_frame(100, seed=1)
    ColumnModel.fits = 0
    result = run_cv(_toy_spec(), X, y, make_folds(y),
                    {"other": ApplySet(data_mod.to_model_frame(other), other[TARGET].to_numpy()),
                     "unlabeled": ApplySet(data_mod.to_model_frame(other))})
    assert ColumnModel.fits == 5 and result.key == "col__raw"
    assert np.allclose(result.oof, X["Online boarding"] / 5.0)
    assert np.allclose(result.applied["other"], other["Online boarding"] / 5.0)
    assert set(result.applied_auc) == {"other"} and 0.5 < result.cv_auc <= 1.0


def test_store_roundtrip(tmp_path, toy):
    X, y = data_mod.to_model_frame(toy), toy[TARGET].to_numpy()
    result = run_cv(_toy_spec(), X, y, make_folds(y), {"other": ApplySet(X, y)})
    save_run("synthetic", result, {"note": "toy"}, root=tmp_path)
    arrays, meta = load_run("synthetic", "col__raw", root=tmp_path)
    assert np.allclose(arrays["oof"], result.oof) and "apply__other" in arrays
    assert meta["note"] == "toy" and meta["cv_auc"] == pytest.approx(result.cv_auc)
    assert list_runs("synthetic", root=tmp_path) == ["col__raw"] and list_runs("real", root=tmp_path) == []
    assert json.loads((tmp_path / "synthetic" / "col__raw.json").read_text())["train"] == "synthetic"
