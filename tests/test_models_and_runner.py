"""Unit tests for the reference models, the registry and the two-direction runner."""
import numpy as np
import pytest

from airsat import runner
from airsat.config import TARGET
from airsat.data import to_model_frame
from airsat.metrics import auc
from airsat.models import REGISTRY, get_spec
from airsat.oof import list_runs
from tests.conftest import make_toy_frame


@pytest.mark.parametrize("name", ["logreg", "lgbm"])
def test_reference_models_learn_the_toy_signal(name):
    train, valid = make_toy_frame(800, seed=0), make_toy_frame(400, seed=1)
    spec = get_spec(name)
    model = spec.make_model().fit(to_model_frame(train), train[TARGET].to_numpy(),
                                  to_model_frame(valid), valid[TARGET].to_numpy())
    p = model.predict(to_model_frame(valid))
    assert p.min() >= 0 and p.max() <= 1
    assert auc(valid[TARGET], p) > 0.75


def test_lightgbm_fits_without_validation_set():
    train = make_toy_frame(300)
    model = get_spec("lgbm").make_model().fit(to_model_frame(train), train[TARGET].to_numpy())
    assert len(model.predict(to_model_frame(train))) == 300


def test_registry_rejects_unknown_model():
    assert {"logreg", "lgbm"} <= set(REGISTRY)
    with pytest.raises(KeyError, match="unknown model"):
        get_spec("does-not-exist")


def _toy_datasets() -> runner.Datasets:
    syn, test, real = make_toy_frame(500, 0), make_toy_frame(200, 1), make_toy_frame(300, 2)
    return runner.Datasets(to_model_frame(syn), syn[TARGET].to_numpy(),
                           to_model_frame(test.drop(columns=[TARGET])), to_model_frame(real),
                           real[TARGET].to_numpy())


def test_run_direction_scores_the_other_dataset():
    data = _toy_datasets()
    syn = runner.run_direction(get_spec("logreg"), data, "synthetic", sample=500)
    assert set(syn.applied) == {"real", "syn_test"} and set(syn.applied_auc) == {"real"}
    real = runner.run_direction(get_spec("logreg"), data, "real", sample=300)
    assert set(real.applied_auc) == {"synthetic"} and len(real.oof) == 300
    with pytest.raises(ValueError, match="direction"):
        runner.run_direction(get_spec("logreg"), data, "sideways")


def test_run_and_save_writes_both_directions(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "OOF_DIR", tmp_path)
    results = runner.run_and_save(get_spec("logreg"), _toy_datasets(), sample=500)
    assert set(results) == {"synthetic", "real"}
    assert list_runs("synthetic", root=tmp_path / "dev") == ["logreg__onehot"]
    assert list_runs("real", root=tmp_path / "dev") == ["logreg__onehot"]


def test_subsample_and_folds_for():
    df = make_toy_frame(100)
    assert len(runner._subsample(df, 40)) == 40 and runner._subsample(df, None) is df
    y = df[TARGET].to_numpy()
    assert len(np.unique(runner.folds_for("synthetic", y, sample=100))) == 5
