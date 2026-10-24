"""Tests for the RQ1 ladder report: table from the store, paired trick deltas, figures."""
import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from airsat.cv import CVResult  # noqa: E402
from airsat.ladder import ladder_table, predictions, trick_deltas  # noqa: E402
from airsat.ladder_plots import model_name, plot_ladder, plot_tricks  # noqa: E402
from airsat.metrics import auc  # noqa: E402
from airsat.oof import save_run  # noqa: E402

N_SYN, N_REAL = 400, 300


def _scores(y: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    return 1 / (1 + np.exp(-(2 * y - 1 + rng.normal(0, noise, len(y)))))


def _save_model(root, key: str, y_syn, y_real, syn_noise: float, real_noise: float, seed: int) -> None:
    """Store a fake model whose predictions are better on real rows when real_noise is small."""
    rng = np.random.default_rng(seed)
    oof_s, on_real = _scores(y_syn, syn_noise, rng), _scores(y_real, real_noise, rng)
    oof_r, on_syn = _scores(y_real, real_noise, rng), _scores(y_syn, syn_noise, rng)
    save_run("synthetic", CVResult(key, oof_s, [auc(y_syn, oof_s)], auc(y_syn, oof_s),
                                   {"real": on_real}, {"real": auc(y_real, on_real)}, 1.0), root=root)
    save_run("real", CVResult(key, oof_r, [auc(y_real, oof_r)], auc(y_real, oof_r),
                              {"synthetic": on_syn}, {"synthetic": auc(y_syn, on_syn)}, 1.0), root=root)


@pytest.fixture
def store(tmp_path):
    rng = np.random.default_rng(0)
    y = {"synthetic": rng.integers(0, 2, N_SYN), "real": rng.integers(0, 2, N_REAL)}
    _save_model(tmp_path, "xgb__raw", y["synthetic"], y["real"], 1.5, 1.5, seed=1)
    # the "trick" improves synthetic rows only: same real-row noise, much less synthetic noise
    _save_model(tmp_path, "xgb_bins__raw", y["synthetic"], y["real"], 0.3, 1.5, seed=2)
    return tmp_path, y


def test_ladder_table_has_settings_and_gaps(store):
    root, _ = store
    table = ladder_table(root)
    assert list(table.index) and set(table.index) == {"xgb__raw", "xgb_bins__raw"}
    row = table.loc["xgb__raw"]
    assert row["real_gap"] == pytest.approx(row["R->R"] - row["S->R"])
    assert row["syn_gap"] == pytest.approx(row["S->S"] - row["R->S"])
    assert table["R->R"].is_monotonic_decreasing


def test_predictions_pick_the_right_array(store):
    root, y = store
    assert len(predictions("S->S", "xgb__raw", root)) == N_SYN
    assert len(predictions("S->R", "xgb__raw", root)) == N_REAL
    assert len(predictions("R->S", "xgb__raw", root)) == N_SYN


def test_trick_deltas_flag_a_synthetic_only_gain(store):
    root, y = store
    deltas = trick_deltas(y, root=root, n_boot=100).set_index("setting")
    assert set(deltas["variant"]) == {"xgb_bins__raw"}  # pairs with a missing run are skipped
    assert deltas.loc["S->S", "delta"] > 0.05 and deltas.loc["S->S", "significant"]
    assert deltas.loc["S->S", "low"] <= deltas.loc["S->S", "delta"] <= deltas.loc["S->S", "high"]


def test_trick_deltas_empty_when_no_pair_exists(tmp_path):
    assert trick_deltas({"synthetic": np.zeros(2), "real": np.zeros(2)}, root=tmp_path).empty


def test_figures_are_written(store, tmp_path):
    root, y = store
    plot_ladder(ladder_table(root), tmp_path / "ladder.png")
    plot_tricks(trick_deltas(y, root=root, n_boot=50), tmp_path / "tricks.png")
    assert (tmp_path / "ladder.png").stat().st_size > 0 and (tmp_path / "tricks.png").stat().st_size > 0
    assert model_name("lgbm__raw") == "LightGBM" and model_name("new__x") == "new__x"
