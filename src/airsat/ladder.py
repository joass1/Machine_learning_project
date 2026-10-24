"""RQ1 reporting: the model ladder in four settings and what each "trick" adds where.

`trick_deltas` answers the RQ1 question directly: a technique that helps on synthetic passengers
(S->S) but not on real ones (S->R) exploits the generator, not passenger behaviour.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from airsat.config import OOF_DIR
from airsat.fidelity import ladder_from_store
from airsat.metrics import paired_bootstrap_delta
from airsat.oof import list_runs, load_run

# (baseline, variant, what the variant adds); keys are <model>__<feature_set>.
TRICKS = (
    ("logreg__onehot", "mlp__onehot", "Hidden layers (course MLP)"),
    ("logreg__onehot", "lgbm__raw", "Boosted trees"),
    ("lgbm__raw", "xgb__raw", "XGBoost instead of LightGBM"),
    ("xgb__raw", "xgb_bins__raw", "1,024 histogram bins"),
    ("xgb_bins__raw", "xgb_ev__exact_value", "Exact-value target/count encoding"),
    ("lgbm__raw", "catboost_cat__all_categorical", "CatBoost, every column categorical"),
    ("lgbm__raw", "realmlp__raw", "RealMLP"),
    ("lgbm__raw", "tabm__raw", "TabM"),
    ("lgbm__raw", "tabpfn__raw", "TabPFN (10k context)"),
)
# setting -> (training direction, prediction array, dataset whose labels score it)
SETTING_SOURCES = {"S->S": ("synthetic", "oof", "synthetic"), "S->R": ("synthetic", "apply__real", "real"),
                   "R->R": ("real", "oof", "real"), "R->S": ("real", "apply__synthetic", "synthetic")}


def ladder_table(root: Path = OOF_DIR, include_sampled: bool = False) -> pd.DataFrame:
    """One row per model run in both directions: AUC in each setting and the transfer gaps."""
    table = ladder_from_store(root, include_sampled).set_index("key").dropna()
    table["real_gap"] = table["R->R"] - table["S->R"]   # what training on synthetic costs on real passengers
    table["syn_gap"] = table["S->S"] - table["R->S"]
    return table.sort_values("R->R", ascending=False)


def predictions(setting: str, key: str, root: Path = OOF_DIR) -> np.ndarray:
    direction, array, _ = SETTING_SOURCES[setting]
    return load_run(direction, key, root)[0][array]


@dataclass(frozen=True)
class TrickDelta:
    base: str
    variant: str
    label: str
    setting: str
    delta: float
    low: float
    high: float


def trick_deltas(targets: dict[str, np.ndarray], settings=("S->S", "S->R"), tricks=TRICKS,
                 root: Path = OOF_DIR, n_boot: int = 200) -> pd.DataFrame:
    """Paired-bootstrap AUC(variant) - AUC(base) per setting, for tricks whose two runs both exist."""
    available = set(list_runs("synthetic", root)) & set(list_runs("real", root))
    rows = []
    for base, variant, label in tricks:
        if not {base, variant} <= available:
            continue
        for setting in settings:
            y = targets[SETTING_SOURCES[setting][2]]
            d = paired_bootstrap_delta(y, predictions(setting, variant, root), predictions(setting, base, root),
                                       n_boot=n_boot)
            rows.append(TrickDelta(base, variant, label, setting, d.estimate, d.low, d.high))
    frame = pd.DataFrame([r.__dict__ for r in rows])
    if frame.empty:
        return frame
    frame["significant"] = (frame["low"] > 0) | (frame["high"] < 0)
    return frame
