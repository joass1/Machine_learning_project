"""Model registry: name -> factory returning a ModelSpec. Add new rungs of the ladder here.

Optional deep-learning rungs (realmlp, tabm: pytabkit; tabpfn: tabpfn) import their libraries
only when a model is fitted, so listing or building specs never needs them installed.
"""
from typing import Callable

from airsat.cv import ModelSpec
from airsat.models.catboost_cat import catboost_all_categorical
from airsat.models.gbdt import lightgbm_raw, xgboost_bins, xgboost_exact_value, xgboost_raw
from airsat.models.linear import logistic_onehot
from airsat.models.mlp import mlp_onehot
from airsat.models.tabpfn_model import tabpfn_raw
from airsat.models.tabular_nn import realmlp_raw, tabm_raw

# Ladder order (PLAN.md section 4): course methods, GBDTs, artefact-exploiting tricks, NNs, foundation model.
REGISTRY: dict[str, Callable[[], ModelSpec]] = {
    "logreg": logistic_onehot,
    "mlp": mlp_onehot,
    "lgbm": lightgbm_raw,
    "xgb": xgboost_raw,
    "xgb_bins": xgboost_bins,
    "xgb_ev": xgboost_exact_value,
    "catboost_cat": catboost_all_categorical,
    "realmlp": realmlp_raw,
    "tabm": tabm_raw,
    "tabpfn": tabpfn_raw,
}


def get_spec(name: str) -> ModelSpec:
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name]()
