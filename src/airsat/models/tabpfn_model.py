"""TabPFN (Hollmann et al., 2025) as a rung: in-context learning on a stratified subsample.

TabPFN does no gradient training: `fit` stores a context set (at most CONTEXT_ROWS training rows)
and `predict` runs the pretrained transformer over the context plus a batch of query rows.
Uses the TabPFN v2 weights: v2.5 and later are licence-gated behind a Prior Labs login.
tabpfn is imported lazily so the rest of the ladder works without it installed.
"""
import numpy as np
import pandas as pd

from airsat.config import CATEGORICAL, SEED
from airsat.cv import ModelSpec
from airsat.models.inputs import category_codes, stratified_subsample

TABPFN_PARAMS = dict(model_version="v2", context_rows=10_000, n_estimators=4,
                     predict_batch_rows=20_000, ignore_pretraining_limits=True)


class TabPFNModel:
    def __init__(self, model_version: str = "v2", context_rows: int = 10_000, n_estimators: int = 4,
                 predict_batch_rows: int = 20_000, ignore_pretraining_limits: bool = True,
                 seed: int = SEED, device: str | None = None):
        self.model_version, self.context_rows, self.n_estimators = model_version, context_rows, n_estimators
        self.predict_batch_rows, self.ignore_limits = predict_batch_rows, ignore_pretraining_limits
        self.seed, self.device = seed, device
        self.model = None

    def _make(self, columns: list[str]):
        import torch
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        cat_idx = [i for i, c in enumerate(columns) if c in CATEGORICAL]
        return TabPFNClassifier.create_default_for_version(
            ModelVersion(self.model_version), device=device, n_estimators=self.n_estimators,
            categorical_features_indices=cat_idx, ignore_pretraining_limits=self.ignore_limits,
            random_state=self.seed)

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_valid=None, y_valid=None) -> "TabPFNModel":
        """Validation data is unused: TabPFN has no training loop to early-stop."""
        X_ctx, y_ctx = stratified_subsample(X, y, self.context_rows, self.seed)
        self.model = self._make(list(X.columns))
        self.model.fit(category_codes(X_ctx).to_numpy(), np.asarray(y_ctx))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        values = category_codes(X).to_numpy()
        batches = range(0, len(values), self.predict_batch_rows)
        parts = [self.model.predict_proba(values[i:i + self.predict_batch_rows])[:, 1] for i in batches]
        return np.concatenate(parts) if parts else np.zeros(0)


def tabpfn_raw() -> ModelSpec:
    return ModelSpec(name="tabpfn", feature_set="raw", make_model=lambda: TabPFNModel(**TABPFN_PARAMS),
                     params=TABPFN_PARAMS)
