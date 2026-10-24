"""Feature pipelines (FeaturePipeline protocol in airsat.cv) used by rungs of the model ladder."""
import numpy as np
import pandas as pd

from airsat.config import EXACT_VALUE_COLUMNS, SEED
from airsat.encoders import ExactValueEncoder

EV_SMOOTHING = 20.0   # pseudo-count pulling rare values' target share towards the prior
EV_INNER_FOLDS = 5    # inner folds for the training rows' out-of-fold target encodings
EXACT_VALUE_PARAMS = {"ev_columns": EXACT_VALUE_COLUMNS, "ev_smoothing": EV_SMOOTHING,
                      "ev_inner_folds": EV_INNER_FOLDS}


class ExactValueFeatures:
    """Raw features + fold-safe exact-value target (`te_*`) and count (`cnt_*`) encodings.

    `fit_transform` sees only the training fold: its rows get out-of-fold target encodings from
    inner folds, and `unlabeled` rows (e.g. the competition test set) only add to the counts.
    `transform` encodes validation and apply sets with statistics from the whole training fold.
    """

    def __init__(self, columns: list[str] | None = None, smoothing: float = EV_SMOOTHING,
                 n_inner: int = EV_INNER_FOLDS, seed: int = SEED):
        self.columns = list(columns or EXACT_VALUE_COLUMNS)
        self.encoder = ExactValueEncoder(self.columns, smoothing=smoothing, n_inner=n_inner, seed=seed)

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray,
                      unlabeled: pd.DataFrame | None = None) -> pd.DataFrame:
        return self.encoder.fit_transform_oof(X, pd.Series(np.asarray(y), index=X.index), unlabeled)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.encoder.transform(X)
