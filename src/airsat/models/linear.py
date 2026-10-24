"""Linear baseline: logistic regression with one-hot ratings and categoricals (course method)."""
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from airsat.config import CATEGORICAL, NUMERIC, SERVICE_RATINGS
from airsat.cv import ModelSpec


def onehot_preprocessor():
    """Scaled numerics + one-hot ratings (0 = 'not applicable' becomes its own category)."""
    return make_column_transformer(
        (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
        (OneHotEncoder(handle_unknown="ignore"), SERVICE_RATINGS + CATEGORICAL),
    )


class SklearnClassifier:
    """Adapter: any sklearn classifier pipeline -> Estimator protocol."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_valid=None, y_valid=None) -> "SklearnClassifier":
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]


def logistic_onehot() -> ModelSpec:
    params = {"C": 1.0, "max_iter": 2000}
    return ModelSpec(
        name="logreg", feature_set="onehot",
        make_model=lambda: SklearnClassifier(make_pipeline(onehot_preprocessor(), LogisticRegression(**params))),
        params=params,
    )
