"""Course-method neural net: scikit-learn MLP on the same one-hot inputs as the logistic baseline."""
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline

from airsat.config import SEED
from airsat.cv import ModelSpec
from airsat.models.linear import SklearnClassifier, onehot_preprocessor

# Early stopping uses MLPClassifier's own 10% split of the training fold (never the CV validation fold).
MLP_PARAMS = dict(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", alpha=1e-4,
                  batch_size=2048, learning_rate_init=1e-3, max_iter=50, early_stopping=True,
                  validation_fraction=0.1, n_iter_no_change=5, random_state=SEED)


def mlp_onehot() -> ModelSpec:
    return ModelSpec(
        name="mlp", feature_set="onehot",
        make_model=lambda: SklearnClassifier(make_pipeline(onehot_preprocessor(), MLPClassifier(**MLP_PARAMS))),
        params=MLP_PARAMS,
    )
