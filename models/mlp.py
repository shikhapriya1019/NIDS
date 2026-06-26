"""
models/mlp.py
Multi-Layer Perceptron classifier (scikit-learn) with
RandomizedSearchCV and Stratified K-Fold cross-validation.
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from models.base import BaseClassifier


class MLPModel(BaseClassifier):
    """
    Feed-forward neural network (MLP) for binary intrusion detection.
    Uses scikit-learn's MLPClassifier for simplicity and compatibility
    with the sklearn cross-validation API.
    """

    def __init__(self, n_iter: int = 10, cv_folds: int = 5,
                 random_state: int = 42):
        self._name = "MLP Neural Network"
        self.n_iter = n_iter
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.model = None
        self.best_params_ = {}

    @property
    def name(self) -> str:
        return self._name

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        param_dist = {
            "hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64), (128, 64, 32)],
            "activation":         ["relu", "tanh"],
            "alpha":              [1e-4, 1e-3, 1e-2],
            "learning_rate_init": [1e-3, 5e-4, 1e-4],
            "max_iter":           [200],
            "early_stopping":     [True],
            "validation_fraction":[0.1],
        }
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                             random_state=self.random_state)
        search = RandomizedSearchCV(
            MLPClassifier(random_state=self.random_state),
            param_distributions=param_dist,
            n_iter=self.n_iter,
            scoring="f1",
            cv=cv,
            verbose=1,
            random_state=self.random_state,
            n_jobs=-1,
        )
        search.fit(X, y)
        self.model = search.best_estimator_
        self.best_params_ = search.best_params_
        print(f"[{self.name}] Best params: {self.best_params_}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
