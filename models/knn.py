"""
models/knn.py
K-Nearest Neighbours classifier with GridSearchCV.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from models.base import BaseClassifier


class KNNModel(BaseClassifier):
    """
    K-Nearest Neighbours — distance-based baseline model.
    Tuned over number of neighbours and distance metric.
    """

    def __init__(self, cv_folds: int = 5):
        self._name = "K-Nearest Neighbours"
        self.cv_folds = cv_folds
        self.model = None
        self.best_params_ = {}

    @property
    def name(self) -> str:
        return self._name

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        param_grid = {
            "n_neighbors": [3, 5, 7, 11, 15],
            "weights":     ["uniform", "distance"],
            "metric":      ["euclidean", "manhattan"],
        }
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                             random_state=42)
        search = GridSearchCV(
            KNeighborsClassifier(n_jobs=-1),
            param_grid=param_grid,
            scoring="f1",
            cv=cv,
            verbose=0,
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
