"""
models/logistic_regression.py
Logistic Regression — linear baseline with GridSearchCV.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from models.base import BaseClassifier


class LogisticRegressionModel(BaseClassifier):
    """
    Regularised Logistic Regression used as a linear baseline.
    Tuned over regularisation strength and solver.
    """

    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        self._name = "Logistic Regression"
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.model = None
        self.best_params_ = {}

    @property
    def name(self) -> str:
        return self._name

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        param_grid = {
            "C":            [0.001, 0.01, 0.1, 1.0, 10.0],
            "penalty":      ["l2"],
            "solver":       ["lbfgs", "saga"],
            "class_weight": ["balanced", None],
            "max_iter":     [500],
        }
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                             random_state=self.random_state)
        search = GridSearchCV(
            LogisticRegression(random_state=self.random_state),
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
