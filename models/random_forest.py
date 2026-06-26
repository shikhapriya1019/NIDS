"""
models/random_forest.py
Random Forest classifier with Randomized hyperparameter search
and Stratified K-Fold cross-validation.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from models.base import BaseClassifier


class RandomForestModel(BaseClassifier):
    """
    Random Forest with automated hyperparameter tuning via
    RandomizedSearchCV (5-fold Stratified CV).
    """

    def __init__(self, n_iter: int = 10, cv_folds: int = 5,
                 random_state: int = 42):
        self._name = "Random Forest"
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
            "n_estimators":      [100, 200, 300],
            "max_depth":         [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf":  [1, 2, 4],
            "max_features":      ["sqrt", "log2"],
            "class_weight":      ["balanced", None],
        }
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                             random_state=self.random_state)
        search = RandomizedSearchCV(
            RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
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
