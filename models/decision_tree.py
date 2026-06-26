"""
models/decision_tree.py
Decision Tree with GridSearchCV and Stratified K-Fold CV.
Serves as an interpretable baseline model.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from models.base import BaseClassifier


class DecisionTreeModel(BaseClassifier):
    """
    CART Decision Tree with full grid search over depth and
    split criteria. Used as an interpretable baseline.
    """

    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        self._name = "Decision Tree"
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.model = None
        self.best_params_ = {}

    @property
    def name(self) -> str:
        return self._name

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        param_grid = {
            "max_depth":         [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf":  [1, 2, 4],
            "criterion":         ["gini", "entropy"],
            "class_weight":      ["balanced", None],
        }
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                             random_state=self.random_state)
        search = GridSearchCV(
            DecisionTreeClassifier(random_state=self.random_state),
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
