"""
models/xgboost_model.py
XGBoost classifier with Randomized hyperparameter search
and Stratified K-Fold cross-validation.
"""

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from models.base import BaseClassifier


class XGBoostModel(BaseClassifier):
    """
    Gradient-boosted trees (XGBoost) with automated tuning.
    scale_pos_weight compensates for class imbalance automatically.
    """

    def __init__(self, n_iter: int = 10, cv_folds: int = 5,
                 random_state: int = 42):
        self._name = "XGBoost"
        self.n_iter = n_iter
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.model = None
        self.best_params_ = {}

    @property
    def name(self) -> str:
        return self._name

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Compute imbalance ratio for scale_pos_weight
        n_neg = int((y == 0).sum())
        n_pos = int((y == 1).sum())
        scale = n_neg / max(n_pos, 1)

        param_dist = {
            "n_estimators":    [100, 200, 300],
            "max_depth":       [3, 5, 7, 9],
            "learning_rate":   [0.01, 0.05, 0.1, 0.2],
            "subsample":       [0.6, 0.8, 1.0],
            "colsample_bytree":[0.6, 0.8, 1.0],
            "gamma":           [0, 0.1, 0.3],
            "reg_alpha":       [0, 0.1, 1.0],
        }
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                             random_state=self.random_state)
        search = RandomizedSearchCV(
            XGBClassifier(
                scale_pos_weight=scale,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0,
            ),
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
