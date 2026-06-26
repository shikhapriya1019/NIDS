"""
preprocessing/feature_selection.py

Provides a wrapper around scikit-learn's SelectKBest (chi2) for
supervised feature selection. Only fitted on training data.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline


def select_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test_dict: dict,
    k: int = 20,
) -> tuple[np.ndarray, dict, list[int]]:
    """
    Select the top *k* features using ANOVA F-statistic.

    Parameters
    ----------
    X_train : np.ndarray
        Scaled training feature matrix.
    y_train : np.ndarray
        Binary training labels (0 = benign, 1 = attack).
    X_test_dict : dict
        Mapping of dataset name → scaled test feature array.
    k : int
        Number of features to retain (default 20).

    Returns
    -------
    X_train_sel : np.ndarray
        Training matrix with selected features.
    X_test_sel_dict : dict
        Test matrices with selected features.
    selected_indices : list[int]
        Indices of the retained features.
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_sel = selector.fit_transform(X_train, y_train)
    selected_indices = list(selector.get_support(indices=True))

    X_test_sel_dict = {
        name: arr[:, selected_indices]
        for name, arr in X_test_dict.items()
    }

    print(f"[feature_selection] Selected {k} features: indices {selected_indices}")
    return X_train_sel, X_test_sel_dict, selected_indices
