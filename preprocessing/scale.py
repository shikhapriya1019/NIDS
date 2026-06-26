"""
preprocessing/scale.py
MinMax scaling fitted on the full CICIDS2017 training set
(benign + attack) for supervised learning.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def fit_scaler(X_train: pd.DataFrame) -> MinMaxScaler:
    """
    Fit a MinMaxScaler on the full training feature matrix.

    The scaler is fitted exclusively on training data to prevent
    data leakage into test or cross-dataset evaluation sets.

    Parameters
    ----------
    X_train : pd.DataFrame
        Full training feature matrix (benign + attack rows).

    Returns
    -------
    MinMaxScaler
        Fitted scaler instance.
    """
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    return scaler


def transform(df: pd.DataFrame, scaler: MinMaxScaler) -> np.ndarray:
    """
    Apply a pre-fitted scaler to a feature DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame to transform.
    scaler : MinMaxScaler
        Scaler fitted on training data.

    Returns
    -------
    np.ndarray
        Scaled feature array of shape (n_samples, n_features).
    """
    return scaler.transform(df)
