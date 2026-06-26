"""
models/base.py
Abstract base class for all supervised classifier wrappers.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseClassifier(ABC):
    """
    Minimal interface every supervised model must implement.

    All wrappers expose:
        fit(X, y)          — train the model
        predict(X)         — return binary class labels
        predict_proba(X)   — return probability of attack class (for ROC/PR)
        name               — human-readable identifier
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the model's display name."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on feature matrix X and binary labels y."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions (0 or 1) for each sample."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability scores for the attack class (class 1)."""
