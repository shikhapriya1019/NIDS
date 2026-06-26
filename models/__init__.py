"""
models/__init__.py
Supervised classifier wrappers used in the NIDS pipeline.
XGBoost is imported optionally so the pipeline works even if
the package is not installed (falls back gracefully).
"""

from models.random_forest import RandomForestModel
from models.decision_tree import DecisionTreeModel
from models.logistic_regression import LogisticRegressionModel
from models.knn import KNNModel
from models.mlp import MLPModel

try:
    from models.xgboost_model import XGBoostModel
    _XGBOOST_AVAILABLE = True
except ImportError:
    XGBoostModel = None          # type: ignore[assignment,misc]
    _XGBOOST_AVAILABLE = False

__all__ = [
    "RandomForestModel",
    "XGBoostModel",
    "DecisionTreeModel",
    "LogisticRegressionModel",
    "KNNModel",
    "MLPModel",
    "_XGBOOST_AVAILABLE",
]
