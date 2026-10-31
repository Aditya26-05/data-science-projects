"""
Model Definitions and Factory.
Wraps Logistic Regression, Random Forest, and XGBoost with consistent interfaces
supporting sample weighting, hyperparameter overrides, and calibrated probability outputs.
"""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost as xgb


class ScaledLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Pipeline wrapper for Logistic Regression with standard feature scaling.
    Exposes sample_weight forwarding during fit.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: Optional[int] = 42):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver="lbfgs"
        )

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y, sample_weight=sample_weight)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


def get_model(
    model_name: str,
    random_state: int = 42,
    robust_config: bool = False,
    extra_params: Optional[Dict[str, Any]] = None
) -> BaseEstimator:
    """
    Factory function returning an initialized classifier instance.

    Parameters
    ----------
    model_name : str
        One of 'logistic_regression', 'random_forest', 'xgboost'.
    random_state : int
        Seed for reproducibility.
    robust_config : bool
        Whether to instantiate with noise-regularized hyperparameters.
    extra_params : Optional[Dict[str, Any]]
        Arbitrary extra parameters to override defaults.

    Returns
    -------
    BaseEstimator
        Configured scikit-learn compatible classifier.
    """
    extra_params = extra_params or {}
    name = model_name.lower().replace(" ", "_")

    if name in ["logistic_regression", "lr", "logreg"]:
        # Standard: C=1.0. Robust: stronger regularization (smaller C, e.g. 0.2) to prevent boundary tilt
        params = {
            "C": 0.2 if robust_config else 1.0,
            "max_iter": 1000,
            "random_state": random_state
        }
        params.update(extra_params)
        return ScaledLogisticRegression(**params)

    elif name in ["random_forest", "rf"]:
        # Standard: unrestricted depth (fits noise in leaf nodes).
        # Robust: restricted max_depth, higher min_samples_leaf, sub-sampling
        if robust_config:
            params = {
                "n_estimators": 150,
                "max_depth": 5,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
                "max_features": "sqrt",
                "random_state": random_state,
                "n_jobs": -1
            }
        else:
            params = {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": random_state,
                "n_jobs": -1
            }
        params.update(extra_params)
        return RandomForestClassifier(**params)

    elif name in ["xgboost", "xgb"]:
        # Standard: default boosting with standard depth.
        # Robust: conservative learning rate, column/row subsampling, L2 lambda, minimum child weight
        if robust_config:
            params = {
                "n_estimators": 100,
                "max_depth": 3,
                "learning_rate": 0.05,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "min_child_weight": 3,
                "reg_lambda": 3.0,
                "eval_metric": "logloss",
                "random_state": random_state,
                "n_jobs": -1
            }
        else:
            params = {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "eval_metric": "logloss",
                "random_state": random_state,
                "n_jobs": -1
            }
        params.update(extra_params)
        return xgb.XGBClassifier(**params)

    else:
        raise ValueError(f"Unknown model_name: {model_name}. Choose from 'logistic_regression', 'random_forest', 'xgboost'.")
