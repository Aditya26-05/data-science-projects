"""
Mitigation Strategies for Label Noise.
Implements:
1. Data Cleaning / Pruning via Out-of-Fold Cross-Validated Confident Filtering
2. Neighborhood Consistency Filtering (ENN - Edited Nearest Neighbor)
3. Noise-Aware Soft Sample Reweighting
4. Noise Detection Performance Metrics (Precision, Recall, F1 against true noise mask)
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.base import clone, BaseEstimator


class OutOfFoldCleaner:
    """
    Cleans label noise using Out-of-Fold (OOF) cross-validated probability estimates.
    Identifies samples where observed label disagrees with consensus cross-validated predictions.
    """

    def __init__(self, estimator: BaseEstimator, n_splits: int = 5, confidence_threshold: float = 0.35, random_state: int = 42):
        self.estimator = estimator
        self.n_splits = n_splits
        self.confidence_threshold = confidence_threshold
        self.random_state = random_state

    def detect_noise(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes OOF class probabilities and identifies noisy label indices.

        Returns
        -------
        is_clean : np.ndarray
            Boolean mask where True indicates sample is retained as clean.
        oof_confidences : np.ndarray
            Predicted probability assigned to the observed label: P(Y = y_obs | x).
        """
        n_samples = len(y)
        oof_confidences = np.zeros(n_samples, dtype=float)
        oof_preds = np.zeros(n_samples, dtype=int)

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for train_idx, val_idx in skf.split(X, y):
            clf = clone(self.estimator)
            clf.fit(X[train_idx], y[train_idx])

            # Get predicted probabilities on validation fold
            probs = clf.predict_proba(X[val_idx])
            pred_classes = clf.classes_[np.argmax(probs, axis=1)]
            oof_preds[val_idx] = pred_classes

            # Probability of observed label
            for i, idx in enumerate(val_idx):
                observed_c = y[idx]
                class_col = np.where(clf.classes_ == observed_c)[0]
                if len(class_col) > 0:
                    oof_confidences[idx] = probs[i, class_col[0]]
                else:
                    oof_confidences[idx] = 0.0

        # Sample is deemed noisy if:
        # 1. Observed label probability is below threshold, AND
        # 2. Predicted class differs from observed label
        is_noisy = (oof_confidences < self.confidence_threshold) & (oof_preds != y)
        is_clean = ~is_noisy

        return is_clean, oof_confidences

    def clean_dataset(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filters out suspected noisy samples from the dataset.

        Returns
        -------
        X_clean, y_clean, is_clean_mask
        """
        is_clean, _ = self.detect_noise(X, y)
        return X[is_clean], y[is_clean], is_clean


class NeighborhoodCleaner:
    """
    Cleans label noise using feature-space neighborhood consistency (ENN rule).
    If a sample's label disagrees with the majority of its k nearest neighbors, it is pruned.
    """

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors

    def detect_noise(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Returns boolean mask where True means the sample is clean.
        """
        from sklearn.preprocessing import StandardScaler
        X_norm = StandardScaler().fit_transform(X)

        nn = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
        nn.fit(X_norm)
        indices = nn.kneighbors(X_norm, return_distance=False)

        # Exclude self (index 0)
        neighbor_indices = indices[:, 1:]
        neighbor_labels = y[neighbor_indices]

        # Majority vote among neighbors
        clean_mask = np.zeros(len(y), dtype=bool)
        for i in range(len(y)):
            vals, counts = np.unique(neighbor_labels[i], return_counts=True)
            majority_label = vals[np.argmax(counts)]
            clean_mask[i] = (majority_label == y[i])

        return clean_mask


def compute_soft_sample_weights(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    gamma: float = 2.0,
    min_weight: float = 0.05,
    random_state: int = 42
) -> np.ndarray:
    """
    Computes smooth, continuous sample weights based on cross-validated consensus.
    Samples whose observed label has low model confidence receive small weights:
        w_i = max(min_weight, (P(Y = y_obs | x_i)) ** gamma)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    probs_observed = np.zeros(len(y), dtype=float)

    for train_idx, val_idx in skf.split(X, y):
        clf = clone(estimator)
        clf.fit(X[train_idx], y[train_idx])
        probs = clf.predict_proba(X[val_idx])

        for i, idx in enumerate(val_idx):
            observed_c = y[idx]
            class_col = np.where(clf.classes_ == observed_c)[0]
            if len(class_col) > 0:
                probs_observed[idx] = probs[i, class_col[0]]
            else:
                probs_observed[idx] = 0.0

    # Soft power weighting
    weights = np.maximum(min_weight, probs_observed ** gamma)
    # Normalize weights so mean is 1.0 (prevents learning rate distortion)
    weights = weights / np.mean(weights)
    return weights


def evaluate_noise_detection(
    true_noise_mask: np.ndarray,
    detected_noise_mask: np.ndarray
) -> Dict[str, float]:
    """
    Evaluates how accurately a data cleaning method identified injected label noise.
    """
    prec = precision_score(true_noise_mask, detected_noise_mask, zero_division=0)
    rec = recall_score(true_noise_mask, detected_noise_mask, zero_division=0)
    f1 = f1_score(true_noise_mask, detected_noise_mask, zero_division=0)

    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1)
    }
