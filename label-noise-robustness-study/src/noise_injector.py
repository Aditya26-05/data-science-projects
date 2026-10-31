"""
Label Noise Injection Module.
Provides reproducible utilities for injecting symmetric (uniform) and
asymmetric (class-conditional) label noise into classification datasets.
"""

from typing import Tuple, Optional, Union
import numpy as np


class LabelNoiseInjector:
    """
    Injects synthetic label noise into true class labels.

    Supports:
    - Symmetric (uniform) noise: randomly flips label to any other class with probability eta.
    - Asymmetric (pair-flip) noise: simulates realistic misclassification where class i flips to class j.
    """

    def __init__(self, random_state: Optional[int] = 42):
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def inject_symmetric_noise(
        self,
        y: np.ndarray,
        noise_rate: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Injects uniform symmetric label noise.

        Parameters
        ----------
        y : np.ndarray
            1D array of ground truth class labels (integer-encoded: 0, 1, ..., C-1).
        noise_rate : float
            Fraction of labels to corrupt, between 0.0 and 1.0.

        Returns
        -------
        y_noisy : np.ndarray
            Array of labels containing corrupted entries.
        noise_mask : np.ndarray
            Boolean array where True indicates the label was flipped.
        """
        if not (0.0 <= noise_rate <= 1.0):
            raise ValueError(f"noise_rate must be between 0.0 and 1.0, got {noise_rate}")

        y = np.asarray(y, dtype=int)
        y_noisy = y.copy()
        n_samples = len(y)

        if noise_rate == 0.0:
            return y_noisy, np.zeros(n_samples, dtype=bool)

        classes = np.unique(y)
        n_classes = len(classes)
        if n_classes < 2:
            raise ValueError("At least 2 distinct classes required to inject label noise.")

        # Determine which samples to corrupt
        n_noisy = int(round(noise_rate * n_samples))
        noisy_indices = self.rng.choice(n_samples, size=n_noisy, replace=False)

        for idx in noisy_indices:
            current_label = y[idx]
            # Choose from remaining classes uniformly
            candidate_classes = classes[classes != current_label]
            y_noisy[idx] = self.rng.choice(candidate_classes)

        noise_mask = (y != y_noisy)
        return y_noisy, noise_mask

    def inject_asymmetric_noise(
        self,
        y: np.ndarray,
        noise_rate: float,
        transition_matrix: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Injects class-conditional asymmetric label noise.
        For binary classification, by default flips class 0 -> class 1 with noise_rate,
        or uses a provided Markov transition matrix T[i, j] = P(Y_noisy=j | Y_true=i).

        Parameters
        ----------
        y : np.ndarray
            Ground truth class labels.
        noise_rate : float
            Noise level scaling the off-diagonal transition probabilities.
        transition_matrix : Optional[np.ndarray]
            C x C matrix of transition probabilities.

        Returns
        -------
        y_noisy : np.ndarray
            Noisy labels.
        noise_mask : np.ndarray
            Boolean mask of corrupted labels.
        """
        y = np.asarray(y, dtype=int)
        y_noisy = y.copy()
        classes = np.unique(y)
        n_classes = len(classes)

        if transition_matrix is None:
            # Default pair-flip transition: class i -> (i+1) % C
            T = np.eye(n_classes) * (1.0 - noise_rate)
            for i in range(n_classes):
                next_class = (i + 1) % n_classes
                T[i, next_class] += noise_rate
        else:
            T = np.asarray(transition_matrix)
            if T.shape != (n_classes, n_classes):
                raise ValueError(f"Transition matrix shape {T.shape} must be ({n_classes}, {n_classes})")

        for c in range(n_classes):
            idx_c = np.where(y == c)[0]
            if len(idx_c) == 0:
                continue
            probs = T[c]
            new_labels = self.rng.choice(classes, size=len(idx_c), p=probs)
            y_noisy[idx_c] = new_labels

        noise_mask = (y != y_noisy)
        return y_noisy, noise_mask
