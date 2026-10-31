"""
Experiment Harness Module.
Orchestrates systematic label noise degradation benchmarks across multiple seeds,
evaluates mitigation strategies, and records comprehensive performance metrics.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss

from src.noise_injector import LabelNoiseInjector
from src.models import get_model
from src.mitigation import (
    OutOfFoldCleaner,
    NeighborhoodCleaner,
    compute_soft_sample_weights,
    evaluate_noise_detection
)


def load_dataset(name: str = "breast_cancer", random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads benchmark dataset for label noise experimentation.
    """
    if name == "breast_cancer":
        data = load_breast_cancer()
        X, y = data.data, data.target
    elif name == "moons":
        X, y = make_moons(n_samples=1200, noise=0.25, random_state=random_state)
    elif name == "synthetic_tabular":
        X, y = make_classification(
            n_samples=2000,
            n_features=20,
            n_informative=12,
            n_redundant=4,
            n_classes=2,
            flip_y=0.0,  # 0 base noise
            class_sep=1.0,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown dataset name: {name}")

    return X, y


def run_noise_degradation_study(
    X: np.ndarray,
    y: np.ndarray,
    noise_rates: List[float] = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50],
    models: List[str] = ["Logistic Regression", "Random Forest", "XGBoost"],
    seeds: List[int] = [42, 101, 2024],
    test_size: float = 0.25
) -> pd.DataFrame:
    """
    Trains each model on increasingly noisy training labels across multiple seeds.
    Evaluation is always conducted against the 100% clean ground truth test split.
    """
    records = []

    for seed in seeds:
        # Split into train and clean test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )

        injector = LabelNoiseInjector(random_state=seed)

        for noise_rate in noise_rates:
            # Corrupt only training labels; test labels remain ground truth
            y_train_noisy, noise_mask = injector.inject_symmetric_noise(y_train, noise_rate)
            empirical_noise = np.mean(noise_mask)

            for model_name in models:
                clf = get_model(model_name, random_state=seed, robust_config=False)
                clf.fit(X_train, y_train_noisy)

                # Predictions on clean test set
                y_pred = clf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="macro")

                # Probabilities for AUC and Log Loss
                try:
                    y_proba = clf.predict_proba(X_test)
                    if len(np.unique(y_test)) == 2:
                        auc = roc_auc_score(y_test, y_proba[:, 1])
                    else:
                        auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
                    loss = log_loss(y_test, y_proba)
                except Exception:
                    auc = np.nan
                    loss = np.nan

                records.append({
                    "seed": seed,
                    "model": model_name,
                    "noise_rate": noise_rate,
                    "empirical_noise": empirical_noise,
                    "accuracy": acc,
                    "macro_f1": f1,
                    "roc_auc": auc,
                    "log_loss": loss
                })

    df_results = pd.DataFrame(records)
    return df_results


def run_mitigation_benchmark(
    X: np.ndarray,
    y: np.ndarray,
    target_noise_rate: float = 0.30,
    models: List[str] = ["Logistic Regression", "Random Forest", "XGBoost"],
    seed: int = 42,
    test_size: float = 0.25
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Evaluates mitigation techniques at a severe noise level (e.g. 30% label noise).
    Compares:
    1. Clean Baseline (0% noise)
    2. Unmitigated Noisy Model
    3. Robust Regularization
    4. Data Cleaning via Out-of-Fold (OOF) Filtering
    5. Soft Noise-Aware Sample Reweighting
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    injector = LabelNoiseInjector(random_state=seed)
    y_train_noisy, true_noise_mask = injector.inject_symmetric_noise(y_train, target_noise_rate)

    results = []
    noise_detection_metrics = {}

    # 1. Clean Baseline & 2. Unmitigated Noisy
    for model_name in models:
        # Clean baseline
        clf_clean = get_model(model_name, random_state=seed, robust_config=False)
        clf_clean.fit(X_train, y_train)
        acc_clean = accuracy_score(y_test, clf_clean.predict(X_test))
        results.append({
            "model": model_name,
            "strategy": "Clean Baseline",
            "accuracy": acc_clean
        })

        # Unmitigated noisy
        clf_noisy = get_model(model_name, random_state=seed, robust_config=False)
        clf_noisy.fit(X_train, y_train_noisy)
        acc_noisy = accuracy_score(y_test, clf_noisy.predict(X_test))
        results.append({
            "model": model_name,
            "strategy": "Unmitigated Noisy",
            "accuracy": acc_noisy
        })

        # 3. Robust Regularization
        clf_robust = get_model(model_name, random_state=seed, robust_config=True)
        clf_robust.fit(X_train, y_train_noisy)
        acc_robust = accuracy_score(y_test, clf_robust.predict(X_test))
        results.append({
            "model": model_name,
            "strategy": "Robust Regularization",
            "accuracy": acc_robust
        })

    # 4. Data Cleaning via Out-of-Fold Cross-Validation
    # Use Random Forest as clean detector due to strong ensembling stability
    rf_cleaner = get_model("random_forest", random_state=seed, robust_config=True)
    oof_cleaner = OutOfFoldCleaner(estimator=rf_cleaner, n_splits=5, confidence_threshold=0.35, random_state=seed)
    X_cleaned, y_cleaned, clean_mask = oof_cleaner.clean_dataset(X_train, y_train_noisy)

    # Noise detection quality
    detected_noisy_mask = ~clean_mask
    noise_detection_metrics["OOF_Cleaner"] = evaluate_noise_detection(true_noise_mask, detected_noisy_mask)

    for model_name in models:
        clf_cleaned = get_model(model_name, random_state=seed, robust_config=False)
        clf_cleaned.fit(X_cleaned, y_cleaned)
        acc_cleaned = accuracy_score(y_test, clf_cleaned.predict(X_test))
        results.append({
            "model": model_name,
            "strategy": "Data Cleaning (OOF Filter)",
            "accuracy": acc_cleaned
        })

    # 5. Noise-Aware Soft Sample Reweighting
    weights = compute_soft_sample_weights(
        estimator=rf_cleaner,
        X=X_train,
        y=y_train_noisy,
        n_splits=5,
        gamma=2.0,
        random_state=seed
    )

    for model_name in models:
        clf_weighted = get_model(model_name, random_state=seed, robust_config=False)
        clf_weighted.fit(X_train, y_train_noisy, sample_weight=weights)
        acc_weighted = accuracy_score(y_test, clf_weighted.predict(X_test))
        results.append({
            "model": model_name,
            "strategy": "Soft Sample Reweighting",
            "accuracy": acc_weighted
        })

    df_mitigation = pd.DataFrame(results)

    # Calculate recovery rate relative to clean vs noisy drop
    # Recovery = (Acc_strat - Acc_noisy) / (Acc_clean - Acc_noisy) * 100
    pivoted = df_mitigation.pivot(index="model", columns="strategy", values="accuracy")
    recovery_dict = {}
    for col in pivoted.columns:
        if col not in ["Clean Baseline", "Unmitigated Noisy"]:
            clean_col = pivoted["Clean Baseline"]
            noisy_col = pivoted["Unmitigated Noisy"]
            drop = clean_col - noisy_col
            gain = pivoted[col] - noisy_col
            # Protect against division by zero
            recovery = np.where(drop > 1e-5, (gain / drop) * 100, 0.0)
            recovery_dict[col] = recovery

    return df_mitigation, noise_detection_metrics
