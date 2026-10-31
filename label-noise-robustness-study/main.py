"""
Main Execution Script for Label Noise Study.
Runs the complete experiment pipeline:
1. Systematic Noise Degradation across [0%, 5%, 10%, 20%, 30%, 40%, 50%] noise
2. Mitigation Benchmark (Data Cleaning & Noise-Aware Reweighting) at 30% noise
3. 2D Decision Boundary Contour Generation
4. Exporting summary metrics and publication-ready figures
"""

import os
import time
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

from src.experiment import (
    load_dataset,
    run_noise_degradation_study,
    run_mitigation_benchmark
)
from src.noise_injector import LabelNoiseInjector
from src.models import get_model
from src.mitigation import OutOfFoldCleaner, compute_soft_sample_weights
from src.visualization import (
    plot_accuracy_degradation,
    plot_mitigation_comparison,
    plot_decision_boundaries_2d
)


def main():
    print("=" * 75)
    print("   EFFECT OF LABEL NOISE ON MODEL ACCURACY & MITIGATION STRATEGIES")
    print("   Models: Logistic Regression, Random Forest, XGBoost")
    print("=" * 75)

    os.makedirs("results/figures", exist_ok=True)
    start_time = time.time()

    # -------------------------------------------------------------
    # 1. NOISE DEGRADATION EXPERIMENT (Breast Cancer Benchmark)
    # -------------------------------------------------------------
    print("\n[Phase 1] Running Systematic Noise Degradation Study (Multi-Seed)...")
    X, y = load_dataset("breast_cancer")
    noise_rates = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    seeds = [42, 101, 2024]

    df_degradation = run_noise_degradation_study(
        X, y,
        noise_rates=noise_rates,
        models=["Logistic Regression", "Random Forest", "XGBoost"],
        seeds=seeds
    )

    df_degradation.to_csv("results/noise_degradation_metrics.csv", index=False)
    print(" -> Saved degradation metrics to 'results/noise_degradation_metrics.csv'")

    # Summary table
    summary_agg = df_degradation.groupby(["model", "noise_rate"])["accuracy"].agg(["mean", "std"]).reset_index()
    print("\n--- Accuracy vs. Noise Level (Mean ± Std over 3 seeds) ---")
    pivoted_summary = summary_agg.pivot(index="noise_rate", columns="model", values="mean")
    print(pivoted_summary.map(lambda v: f"{v*100:.1f}%"))

    # Plot Degradation Curves
    plot_accuracy_degradation(df_degradation, "results/figures/accuracy_vs_noise.png")
    print(" -> Saved figure: results/figures/accuracy_vs_noise.png")

    # -------------------------------------------------------------
    # 2. MITIGATION STRATEGIES BENCHMARK (at 30% Label Noise)
    # -------------------------------------------------------------
    print("\n[Phase 2] Evaluating Robust Mitigation Strategies at 30% Noise...")
    target_noise = 0.30
    df_mitigation, detection_metrics = run_mitigation_benchmark(
        X, y,
        target_noise_rate=target_noise,
        models=["Logistic Regression", "Random Forest", "XGBoost"],
        seed=42
    )

    df_mitigation.to_csv("results/mitigation_results.csv", index=False)
    print(" -> Saved mitigation results to 'results/mitigation_results.csv'")

    print("\n--- Mitigation Strategy Comparison (Clean vs 30% Noisy vs Mitigated) ---")
    pivoted_mitigation = df_mitigation.pivot(index="model", columns="strategy", values="accuracy")
    print((pivoted_mitigation * 100).round(2).astype(str) + "%")

    print("\n--- Noise Detection Quality (OOF Data Cleaning Filter) ---")
    oof_metrics = detection_metrics.get("OOF_Cleaner", {})
    print(f" Precision: {oof_metrics.get('precision', 0):.3f}")
    print(f" Recall:    {oof_metrics.get('recall', 0):.3f}")
    print(f" F1-Score:  {oof_metrics.get('f1', 0):.3f}")

    # Plot Mitigation Comparison
    plot_mitigation_comparison(df_mitigation, "results/figures/mitigation_comparison.png")
    print(" -> Saved figure: results/figures/mitigation_comparison.png")

    # -------------------------------------------------------------
    # 3. 2D DECISION BOUNDARY VISUALIZATION (Moons Dataset)
    # -------------------------------------------------------------
    print("\n[Phase 3] Generating 2D Decision Boundaries (Clean vs Noisy vs Mitigated)...")
    X_2d, y_2d = load_dataset("moons", random_state=42)
    injector_2d = LabelNoiseInjector(random_state=42)
    y_2d_noisy, _ = injector_2d.inject_symmetric_noise(y_2d, noise_rate=target_noise)

    # Train OOF cleaner and sample reweighter on 2D data
    rf_guide = get_model("random_forest", random_state=42, robust_config=True)
    cleaner_2d = OutOfFoldCleaner(rf_guide, n_splits=5, confidence_threshold=0.35, random_state=42)
    X_clean_2d, y_clean_2d, _ = cleaner_2d.clean_dataset(X_2d, y_2d_noisy)
    weights_2d = compute_soft_sample_weights(rf_guide, X_2d, y_2d_noisy, random_state=42)

    # Models dictionary for grid
    models_dict = {
        "Clean Logistic Reg": get_model("logistic_regression", 42).fit(X_2d, y_2d),
        "30% Noisy Logistic Reg": get_model("logistic_regression", 42).fit(X_2d, y_2d_noisy),
        "Mitigated Logistic Reg (Reweighted)": get_model("logistic_regression", 42).fit(X_2d, y_2d_noisy, sample_weight=weights_2d),

        "Clean Random Forest": get_model("random_forest", 42).fit(X_2d, y_2d),
        "30% Noisy Random Forest": get_model("random_forest", 42).fit(X_2d, y_2d_noisy),
        "Mitigated Random Forest (OOF Cleaned)": get_model("random_forest", 42).fit(X_clean_2d, y_clean_2d),

        "Clean XGBoost": get_model("xgboost", 42).fit(X_2d, y_2d),
        "30% Noisy XGBoost": get_model("xgboost", 42).fit(X_2d, y_2d_noisy),
        "Mitigated XGBoost (Robust+Reweighted)": get_model("xgboost", 42, robust_config=True).fit(X_2d, y_2d_noisy, sample_weight=weights_2d),
    }

    plot_decision_boundaries_2d(
        models_dict=models_dict,
        X=X_2d,
        y_true=y_2d,
        y_noisy=y_2d_noisy,
        noise_rate=target_noise,
        save_path="results/figures/decision_boundaries.png"
    )
    print(" -> Saved figure: results/figures/decision_boundaries.png")

    elapsed = time.time() - start_time
    print(f"\n[Done] Pipeline finished successfully in {elapsed:.2f} seconds.")
    print("=" * 75)


if __name__ == "__main__":
    main()
