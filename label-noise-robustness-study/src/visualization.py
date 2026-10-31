"""
Visualization Module for Label Noise Study.
Generates publication-quality charts:
1. Accuracy Degradation Curves across noise levels
2. Mitigation Strategy Comparison (Noisy vs Cleaned vs Reweighted)
3. 2D Decision Boundary Contour Plots showcasing noise overfitting and recovery
"""

import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator


# Styling
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "figure.dpi": 300
})


def plot_accuracy_degradation(
    df_results: pd.DataFrame,
    save_path: str = "results/figures/accuracy_vs_noise.png"
):
    """
    Plots model test accuracy as a function of training label noise level.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(9, 6))

    palette = {
        "Logistic Regression": "#2b5c8f",
        "Random Forest": "#27ae60",
        "XGBoost": "#e74c3c"
    }
    markers = {
        "Logistic Regression": "o",
        "Random Forest": "s",
        "XGBoost": "^"
    }

    # Group by model and noise_rate if multi-seed
    summary = df_results.groupby(["model", "noise_rate"])["accuracy"].agg(["mean", "std"]).reset_index()

    for model_name, grp in summary.groupby("model"):
        color = palette.get(model_name, "#333333")
        marker = markers.get(model_name, "o")
        plt.plot(
            grp["noise_rate"] * 100,
            grp["mean"],
            label=model_name,
            color=color,
            marker=marker,
            linewidth=2.5,
            markersize=8
        )
        if grp["std"].notna().all() and (grp["std"] > 0).any():
            plt.fill_between(
                grp["noise_rate"] * 100,
                grp["mean"] - grp["std"],
                grp["mean"] + grp["std"],
                color=color,
                alpha=0.15
            )

    plt.title("Effect of Label Noise on Test Accuracy", weight="bold", pad=12)
    plt.xlabel("Label Noise Level (%)")
    plt.ylabel("Test Accuracy (Evaluated on Clean Ground Truth)")
    plt.ylim(0.45, 1.02)
    plt.xticks(sorted(df_results["noise_rate"].unique() * 100))
    plt.legend(title="Classifier", loc="lower left", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_mitigation_comparison(
    df_mitigation: pd.DataFrame,
    save_path: str = "results/figures/mitigation_comparison.png"
):
    """
    Plots grouped bar chart comparing performance across mitigation strategies.
    Strategies:
    - Clean Baseline
    - Unmitigated Noisy (e.g. 30% Noise)
    - Robust Regularization
    - Data Cleaning (OOF Filter)
    - Noise-Aware Sample Reweighting
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))

    order_strategies = [
        "Clean Baseline",
        "Unmitigated Noisy",
        "Robust Regularization",
        "Data Cleaning (OOF Filter)",
        "Soft Sample Reweighting"
    ]
    # Filter only available strategies
    avail_strategies = [s for s in order_strategies if s in df_mitigation["strategy"].unique()]

    colors = ["#7f8c8d", "#e74c3c", "#f39c12", "#3498db", "#2ecc71"]
    palette = dict(zip(order_strategies, colors))

    ax = sns.barplot(
        data=df_mitigation,
        x="model",
        y="accuracy",
        hue="strategy",
        hue_order=avail_strategies,
        palette=palette,
        edgecolor="black",
        linewidth=0.8,
        errorbar=None
    )

    plt.title("Recovery of Model Robustness via Mitigation Strategies (at 30% Noise)", weight="bold", pad=14)
    plt.xlabel("Model Architecture")
    plt.ylabel("Test Accuracy on Clean Ground Truth")
    plt.ylim(0.50, 1.05)
    plt.legend(title="Strategy", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    # Annotate bar values
    for p in ax.patches:
        height = p.get_height()
        if height > 0.0:
            ax.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 2),
                textcoords="offset points"
            )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_decision_boundaries_2d(
    models_dict: Dict[str, BaseEstimator],
    X: np.ndarray,
    y_true: np.ndarray,
    y_noisy: np.ndarray,
    noise_rate: float,
    save_path: str = "results/figures/decision_boundaries.png"
):
    """
    Visualizes 2D decision boundaries for multiple models under clean vs noisy conditions.
    models_dict format:
    {
        'Clean LR': lr_clean, 'Noisy LR': lr_noisy, 'Mitigated LR': lr_mitigated,
        'Clean RF': rf_clean, ...
    }
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n_models = len(models_dict)
    cols = 3
    rows = int(np.ceil(n_models / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = np.array(axes).reshape(-1)

    # Grid for contour
    x_min, x_max = X[:, 0].min() - 0.6, X[:, 0].max() + 0.6
    y_min, y_max = X[:, 1].min() - 0.6, X[:, 1].max() + 0.6
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    for idx, (title, clf) in enumerate(models_dict.items()):
        ax = axes[idx]
        # Predict meshgrid probabilities or classes
        try:
            probs = clf.predict_proba(grid_points)[:, 1]
            Z = probs.reshape(xx.shape)
            contour = ax.contourf(xx, yy, Z, levels=20, cmap="coolwarm", alpha=0.6)
            ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=1.5)
        except Exception:
            preds = clf.predict(grid_points)
            Z = preds.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap="coolwarm", alpha=0.5)

        # Plot data points
        if "Clean" in title:
            labels_to_plot = y_true
            corrupted = np.zeros(len(y_true), dtype=bool)
        else:
            labels_to_plot = y_noisy
            corrupted = (y_true != y_noisy)

        # Correct labels
        ax.scatter(
            X[~corrupted, 0], X[~corrupted, 1],
            c=labels_to_plot[~corrupted],
            cmap="coolwarm",
            edgecolors="k",
            s=25,
            alpha=0.8,
            label="Original"
        )
        # Corrupted labels with yellow cross highlight
        if corrupted.any():
            ax.scatter(
                X[corrupted, 0], X[corrupted, 1],
                c="gold",
                marker="x",
                s=40,
                linewidths=1.8,
                label=f"Flipped ({int(noise_rate*100)}%)"
            )

        ax.set_title(title, weight="bold", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide extra subplots
    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        f"Decision Boundaries: Clean vs {int(noise_rate*100)}% Noisy vs Mitigated Models",
        weight="bold",
        fontsize=15,
        y=0.99
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
