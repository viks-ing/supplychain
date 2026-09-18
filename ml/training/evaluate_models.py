"""
Model Evaluation and Diagnostic Plotting for Vyuha ML.
Generates Predicted vs Actual plots, error residual distributions,
and feature importance charts across all three targets.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.feature_engineering import CANONICAL_FEATURES


def generate_evaluation_artifacts():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "training_dataset.csv")
    models_dir = os.path.join(base_dir, "models")
    eval_dir = os.path.join(base_dir, "evaluation")
    plots_dir = os.path.join(eval_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    X = df[CANONICAL_FEATURES].values
    groups = df["network_family"].values

    # Same 70/15/15 split
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=42)
    train_idx, temp_idx = next(gss1.split(X, groups=groups))
    temp_groups = groups[temp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=42)
    val_rel_idx, test_rel_idx = next(gss2.split(temp_idx, groups=temp_groups))
    test_idx = temp_idx[test_rel_idx]

    X_test = X[test_idx]

    targets_info = [
        {"name": "cost_impact_pct", "label": "Cost Impact (%)", "file": "cost_impact_xgb.joblib"},
        {"name": "delay_days", "label": "Additional Delay (Days)", "file": "delay_xgb.joblib"},
        {"name": "risk_score", "label": "Supply Chain Risk Score (0-100)", "file": "risk_xgb.joblib"}
    ]

    predictions = {}
    metrics_summary = {}

    for t_info in targets_info:
        target = t_info["name"]
        model_path = os.path.join(models_dir, t_info["file"])
        model = joblib.load(model_path)

        y_test = df[target].values[test_idx]
        y_pred = model.predict(X_test)
        predictions[target] = (y_test, y_pred)

        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))

        metrics_summary[target] = {
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "r2": round(r2, 4)
        }

    # -------------------------------------------------------------
    # 1. Plot: Predicted vs Actual
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for i, t_info in enumerate(targets_info):
        target = t_info["name"]
        y_test, y_pred = predictions[target]
        m = metrics_summary[target]

        axes[i].scatter(y_test, y_pred, alpha=0.45, color="#1f77b4", edgecolors="none", s=22)
        min_v = min(np.min(y_test), np.min(y_pred))
        max_v = max(np.max(y_test), np.max(y_pred))
        axes[i].plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1.5, label="Perfect Fit (y=x)")
        axes[i].set_title(f"{t_info['label']}\nR² = {m['r2']:.4f} | RMSE = {m['rmse']:.3f} | MAE = {m['mae']:.3f}", fontsize=11, fontweight="bold")
        axes[i].set_xlabel(f"Actual {t_info['label']}", fontsize=10)
        axes[i].set_ylabel(f"Predicted {t_info['label']}", fontsize=10)
        axes[i].legend(loc="upper left")
        axes[i].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    pred_plot_path = os.path.join(plots_dir, "predicted_vs_actual.png")
    plt.savefig(pred_plot_path, dpi=180)
    plt.close()
    print(f"Saved: {pred_plot_path}")

    # -------------------------------------------------------------
    # 2. Plot: Residual Error Distributions
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, t_info in enumerate(targets_info):
        target = t_info["name"]
        y_test, y_pred = predictions[target]
        residuals = y_pred - y_test

        axes[i].hist(residuals, bins=35, color="#2ca02c", edgecolor="black", alpha=0.75, density=True)
        axes[i].axvline(0, color="red", linestyle="--", linewidth=1.5)
        axes[i].set_title(f"Residuals: {t_info['label']}\nMean: {np.mean(residuals):.3f} | Std: {np.std(residuals):.3f}", fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Prediction Error (Predicted - Actual)", fontsize=10)
        axes[i].set_ylabel("Density", fontsize=10)
        axes[i].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    error_plot_path = os.path.join(plots_dir, "error_distribution.png")
    plt.savefig(error_plot_path, dpi=180)
    plt.close()
    print(f"Saved: {error_plot_path}")

    # -------------------------------------------------------------
    # 3. Plot: Feature Importances
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    for i, t_info in enumerate(targets_info):
        target = t_info["name"]
        model = joblib.load(os.path.join(models_dir, t_info["file"]))
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1][:10]

        top_names = [CANONICAL_FEATURES[idx] for idx in sorted_idx][::-1]
        top_scores = [importances[idx] for idx in sorted_idx][::-1]

        axes[i].barh(top_names, top_scores, color="#ff7f0e", alpha=0.85, edgecolor="none")
        axes[i].set_title(f"Top 10 Features: {t_info['label']}", fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Relative Importance (Gain)", fontsize=10)
        axes[i].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fi_plot_path = os.path.join(plots_dir, "feature_importance.png")
    plt.savefig(fi_plot_path, dpi=180)
    plt.close()
    print(f"Saved: {fi_plot_path}")

    print("\nModel Evaluation Artifacts Generated Successfully!")


if __name__ == "__main__":
    generate_evaluation_artifacts()
