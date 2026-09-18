"""
Comprehensive Model Evaluation and Diagnostic Suite for Vyuha ML.
Generates:
1. Regression metrics (R², RMSE, MAE, MAPE, Median AE)
2. Multi-Class Risk Tier Confusion Matrix (Low, Moderate, High, Critical)
3. Binary High-Risk Alert Confusion Matrix & Metrics (Accuracy, Precision, Recall, Specificity, Balanced Acc, F1, MCC, ROC-AUC)
4. Critical Delay & Cost Surge Confusion Matrices
5. High-resolution diagnostic plots and heatmaps in ml/evaluation/plots/
"""

import os
import sys
import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, median_absolute_error,
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, balanced_accuracy_score, cohen_kappa_score
)
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.feature_engineering import CANONICAL_FEATURES


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error with numerical safeguarding."""
    mask = np.abs(y_true) > 0.5
    if np.sum(mask) == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def to_risk_tier(scores: np.ndarray) -> np.ndarray:
    """Categorizes numerical 0-100 risk score into standard risk tiers."""
    return np.select(
        [scores <= 25.0, scores <= 50.0, scores <= 75.0],
        ["Low", "Moderate", "High"],
        default="Critical"
    )


def compute_binary_confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold_name: str) -> Dict[str, Any]:
    """Computes all classification metrics derived directly according to the 2x2 confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp = int(cm[0][0]), int(cm[0][1])
    fn, tp = int(cm[1][0]), int(cm[1][1])
    total = tn + fp + fn + tp

    acc = float((tp + tn) / total) if total > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    bal_acc = float((rec + spec) / 2.0)
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    mcc = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0

    return {
        "condition": threshold_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall_sensitivity": round(rec, 4),
        "specificity": round(spec, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "f1_score": round(f1, 4),
        "matthews_corrcoef": round(mcc, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "confusion_matrix_breakdown": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "total_samples": total
        }
    }


def compute_multiclass_tier_metrics(tier_true: np.ndarray, tier_pred: np.ndarray, labels: List[str]) -> Dict[str, Any]:
    """Computes comprehensive metrics according to the multi-class confusion matrix."""
    cm = confusion_matrix(tier_true, tier_pred, labels=labels)
    acc = float(accuracy_score(tier_true, tier_pred))
    bal_acc = float(balanced_accuracy_score(tier_true, tier_pred))
    kappa = float(cohen_kappa_score(tier_true, tier_pred))

    macro_prec = float(precision_score(tier_true, tier_pred, labels=labels, average="macro", zero_division=0))
    macro_rec = float(recall_score(tier_true, tier_pred, labels=labels, average="macro", zero_division=0))
    macro_f1 = float(f1_score(tier_true, tier_pred, labels=labels, average="macro", zero_division=0))

    weighted_prec = float(precision_score(tier_true, tier_pred, labels=labels, average="weighted", zero_division=0))
    weighted_rec = float(recall_score(tier_true, tier_pred, labels=labels, average="weighted", zero_division=0))
    weighted_f1 = float(f1_score(tier_true, tier_pred, labels=labels, average="weighted", zero_division=0))

    per_class = {}
    total_samples = len(tier_true)
    for i, label in enumerate(labels):
        tp = int(cm[i, i])
        fn = int(np.sum(cm[i, :]) - tp)
        fp = int(np.sum(cm[:, i]) - tp)
        tn = int(total_samples - tp - fp - fn)

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        s = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        per_class[label] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "specificity": round(s, 4),
            "f1_score": round(f, 4),
            "support": int(np.sum(cm[i, :])),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn
        }

    return {
        "classes": labels,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "cohen_kappa": round(kappa, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_precision": round(weighted_prec, 4),
        "weighted_recall": round(weighted_rec, 4),
        "weighted_f1": round(weighted_f1, 4),
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class
    }


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

    # 70/15/15 Group split by network family
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
    regression_summary = {}

    for t_info in targets_info:
        target = t_info["name"]
        model_path = os.path.join(models_dir, t_info["file"])
        model = joblib.load(model_path)

        y_test = df[target].values[test_idx]
        y_pred = model.predict(X_test)
        predictions[target] = (y_test, y_pred)

        mae = float(mean_absolute_error(y_test, y_pred))
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))
        mape = float(calculate_mape(y_test, y_pred))
        med_ae = float(median_absolute_error(y_test, y_pred))

        regression_summary[target] = {
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "r2": round(r2, 4),
            "mape": round(mape, 2),
            "median_absolute_error": round(med_ae, 3)
        }

    # =============================================================
    # Classification Metrics Followed According to Confusion Matrix
    # =============================================================
    y_risk_true, y_risk_pred = predictions["risk_score"]
    y_delay_true, y_delay_pred = predictions["delay_days"]
    y_cost_true, y_cost_pred = predictions["cost_impact_pct"]

    # 1. Multi-Class Risk Tier (Low, Moderate, High, Critical)
    tier_labels = ["Low", "Moderate", "High", "Critical"]
    tier_true = to_risk_tier(y_risk_true)
    tier_pred = to_risk_tier(y_risk_pred)
    tier_metrics = compute_multiclass_tier_metrics(tier_true, tier_pred, tier_labels)

    # 2. Binary High-Risk Alert (Risk Score > 50)
    bin_risk_true = (y_risk_true > 50.0).astype(int)
    bin_risk_pred = (y_risk_pred > 50.0).astype(int)
    bin_risk_metrics = compute_binary_confusion_metrics(bin_risk_true, bin_risk_pred, "risk_score > 50.0")

    # 3. Critical Delay Alert (Delay > 5 days)
    bin_delay_true = (y_delay_true > 5.0).astype(int)
    bin_delay_pred = (y_delay_pred > 5.0).astype(int)
    delay_alert_metrics = compute_binary_confusion_metrics(bin_delay_true, bin_delay_pred, "delay_days > 5.0")

    # 4. Severe Cost Surge Alert (Cost Impact > 15%)
    bin_cost_true = (y_cost_true > 15.0).astype(int)
    bin_cost_pred = (y_cost_pred > 15.0).astype(int)
    cost_alert_metrics = compute_binary_confusion_metrics(bin_cost_true, bin_cost_pred, "cost_impact_pct > 15.0%")

    classification_summary = {
        "risk_tier_multiclass": tier_metrics,
        "high_risk_binary_alert": bin_risk_metrics,
        "critical_delay_alert": delay_alert_metrics,
        "cost_surge_alert": cost_alert_metrics
    }

    # Assemble and save enhanced metrics.json
    all_metrics = {
        "regression_benchmarks": regression_summary,
        "classification_confusion_matrices": classification_summary
    }

    metrics_json_path = os.path.join(eval_dir, "metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved expanded evaluation metrics to: {metrics_json_path}")

    # =============================================================
    # Diagnostic Plots
    # =============================================================

    # 1. Plot: Predicted vs Actual
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for i, t_info in enumerate(targets_info):
        target = t_info["name"]
        y_test, y_pred = predictions[target]
        m = regression_summary[target]

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
    plt.savefig(os.path.join(plots_dir, "predicted_vs_actual.png"), dpi=180)
    plt.close()

    # 2. Plot: Residual Error Distributions
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
    plt.savefig(os.path.join(plots_dir, "error_distribution.png"), dpi=180)
    plt.close()

    # 3. Plot: Feature Importances
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
    plt.savefig(os.path.join(plots_dir, "feature_importance.png"), dpi=180)
    plt.close()

    # 4. Plot: Multi-Class Risk Tier Confusion Matrix (4x4)
    cm_4x4 = np.array(tier_metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_4x4, interpolation="nearest", cmap=plt.cm.Blues)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Sample Count", rotation=-90, va="bottom", fontsize=10)

    ax.set(
        xticks=np.arange(len(tier_labels)),
        yticks=np.arange(len(tier_labels)),
        xticklabels=tier_labels,
        yticklabels=tier_labels,
        ylabel="Actual Risk Tier",
        xlabel="Predicted Risk Tier",
        title=f"Supply Chain Risk Tier Confusion Matrix\nAccuracy: {tier_metrics['accuracy']*100:.2f}% | Balanced Acc: {tier_metrics['balanced_accuracy']*100:.2f}% | Macro F1: {tier_metrics['macro_f1']:.4f}"
    )
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=10, fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontsize=10, fontweight="bold")

    thresh = cm_4x4.max() / 2.0
    for r in range(cm_4x4.shape[0]):
        row_sum = np.sum(cm_4x4[r, :])
        for c in range(cm_4x4.shape[1]):
            val = cm_4x4[r, c]
            pct = (val / row_sum * 100.0) if row_sum > 0 else 0.0
            color = "white" if val > thresh else "black"
            ax.text(c, r, f"{val}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    plt.tight_layout()
    cm_tier_path = os.path.join(plots_dir, "confusion_matrix_risk_tier.png")
    plt.savefig(cm_tier_path, dpi=200)
    plt.close()
    print(f"Saved: {cm_tier_path}")

    # 5. Plot: Binary High-Risk Alert Confusion Matrix (2x2)
    bin_cm = np.array(bin_risk_metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(bin_cm, interpolation="nearest", cmap=plt.cm.Greens)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Count", rotation=-90, va="bottom", fontsize=10)

    bin_labels = ["Low / Moderate (≤50)", "High / Critical (>50)"]
    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=bin_labels,
        yticklabels=bin_labels,
        ylabel="Actual Disruption Severity",
        xlabel="Predicted Disruption Severity",
        title=f"Binary High-Risk Alert Confusion Matrix\nAccuracy: {bin_risk_metrics['accuracy']*100:.2f}% | Recall: {bin_risk_metrics['recall_sensitivity']*100:.2f}% | Specificity: {bin_risk_metrics['specificity']*100:.2f}%"
    )
    plt.setp(ax.get_xticklabels(), fontsize=9, fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontsize=9, fontweight="bold")

    cell_tags = [["True Negative (TN)", "False Positive (FP)"], ["False Negative (FN)", "True Positive (TP)"]]
    thresh_bin = bin_cm.max() / 2.0
    for r in range(2):
        for c in range(2):
            val = bin_cm[r, c]
            tag = cell_tags[r][c]
            color = "white" if val > thresh_bin else "black"
            ax.text(c, r, f"{tag}\n{val}\n({val/np.sum(bin_cm)*100:.1f}%)", ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    plt.tight_layout()
    cm_bin_path = os.path.join(plots_dir, "confusion_matrix_binary_risk.png")
    plt.savefig(cm_bin_path, dpi=200)
    plt.close()
    print(f"Saved: {cm_bin_path}")

    # 6. Plot: Side-by-Side Critical Delay & Cost Surge Alerts
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    # Delay alert
    d_cm = np.array(delay_alert_metrics["confusion_matrix"])
    im0 = axes[0].imshow(d_cm, interpolation="nearest", cmap=plt.cm.Oranges)
    axes[0].set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["Normal (≤5d)", "Critical (>5d)"],
        yticklabels=["Normal (≤5d)", "Critical (>5d)"],
        ylabel="Actual Delay",
        xlabel="Predicted Delay",
        title=f"Critical Delay Warning (>5 Days)\nAccuracy: {delay_alert_metrics['accuracy']*100:.2f}% | F1: {delay_alert_metrics['f1_score']:.4f}"
    )
    for r in range(2):
        for c in range(2):
            axes[0].text(c, r, f"{d_cm[r,c]}\n({d_cm[r,c]/np.sum(d_cm)*100:.1f}%)", ha="center", va="center", color="white" if d_cm[r,c]>d_cm.max()/2 else "black", fontweight="bold")

    # Cost alert
    c_cm = np.array(cost_alert_metrics["confusion_matrix"])
    im1 = axes[1].imshow(c_cm, interpolation="nearest", cmap=plt.cm.Purples)
    axes[1].set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["Standard (≤15%)", "Surge (>15%)"],
        yticklabels=["Standard (≤15%)", "Surge (>15%)"],
        ylabel="Actual Cost Surge",
        xlabel="Predicted Cost Surge",
        title=f"Cost Surge Alert (>15% Impact)\nAccuracy: {cost_alert_metrics['accuracy']*100:.2f}% | F1: {cost_alert_metrics['f1_score']:.4f}"
    )
    for r in range(2):
        for c in range(2):
            axes[1].text(c, r, f"{c_cm[r,c]}\n({c_cm[r,c]/np.sum(c_cm)*100:.1f}%)", ha="center", va="center", color="white" if c_cm[r,c]>c_cm.max()/2 else "black", fontweight="bold")

    plt.tight_layout()
    cm_alerts_path = os.path.join(plots_dir, "confusion_matrix_alerts.png")
    plt.savefig(cm_alerts_path, dpi=200)
    plt.close()
    print(f"Saved: {cm_alerts_path}")

    print("\n" + "="*70)
    print("EVALUATION METRICS STRICTLY DERIVED ACCORDING TO CONFUSION MATRIX:")
    print("="*70)
    print(f"1. Risk Tier Multi-Class Accuracy:  {tier_metrics['accuracy']*100:.2f}%")
    print(f"   Balanced Accuracy:              {tier_metrics['balanced_accuracy']*100:.2f}%")
    print(f"   Cohen Kappa:                    {tier_metrics['cohen_kappa']:.4f}")
    print(f"   Macro Precision / Recall / F1:  {tier_metrics['macro_precision']:.4f} / {tier_metrics['macro_recall']:.4f} / {tier_metrics['macro_f1']:.4f}")
    print(f"2. Binary High-Risk Alert Accuracy: {bin_risk_metrics['accuracy']*100:.2f}%")
    print(f"   Sensitivity (Recall):           {bin_risk_metrics['recall_sensitivity']*100:.2f}%")
    print(f"   Specificity:                    {bin_risk_metrics['specificity']*100:.2f}%")
    print(f"   Precision:                      {bin_risk_metrics['precision']*100:.2f}%")
    print(f"   F1 Score:                       {bin_risk_metrics['f1_score']:.4f}")
    print(f"   Matthews Correlation (MCC):     {bin_risk_metrics['matthews_corrcoef']:.4f}")
    print(f"3. Critical Delay Alert Accuracy:   {delay_alert_metrics['accuracy']*100:.2f}% | Specificity: {delay_alert_metrics['specificity']*100:.2f}%")
    print(f"4. Cost Surge Alert Accuracy:       {cost_alert_metrics['accuracy']*100:.2f}% | Specificity: {cost_alert_metrics['specificity']*100:.2f}%")
    print("="*70)


if __name__ == "__main__":
    generate_evaluation_artifacts()
