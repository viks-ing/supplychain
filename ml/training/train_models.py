"""
Model Training & Benchmarking Pipeline for Vyuha ML.
Trains 3 primary XGBoost Regressors (Cost Impact %, Delay Days, Risk Score)
and evaluates them against Linear Regression (Ridge) and Random Forest baselines.
Uses strict 70% Train / 15% Validation / 15% Test splitting by network family.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.feature_engineering import CANONICAL_FEATURES


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.abs(y_true) > 0.5
    if np.sum(mask) == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(calculate_mape(y_true, y_pred))
    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 4),
        "mape": round(mape, 2)
    }


def train_and_benchmark():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "training_dataset.csv")
    models_dir = os.path.join(base_dir, "models")
    eval_dir = os.path.join(base_dir, "evaluation")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f"Loading Vyuha ML Graph-Driven Training Data from:\n{data_path}")
    print(f"=======================================================")
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} samples with {len(df.columns)} total columns.")

    targets = ["cost_impact_pct", "delay_days", "risk_score"]
    feature_cols = CANONICAL_FEATURES

    # Verify no target leakage in features
    for t in targets:
        assert t not in feature_cols, f"Critical: Target {t} found in feature columns!"

    X = df[feature_cols].values
    groups = df["network_family"].values

    # -------------------------------------------------------------
    # 70% Train / 15% Validation / 15% Test Splitting
    # -------------------------------------------------------------
    # First split: 70% Train, 30% Temp (Val + Test)
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=42)
    train_idx, temp_idx = next(gss1.split(X, groups=groups))

    # Second split: 50% / 50% of Temp -> 15% Val, 15% Test
    temp_groups = groups[temp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=42)
    val_rel_idx, test_rel_idx = next(gss2.split(temp_idx, groups=temp_groups))
    
    val_idx = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]

    print(f"Data Splitting (Group-based by Network Family to prevent structural leakage):")
    print(f"  Train: {len(train_idx):,} samples (70%)")
    print(f"  Val:   {len(val_idx):,} samples (15%)")
    print(f"  Test:  {len(test_idx):,} samples (15%)")

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]

    comparison_results = {}
    feature_importances_dict = {}
    trained_models = {}

    target_hyperparams = {
        "cost_impact_pct": {
            "n_estimators": 250,
            "max_depth": 6,
            "learning_rate": 0.06,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 2
        },
        "delay_days": {
            "n_estimators": 250,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 2
        },
        "risk_score": {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 2
        }
    }

    for target in targets:
        print(f"\n{'-'*60}")
        print(f"Training & Benchmarking Models for Target: '{target}'")
        print(f"{'-'*60}")

        y = df[target].values
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

        # 1. Baseline: Ridge Linear Regression
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        y_pred_ridge = ridge.predict(X_test)
        metrics_ridge = evaluate_predictions(y_test, y_pred_ridge)
        print(f"  [Baseline 1] Ridge Regression -> R2: {metrics_ridge['r2']:.4f} | RMSE: {metrics_ridge['rmse']:.3f} | MAE: {metrics_ridge['mae']:.3f}")

        # 2. Baseline: Random Forest Regressor
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        metrics_rf = evaluate_predictions(y_test, y_pred_rf)
        print(f"  [Baseline 2] Random Forest    -> R2: {metrics_rf['r2']:.4f} | RMSE: {metrics_rf['rmse']:.3f} | MAE: {metrics_rf['mae']:.3f}")

        # 3. Primary Model: XGBoost Regressor
        hp = target_hyperparams[target]
        xgb_model = xgb.XGBRegressor(
            n_estimators=hp["n_estimators"],
            max_depth=hp["max_depth"],
            learning_rate=hp["learning_rate"],
            subsample=hp["subsample"],
            colsample_bytree=hp["colsample_bytree"],
            min_child_weight=hp["min_child_weight"],
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        y_pred_xgb = xgb_model.predict(X_test)
        metrics_xgb = evaluate_predictions(y_test, y_pred_xgb)
        print(f"  [PRIMARY]    XGBoost Regressor -> R2: {metrics_xgb['r2']:.4f} | RMSE: {metrics_xgb['rmse']:.3f} | MAE: {metrics_xgb['mae']:.3f}")

        comparison_results[target] = {
            "linear_regression": metrics_ridge,
            "random_forest": metrics_rf,
            "xgboost": metrics_xgb,
            "best_model": "xgboost"
        }

        # Save XGBoost Model
        model_filename = f"{target.replace('_pct', '')}_xgb.joblib" if "cost" in target else f"{target.replace('_score', '')}_xgb.joblib"
        if target == "cost_impact_pct":
            model_filename = "cost_impact_xgb.joblib"
        elif target == "delay_days":
            model_filename = "delay_xgb.joblib"
        elif target == "risk_score":
            model_filename = "risk_xgb.joblib"
            
        model_save_path = os.path.join(models_dir, model_filename)
        joblib.dump(xgb_model, model_save_path)
        print(f"  Saved primary model to: {model_save_path}")

        trained_models[target] = xgb_model

        # Extract Feature Importances
        importances = xgb_model.feature_importances_
        feature_importances_dict[target] = {
            feat: round(float(imp), 5) for feat, imp in zip(feature_cols, importances)
        }

    # -------------------------------------------------------------
    # Save Feature Schema (Section 14 requirement)
    # -------------------------------------------------------------
    schema_path = os.path.join(models_dir, "feature_schema.json")
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump({"features": feature_cols}, f, indent=2)
    print(f"\nSaved feature schema to: {schema_path}")

    # -------------------------------------------------------------
    # Save Model Metadata (Section 17 requirement)
    # -------------------------------------------------------------
    metadata = {
        "model_version": "vyuha-xgb-v2",
        "training_samples": len(train_idx),
        "validation_samples": len(val_idx),
        "test_samples": len(test_idx),
        "features": feature_cols,
        "targets": targets,
        "algorithm": "XGBoost",
        "training_data_type": "representative network simulations using public baseline indicators",
        "geographic_scope": "India",
        "benchmark_metrics": comparison_results
    }
    metadata_path = os.path.join(models_dir, "model_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to: {metadata_path}")

    # -------------------------------------------------------------
    # Save Evaluation Metrics & Feature Importances (Section 19)
    # -------------------------------------------------------------
    metrics_path = os.path.join(eval_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)
    print(f"Saved comparative metrics to: {metrics_path}")

    # Feature Importance CSV
    fi_df = pd.DataFrame(feature_importances_dict)
    fi_df.index.name = "feature"
    fi_csv_path = os.path.join(eval_dir, "feature_importance.csv")
    fi_df.to_csv(fi_csv_path)
    print(f"Saved feature importances to: {fi_csv_path}")

    print("\nTraining and Benchmarking Pipeline Completed Successfully!")
    return comparison_results


if __name__ == "__main__":
    train_and_benchmark()
