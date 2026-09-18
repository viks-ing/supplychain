"""
Zero-Hardcoding Model Training & Benchmarking Pipeline for Vyuha ML.
Trains XGBoost models directly on the Kaggle DataCo Smart Supply Chain Dataset
(180,519 transactions) using dynamic schema discovery and automated feature pipelines.
"""

import os
import re
import sys
import json
import argparse
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score,
    median_absolute_error
)
import xgboost as xgb
import joblib

# Add parent directory to sys.path to import from services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.feature_engineering import DynamicFeaturePipeline


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error with numerical safeguarding."""
    mask = np.abs(y_true) > 0.5
    if np.sum(mask) == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else 0.0
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    return {
        "task": "classification",
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm
    }


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(calculate_mape(y_true, y_pred))
    med_ae = float(median_absolute_error(y_true, y_pred))
    
    return {
        "task": "regression",
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "mape": round(mape, 2),
        "median_absolute_error": round(med_ae, 4)
    }


def determine_task_type(series: pd.Series) -> str:
    """Dynamically identifies whether a target is classification or regression."""
    unique_count = series.nunique(dropna=True)
    if unique_count <= 10 or pd.api.types.is_bool_dtype(series):
        return "classification"
    return "regression"


def train_models(
    data_path: Optional[str] = None,
    target: Optional[str] = None,
    artifacts_dir: Optional[str] = None,
    test_size: float = 0.20,
    random_state: int = 42,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.08
) -> Dict[str, Any]:
    """
    Zero-hardcoding training function for Kaggle DataCo Supply Chain dataset with XGBoost.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Dynamically resolve dataset path
    if data_path is None:
        data_path = os.environ.get(
            "DATACO_DATA_PATH",
            os.path.join(base_dir, "data", "DataCoSupplyChainDataset.csv")
        )
        
    if artifacts_dir is None:
        artifacts_dir = os.path.join(base_dir, "models", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    print(f"\n=======================================================")
    print(f"Loading Kaggle DataCo Supply Chain Dataset from:\n{data_path}")
    print(f"=======================================================")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at: {data_path}")
        
    df = pd.read_csv(data_path, encoding="latin1")
    print(f"Loaded {len(df):,} records and {len(df.columns)} columns.")
    
    # Compute derived target for delay days dynamically if columns are present
    real_ship_col = next((c for c in df.columns if "shipping (real)" in c.lower()), None)
    sched_ship_col = next((c for c in df.columns if "scheduled" in c.lower()), None)
    if real_ship_col and sched_ship_col and "delay_days" not in df.columns:
        df["delay_days"] = df[real_ship_col] - df[sched_ship_col]
        print(f"Computed dynamic target 'delay_days' ({real_ship_col} - {sched_ship_col}).")
        
    # Discover available targets dynamically
    available_targets = []
    
    if target is not None:
        for t_name in str(target).split(","):
            t_name = t_name.strip()
            if t_name in df.columns:
                task = determine_task_type(df[t_name])
                safe_name = re.sub(r"[^\w]", "_", t_name).strip("_").lower()
                available_targets.append({
                    "column": t_name,
                    "task": task,
                    "filename": f"{safe_name}_xgb.joblib"
                })
    else:
        target_specs = [
            {"name": "Late_delivery_risk", "preferred_task": "classification", "filename": "dataco_late_delivery_xgb.joblib"},
            {"name": "delay_days", "preferred_task": "regression", "filename": "dataco_delay_days_xgb.joblib"},
            {"name": "Order Profit Per Order", "fallback": "Benefit per order", "preferred_task": "regression", "filename": "dataco_profit_xgb.joblib"}
        ]
        
        for spec in target_specs:
            col = spec["name"] if spec["name"] in df.columns else spec.get("fallback")
            if col and col in df.columns:
                available_targets.append({
                    "column": col,
                    "task": spec["preferred_task"],
                    "filename": spec["filename"]
                })
                
        # If no default targets matched, auto-discover candidate targets from data
        if not available_targets:
            candidate_cols = [
                c for c in df.columns
                if re.search(r"(?i)(risk|late|delay|taken|time|hours|duration|profit|score)", str(c))
                and pd.api.types.is_numeric_dtype(df[c])
            ]
            for c in candidate_cols[:2]:
                task = determine_task_type(df[c])
                safe_name = re.sub(r"[^\w]", "_", c).strip("_").lower()
                available_targets.append({
                    "column": c,
                    "task": task,
                    "filename": f"{safe_name}_xgb.joblib"
                })

    print(f"Discovered {len(available_targets)} target objectives: {[t['column'] for t in available_targets]}")
    
    # Extract target column names to exclude them from the feature matrix
    target_col_names = [t["column"] for t in available_targets]
    if real_ship_col:
        target_col_names.append(real_ship_col)
        
    # Split train/test
    stratify_col = "Late_delivery_risk" if "Late_delivery_risk" in df.columns else None
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[stratify_col] if stratify_col else None
    )
    print(f"Train samples: {len(train_df):,}, Test samples: {len(test_df):,}")
    
    # Dynamic Feature Pipeline: fits schema and statistical distributions
    pipeline = DynamicFeaturePipeline(target_cols=target_col_names)
    X_train = pipeline.fit_transform(train_df)
    X_test = pipeline.transform(test_df)
    
    print(f"\nDiscovered Features ({len(pipeline.feature_names)} total):")
    print(f"  Numerical Features ({len(pipeline.numerical_cols)}): {pipeline.numerical_cols[:8]}...")
    print(f"  Categorical Features ({len(pipeline.categorical_cols)}): {pipeline.categorical_cols}")
    
    # Save fitted pipeline
    pipeline_path = os.path.join(artifacts_dir, "dataco_feature_pipeline.joblib")
    pipeline.save(pipeline_path)
    # Also save to feature_pipeline.joblib for unified compatibility
    pipeline.save(os.path.join(artifacts_dir, "feature_pipeline.joblib"))
    print(f"Saved Dynamic Feature Pipeline to {pipeline_path}")
    
    results = {}
    feature_importances = {}
    
    for t_spec in available_targets:
        target_name = t_spec["column"]
        task_type = t_spec["task"]
        model_filename = t_spec["filename"]
        
        print(f"\n{'-'*60}")
        print(f"Training XGBoost Model for Target: '{target_name}' [{task_type.upper()}]")
        print(f"{'-'*60}")
        
        y_train = train_df[target_name].values
        y_test = test_df[target_name].values
        
        if task_type == "classification":
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=0.85,
                colsample_bytree=0.85,
                tree_method="hist",
                random_state=random_state,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
            metrics = evaluate_classification(y_test, y_pred, y_prob)
            print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        else:
            model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=0.85,
                colsample_bytree=0.85,
                tree_method="hist",
                random_state=random_state,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = evaluate_regression(y_test, y_pred)
            print(f"  R2 Score:  {metrics['r2']:.4f}")
            print(f"  RMSE:      {metrics['rmse']:.3f}")
            print(f"  MAE:       {metrics['mae']:.3f}")
            print(f"  MAPE:      {metrics['mape']:.2f}%")
            
        # Save model
        save_path = os.path.join(artifacts_dir, model_filename)
        joblib.dump(model, save_path)
        print(f"  Saved trained model to: {save_path}")
        
        # Feature importances dynamically extracted from booster
        booster_importances = model.feature_importances_
        sorted_indices = np.argsort(booster_importances)[::-1][:10]
        top_importances = [
            {"feature": pipeline.feature_names[i], "importance": round(float(booster_importances[i]), 4)}
            for i in sorted_indices
        ]
        feature_importances[target_name] = top_importances
        results[target_name] = metrics

    metrics_export = {
        "dataset": os.path.basename(data_path),
        "total_records": len(df),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "num_features": len(pipeline.feature_names),
        "features": pipeline.feature_names,
        "targets": results,
        "feature_importances": feature_importances
    }
    
    # Save metrics JSON
    metrics_path = os.path.join(artifacts_dir, "dataco_evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_export, f, indent=2)
    with open(os.path.join(artifacts_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics_export, f, indent=2)
        
    print(f"\n=======================================================")
    print(f"Training Complete! Saved metrics to:\n{metrics_path}")
    print(f"=======================================================\n")
    return metrics_export


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vyuha ML - Kaggle DataCo XGBoost Training Pipeline")
    parser.add_argument("--data", type=str, default=None, help="Path to DataCo Supply Chain CSV dataset")
    parser.add_argument("--target", type=str, default=None, help="Target column name(s) comma-separated (e.g. 'Time_taken (min)', 'delivery_delay_hours')")
    parser.add_argument("--artifacts-dir", type=str, default=None, help="Directory to save artifacts")
    parser.add_argument("--test-size", type=float, default=0.20, help="Test set fraction")
    parser.add_argument("--n-estimators", type=int, default=300, help="Number of XGBoost boosting rounds")
    parser.add_argument("--max-depth", type=int, default=6, help="Maximum tree depth")
    parser.add_argument("--learning-rate", type=float, default=0.08, help="XGBoost learning rate")
    
    args = parser.parse_args()
    train_models(
        data_path=args.data,
        target=args.target,
        artifacts_dir=args.artifacts_dir,
        test_size=args.test_size,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate
    )
