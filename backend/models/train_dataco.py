"""
Kaggle DataCo Smart Supply Chain Model Training Pipeline using XGBoost.
Trains models on the 180,519-record DataCo Supply Chain Dataset for:
1. Late Delivery Risk (Binary Classification & Risk Probability)
2. Delivery Delay Days (Regression: Actual - Scheduled Days)
3. Benefit / Profit Impact (Regression)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score, confusion_matrix,
    matthews_corrcoef, balanced_accuracy_score
)
import xgboost as xgb
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.dataco_feature_engineering import DataCoFeaturePipeline


def train_dataco_models():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "DataCoSupplyChainDataset.csv")
    artifacts_dir = os.path.join(base_dir, "models", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    print(f"Loading Kaggle DataCo Supply Chain Dataset from {data_path}...")
    df = pd.read_csv(data_path, encoding='latin1')
    print(f"Dataset shape: {df.shape}")
    
    # Calculate delivery delay days
    df['delay_days'] = df['Days for shipping (real)'] - df['Days for shipment (scheduled)']
    
    # 80/20 train/test split
    train_df, test_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df['Late_delivery_risk'])
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    # Feature Pipeline
    pipeline = DataCoFeaturePipeline()
    X_train = pipeline.fit_transform(train_df)
    X_test = pipeline.transform(test_df)
    
    pipeline_path = os.path.join(artifacts_dir, "dataco_feature_pipeline.joblib")
    joblib.dump(pipeline, pipeline_path)
    print(f"Saved DataCo Feature Pipeline to {pipeline_path}")
    
    results = {}
    feature_importances = {}
    
    # -------------------------------------------------------------
    # 1. Target 1: Late Delivery Risk (Binary Classification)
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("Training XGBoost Classifier for Late_delivery_risk")
    print("="*60)
    y_train_late = train_df['Late_delivery_risk'].values
    y_test_late = test_df['Late_delivery_risk'].values
    
    xgb_classifier = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )
    xgb_classifier.fit(X_train, y_train_late)
    y_pred_late = xgb_classifier.predict(X_test)
    y_prob_late = xgb_classifier.predict_proba(X_test)[:, 1]
    
    acc = float(accuracy_score(y_test_late, y_pred_late))
    prec = float(precision_score(y_test_late, y_pred_late))
    rec = float(recall_score(y_test_late, y_pred_late))
    f1 = float(f1_score(y_test_late, y_pred_late))
    auc = float(roc_auc_score(y_test_late, y_prob_late))
    cm = confusion_matrix(y_test_late, y_pred_late).tolist()
    
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    balanced_acc = float((rec + specificity) / 2.0)
    mcc = float(matthews_corrcoef(y_test_late, y_pred_late))
    
    print(f"  Late Delivery Risk -> Accuracy: {acc*100:.2f}%, Specificity: {specificity*100:.2f}%, F1: {f1:.4f}, ROC-AUC: {auc:.4f}, MCC: {mcc:.4f}")
    
    clf_path = os.path.join(artifacts_dir, "dataco_late_delivery_xgb.joblib")
    joblib.dump(xgb_classifier, clf_path)
    print(f"  Saved Late Delivery Classifier to {clf_path}")
    
    results['late_delivery_risk'] = {
        'task': 'classification',
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'specificity': round(specificity, 4),
        'balanced_accuracy': round(balanced_acc, 4),
        'f1_score': round(f1, 4),
        'matthews_corrcoef': round(mcc, 4),
        'false_positive_rate': round(fpr, 4),
        'false_negative_rate': round(fnr, 4),
        'roc_auc': round(auc, 4),
        'confusion_matrix': cm,
        'confusion_matrix_breakdown': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
    }
    
    # Feature importances for classification
    clf_fi = xgb_classifier.feature_importances_
    top_fi_idx = np.argsort(clf_fi)[::-1]
    feature_importances['late_delivery_risk'] = [
        {'feature': pipeline.feature_names[i], 'importance': round(float(clf_fi[i]), 4)}
        for i in top_fi_idx
    ]
    
    # -------------------------------------------------------------
    # 2. Target 2: Delivery Delay Days (Regression)
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("Training XGBoost Regressor for Delivery Delay Days")
    print("="*60)
    y_train_delay = train_df['delay_days'].values
    y_test_delay = test_df['delay_days'].values
    
    xgb_reg_delay = xgb.XGBRegressor(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )
    xgb_reg_delay.fit(X_train, y_train_delay)
    y_pred_delay = xgb_reg_delay.predict(X_test)
    
    mae_delay = float(mean_absolute_error(y_test_delay, y_pred_delay))
    rmse_delay = float(np.sqrt(mean_squared_error(y_test_delay, y_pred_delay)))
    r2_delay = float(r2_score(y_test_delay, y_pred_delay))
    
    print(f"  Delivery Delay Days -> R2: {r2_delay:.4f}, RMSE: {rmse_delay:.3f}, MAE: {mae_delay:.3f}")
    
    delay_path = os.path.join(artifacts_dir, "dataco_delay_days_xgb.joblib")
    joblib.dump(xgb_reg_delay, delay_path)
    print(f"  Saved Delay Days Regressor to {delay_path}")
    
    results['delivery_delay_days'] = {
        'task': 'regression',
        'r2': round(r2_delay, 4),
        'rmse': round(rmse_delay, 3),
        'mae': round(mae_delay, 3)
    }
    
    delay_fi = xgb_reg_delay.feature_importances_
    top_delay_idx = np.argsort(delay_fi)[::-1]
    feature_importances['delivery_delay_days'] = [
        {'feature': pipeline.feature_names[i], 'importance': round(float(delay_fi[i]), 4)}
        for i in top_delay_idx
    ]
    
    # -------------------------------------------------------------
    # 3. Target 3: Benefit / Profit per Order (Regression)
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("Training XGBoost Regressor for Benefit per order")
    print("="*60)
    y_train_profit = train_df['Benefit per order'].values
    y_test_profit = test_df['Benefit per order'].values
    
    xgb_reg_profit = xgb.XGBRegressor(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )
    xgb_reg_profit.fit(X_train, y_train_profit)
    y_pred_profit = xgb_reg_profit.predict(X_test)
    
    mae_profit = float(mean_absolute_error(y_test_profit, y_pred_profit))
    rmse_profit = float(np.sqrt(mean_squared_error(y_test_profit, y_pred_profit)))
    r2_profit = float(r2_score(y_test_profit, y_pred_profit))
    
    print(f"  Benefit per Order -> R2: {r2_profit:.4f}, RMSE: {rmse_profit:.3f}, MAE: {mae_profit:.3f}")
    
    profit_path = os.path.join(artifacts_dir, "dataco_profit_xgb.joblib")
    joblib.dump(xgb_reg_profit, profit_path)
    print(f"  Saved Profit Regressor to {profit_path}")
    
    results['benefit_per_order'] = {
        'task': 'regression',
        'r2': round(r2_profit, 4),
        'rmse': round(rmse_profit, 3),
        'mae': round(mae_profit, 3)
    }
    
    profit_fi = xgb_reg_profit.feature_importances_
    top_profit_idx = np.argsort(profit_fi)[::-1]
    feature_importances['benefit_per_order'] = [
        {'feature': pipeline.feature_names[i], 'importance': round(float(profit_fi[i]), 4)}
        for i in top_profit_idx
    ]
    
    # Export full metrics
    metrics_export = {
        "dataset_name": "Kaggle DataCo Smart Supply Chain for Big Data Analysis",
        "dataset_size": len(df),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "features_count": len(pipeline.feature_names),
        "feature_names": pipeline.feature_names,
        "models": results,
        "feature_importances": feature_importances
    }
    
    metrics_path = os.path.join(artifacts_dir, "dataco_evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_export, f, indent=2)
    print(f"\nSaved DataCo evaluation metrics to {metrics_path}")
    print("\nTraining complete!")

if __name__ == "__main__":
    train_dataco_models()
