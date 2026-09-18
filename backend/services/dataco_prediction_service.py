"""
Kaggle DataCo Supply Chain Prediction Service.
Loads trained XGBoost models and uses native Tree SHAP (pred_contribs)
to dynamically compute risk probability, transit delays, and feature contributions
without any hardcoded heuristics or rules.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from typing import Dict, Any, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.dataco_feature_engineering import DataCoFeaturePipeline

class DataCoPredictionService:
    def __init__(self):
        artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "artifacts")
        
        pipeline_path = os.path.join(artifacts_dir, "dataco_feature_pipeline.joblib")
        clf_path = os.path.join(artifacts_dir, "dataco_late_delivery_xgb.joblib")
        delay_path = os.path.join(artifacts_dir, "dataco_delay_days_xgb.joblib")
        profit_path = os.path.join(artifacts_dir, "dataco_profit_xgb.joblib")
        
        self.pipeline: DataCoFeaturePipeline = joblib.load(pipeline_path)
        self.classifier = joblib.load(clf_path)
        self.delay_regressor = joblib.load(delay_path)
        self.profit_regressor = joblib.load(profit_path)
        
        # Cache XGBoost booster for high-speed Tree SHAP inference
        self.clf_booster = self.classifier.get_booster()

    @staticmethod
    def classify_risk_tier(prob: float) -> str:
        if prob < 0.25:
            return "LOW"
        elif prob < 0.50:
            return "MEDIUM"
        elif prob < 0.75:
            return "HIGH"
        else:
            return "CRITICAL"

    def predict(self, order_dict: Dict[str, Any]) -> Dict[str, Any]:
        df = pd.DataFrame([order_dict])
        
        # Transform dynamically handles missing values using learned dataset medians/modes
        X = self.pipeline.transform(df)
        
        # 1. Late Delivery Classification & Probability
        late_pred = int(self.classifier.predict(X)[0])
        late_prob = float(self.classifier.predict_proba(X)[0, 1])
        risk_score = round(late_prob * 100.0, 1)
        risk_level = self.classify_risk_tier(late_prob)
        
        # 2. Predicted Transit Delay (days)
        delay_pred = float(self.delay_regressor.predict(X)[0])
        delay_days = round(max(0.0, delay_pred), 1) if late_pred == 1 else round(delay_pred, 1)
        
        # 3. Predicted Benefit per order
        profit_pred = float(self.profit_regressor.predict(X)[0])
        
        # 4. Tree SHAP Feature Attribution (Zero Hardcoding)
        dmat = xgb.DMatrix(X)
        contribs = self.clf_booster.predict(dmat, pred_contribs=True)[0]
        feature_contribs = contribs[:-1]  # Exclude bias/intercept
        
        sorted_idx = np.argsort(np.abs(feature_contribs))[::-1]
        risk_drivers = []
        for idx in sorted_idx[:5]:
            score = float(feature_contribs[idx])
            feat_name = self.pipeline.feature_names[idx]
            val = df[feat_name].iloc[0] if feat_name in df.columns else "Learned Default"
            
            direction = "increased delay risk" if score > 0 else "reduced delay risk"
            risk_drivers.append({
                "feature": feat_name,
                "value": str(val),
                "shap_contribution": round(score, 4),
                "impact": f"SHAP value of {score:+.3f} ({direction})"
            })

        order_summary = {}
        for col in self.pipeline.categorical_cols[:4] + self.pipeline.numerical_cols[:3]:
            order_summary[col] = str(df[col].iloc[0]) if col in df.columns else str(self.pipeline.column_modes.get(col, ""))

        return {
            "late_delivery_risk_probability": round(late_prob, 4),
            "late_delivery_risk_score": risk_score,
            "late_delivery_prediction": late_pred,
            "is_late_risk": bool(late_pred == 1),
            "risk_level": risk_level,
            "predicted_delay_days": delay_days,
            "predicted_profit_per_order": round(profit_pred, 2),
            "order_summary": order_summary,
            "top_risk_drivers": risk_drivers
        }
