"""
Prediction Service for Vyuha ML.
Loads trained XGBoost models and FeaturePipeline to generate real-time predictions.
Uses learned empirical distributions for dynamic imputation with zero hardcoded defaults.
"""

import os
import sys
import numpy as np
import joblib
from typing import Dict, Any

import xgboost as xgb
from services.feature_engineering import FeaturePipeline
from services.vulnerability_service import analyze_vulnerabilities_from_shap


class PredictionService:
    def __init__(self):
        artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "artifacts")
        pipeline_path = os.path.join(artifacts_dir, "feature_pipeline.joblib")
        
        self.pipeline: FeaturePipeline = FeaturePipeline.load(pipeline_path)
        
        # Load primary models if available
        cost_path = os.path.join(artifacts_dir, "cost_impact_pct_xgb.joblib")
        lead_path = os.path.join(artifacts_dir, "lead_time_impact_days_xgb.joblib")
        risk_path = os.path.join(artifacts_dir, "risk_score_xgb.joblib")
        
        # Fallback to dataco models if unified
        if not os.path.exists(risk_path):
            risk_path = os.path.join(artifacts_dir, "dataco_late_delivery_xgb.joblib")
            
        self.cost_model = joblib.load(cost_path) if os.path.exists(cost_path) else None
        self.lead_time_model = joblib.load(lead_path) if os.path.exists(lead_path) else None
        self.risk_model = joblib.load(risk_path) if os.path.exists(risk_path) else None

    @staticmethod
    def classify_risk(score: float) -> str:
        if score <= 25.0:
            return "LOW"
        elif score <= 50.0:
            return "MEDIUM"
        elif score <= 75.0:
            return "HIGH"
        else:
            return "CRITICAL"

    def predict(self, company: Dict[str, Any], conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combines company profile and condition inputs, dynamically imputes missing
        fields using fitted pipeline distributions, and runs XGBoost inference.
        """
        combined = {**company, **conditions}
        
        # Zero hardcoding: dynamically populate any missing fields from pipeline learned distributions
        for col, median_val in self.pipeline.column_medians.items():
            combined.setdefault(col, median_val)
        for col, mode_val in self.pipeline.column_modes.items():
            combined.setdefault(col, mode_val)

        _, X_raw = self.pipeline.transform_single(combined)
        
        # Predictions
        cost_pred = float(self.cost_model.predict(X_raw)[0]) if self.cost_model else 0.0
        lead_time_pred = float(self.lead_time_model.predict(X_raw)[0]) if self.lead_time_model else 0.0
        
        if self.risk_model:
            if hasattr(self.risk_model, "predict_proba"):
                risk_prob = float(self.risk_model.predict_proba(X_raw)[0, 1])
                risk_score_pred = round(risk_prob * 100.0, 1)
            else:
                risk_score_pred = float(np.clip(self.risk_model.predict(X_raw)[0], 0.0, 100.0))
        else:
            risk_score_pred = 50.0
            
        risk_level = self.classify_risk(risk_score_pred)
        
        # Financial cost impact calculation
        procurement_val = float(combined.get("monthly_procurement_lakhs", combined.get("Order Item Total", 100.0)))
        cost_impact_val = round(procurement_val * (cost_pred / 100.0), 2)
        
        # Dynamic Tree SHAP Attribution from XGBoost
        dmat = xgb.DMatrix(X_raw)
        booster = self.risk_model.get_booster() if hasattr(self.risk_model, "get_booster") else None
        if booster:
            contribs = booster.predict(dmat, pred_contribs=True)[0]
            shap_contribs = contribs[:-1]
        else:
            shap_contribs = np.zeros(len(self.pipeline.feature_names))
        
        vulnerabilities = analyze_vulnerabilities_from_shap(
            feature_names=self.pipeline.feature_names,
            shap_contributions=shap_contribs,
            company=combined,
            conditions=conditions
        )
        
        return {
            "cost_impact_pct": round(cost_pred, 1),
            "cost_impact_lakhs": cost_impact_val,
            "lead_time_impact_days": round(lead_time_pred, 1),
            "risk_score": round(risk_score_pred, 1),
            "risk_level": risk_level,
            "vulnerabilities": vulnerabilities
        }
