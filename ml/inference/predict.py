"""
Production Inference Engine for Vyuha ML Graph-Driven Architecture.
Provides the single canonical entry-point:
`predict_supply_chain_impact(network_state, shock_state, external_conditions)`
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Any, List, Optional

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.graph_engine import SupplyChainGraph, propagate_shock
from features.feature_engineering import extract_ml_features, CANONICAL_FEATURES, feature_dict_to_array


class SupplyChainPredictor:
    """
    Singleton / Cached Predictor that loads the 3 trained XGBoost models
    and provides deterministic simulation and real-time inference.
    """

    def __init__(self, models_dir: Optional[str] = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = models_dir or os.path.join(base_dir, "models")

        cost_path = os.path.join(self.models_dir, "cost_impact_xgb.joblib")
        delay_path = os.path.join(self.models_dir, "delay_xgb.joblib")
        risk_path = os.path.join(self.models_dir, "risk_xgb.joblib")

        if not os.path.exists(cost_path):
            raise FileNotFoundError(f"Model artifacts not found in {self.models_dir}. Please run train_models.py first.")

        self.cost_model: xgb.XGBRegressor = joblib.load(cost_path)
        self.delay_model: xgb.XGBRegressor = joblib.load(delay_path)
        self.risk_model: xgb.XGBRegressor = joblib.load(risk_path)

        # Cache native booster for instant Tree SHAP attribution
        self.risk_booster = self.risk_model.get_booster()

    @staticmethod
    def classify_risk_tier(score: float) -> str:
        """Categorizes numerical 0-100 risk score for UI presentation."""
        if score <= 25.0:
            return "Low"
        elif score <= 50.0:
            return "Moderate"
        elif score <= 75.0:
            return "High"
        else:
            return "Critical"

    def predict(
        self,
        network_state: Dict[str, Any],
        shock_state: Dict[str, Any],
        external_conditions: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        End-to-end pipeline:
        1. Run deterministic graph simulation & disruption propagation.
        2. Extract ML features via canonical extract_ml_features().
        3. Run inference with all 3 XGBoost models.
        4. Calculate Tree SHAP feature attributions.
        5. Return structured JSON output.
        """
        # 1. Graph simulation & shock propagation
        graph = SupplyChainGraph(network_state)
        sim_result = propagate_shock(graph, shock_state)

        # 2. Canonical feature extraction (strictly zero target leakage)
        features = extract_ml_features(
            network_state=network_state,
            shock_state=sim_result,
            external_conditions=external_conditions
        )
        X_vec = feature_dict_to_array(features)

        # 3. XGBoost Inference
        cost_pred = float(self.cost_model.predict(X_vec)[0])
        delay_pred = float(self.delay_model.predict(X_vec)[0])
        risk_pred = float(self.risk_model.predict(X_vec)[0])

        # Safeguards and bounds
        cost_impact_pct = max(0.0, round(cost_pred, 1))
        delay_days = max(0.0, round(delay_pred, 1))
        risk_score = max(0.0, min(100.0, round(risk_pred, 1)))
        risk_tier = self.classify_risk_tier(risk_score)

        # 4. Native Tree SHAP Root-Cause Attribution
        dmat = xgb.DMatrix(X_vec, feature_names=CANONICAL_FEATURES)
        contribs = self.risk_booster.predict(dmat, pred_contribs=True)[0]
        feature_contribs = contribs[:-1]  # Exclude bias/base margin
        
        top_idx = np.argsort(np.abs(feature_contribs))[::-1][:5]
        top_drivers = []
        for idx in top_idx:
            feat = CANONICAL_FEATURES[idx]
            val = features[feat]
            shap_val = float(feature_contribs[idx])
            direction = "increases vulnerability" if shap_val > 0 else "reduces risk exposure"
            top_drivers.append({
                "feature": feat,
                "value": val,
                "shap_impact": round(shap_val, 3),
                "explanation": f"{feat} = {val} ({direction}, SHAP: {shap_val:+.3f})"
            })

        # 5. Assemble structured JSON response
        return {
            "cost_impact_pct": cost_impact_pct,
            "delay_days": delay_days,
            "risk_score": risk_score,
            "risk_tier": risk_tier,
            "affected_nodes": [
                {
                    "node_id": sim_result.get("affected_node_id"),
                    "node_name": sim_result.get("affected_node"),
                    "node_type": sim_result.get("affected_node_type"),
                    "capacity_drop_pct": sim_result.get("affected_capacity_change_pct"),
                    "supply_loss_pct": sim_result.get("affected_supply_percent")
                }
            ],
            "alternative_routes": [
                {
                    "route_path": route,
                    "additional_transit_days": sim_result.get("additional_transit_days"),
                    "additional_transport_cost_pct": sim_result.get("additional_transport_cost_pct")
                }
                for route in sim_result.get("available_alternate_routes", [])
            ],
            "top_risk_drivers": top_drivers,
            "features": features
        }


# Global cached predictor instance
_predictor_instance: Optional[SupplyChainPredictor] = None

def get_predictor() -> SupplyChainPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = SupplyChainPredictor()
    return _predictor_instance


def predict_supply_chain_impact(
    network_state: Dict[str, Any],
    shock_state: Dict[str, Any],
    external_conditions: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    The canonical standalone function required by Section 15:
    predict_supply_chain_impact(network_state, shock_state, external_conditions)
    """
    predictor = get_predictor()
    return predictor.predict(network_state, shock_state, external_conditions)


if __name__ == "__main__":
    # Smoke test with generic attribute shock on an arbitrary dynamic network
    sample_network = {
        "id": "sample_dynamic_net",
        "name": "Dynamic Production Network",
        "industry": "General Engineering",
        "nodes": [
            {"id": "node_origin", "name": "Input Supplier", "type": "supplier", "is_primary": True, "share_of_supply": 1.0},
            {"id": "node_gateway", "name": "Logistics Gateway", "type": "port", "is_primary": True, "share_of_volume": 1.0, "congestion": 15.0},
            {"id": "node_hub", "name": "Processing Plant", "type": "factory", "inventory_days": 18.0},
            {"id": "node_dest", "name": "Fulfillment Terminal", "type": "customer"}
        ],
        "edges": [
            {"id": "edge_1", "source": "node_origin", "target": "node_gateway", "distance_km": 800.0, "transit_time_days": 3.0, "transport_cost_per_unit": 40.0},
            {"id": "edge_2", "source": "node_gateway", "target": "node_hub", "distance_km": 350.0, "transit_time_days": 1.5, "transport_cost_per_unit": 20.0},
            {"id": "edge_3", "source": "node_hub", "target": "node_dest", "distance_km": 150.0, "transit_time_days": 0.8, "transport_cost_per_unit": 12.0}
        ]
    }

    sample_generic_shock = {
        "node_id": "node_gateway",
        "attribute": "congestion",
        "current_value": 15,
        "new_value": 85,
        "duration_days": 21.0
    }

    sample_macro = {
        "usd_inr": 84.2,
        "crude_price": 86.5,
        "freight_index": 140.0
    }

    print("Executing predict_supply_chain_impact() with generic attribute shock...")
    result = predict_supply_chain_impact(sample_network, sample_generic_shock, sample_macro)
    print("\nInference Output:")
    print(json.dumps({k: v for k, v in result.items() if k != "features"}, indent=2))

