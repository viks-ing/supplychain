"""
What-If Scenario Simulation Service for Vyuha ML.
Runs side-by-side Normal vs Shock comparisons, inventory depletion models,
and generates contextual mitigation playbooks.
"""

from typing import Dict, Any
from services.prediction_service import PredictionService
from services.inventory_service import simulate_inventory_depletion
from services.mitigation_service import generate_mitigations

class SimulatorService:
    def __init__(self, prediction_service: PredictionService):
        self.predictor = prediction_service

    def simulate(
        self,
        company: Dict[str, Any],
        shock_parameters: Dict[str, Any],
        baseline_conditions: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        # 1. Normal Baseline Conditions
        if baseline_conditions is None:
            baseline_conditions = {
                "shock_type": "None (Normal Baseline)",
                "duration_days": 0.0,
                "severity_pct": 0.0,
                "crude_price_change_pct": 0.0,
                "usd_inr_change_pct": 0.0,
                "freight_index_change_pct": 0.0,
                "commodity_price_change_pct": 0.0,
                "inflation_rate_pct": 5.0,
                "port_congestion_pct": 20.0,
                "road_congestion_pct": 25.0,
                "transport_delay_days": 0.5
            }
            
        # 2. Predict Normal Baseline
        normal_result = self.predictor.predict(company, baseline_conditions)
        
        # 3. Predict Shocked Scenario
        shock_result = self.predictor.predict(company, shock_parameters)
        
        # 4. Compute Impact Deltas
        cost_delta_pct = round(max(0.0, shock_result["cost_impact_pct"] - normal_result["cost_impact_pct"]), 1)
        cost_delta_lakhs = round(max(0.0, shock_result["cost_impact_lakhs"] - normal_result["cost_impact_lakhs"]), 2)
        lead_time_delta_days = round(max(0.0, shock_result["lead_time_impact_days"] - normal_result["lead_time_impact_days"]), 1)
        risk_score_delta = round(max(0.0, shock_result["risk_score"] - normal_result["risk_score"]), 1)
        
        # 5. Inventory Depletion Simulation
        duration = float(shock_parameters.get("duration_days", 21.0))
        inventory_sim = simulate_inventory_depletion(
            company=company,
            lead_time_delay_days=shock_result["lead_time_impact_days"],
            disruption_duration_days=duration,
            horizon_days=max(45, int(duration + 15))
        )
        
        # 6. Mitigation Recommendations
        mitigations = generate_mitigations(
            company=company,
            shock=shock_parameters,
            vulnerabilities=shock_result["vulnerabilities"],
            risk_score=shock_result["risk_score"],
            stockout_day=inventory_sim["stockout_day"]
        )
        
        return {
            "normal_conditions": normal_result,
            "shock_conditions": shock_result,
            "impact_delta": {
                "additional_cost_pct": cost_delta_pct,
                "additional_cost_lakhs": cost_delta_lakhs,
                "additional_lead_time_days": lead_time_delta_days,
                "risk_score_increase": risk_score_delta,
                "shock_risk_level": shock_result["risk_level"]
            },
            "inventory_simulation": inventory_sim,
            "mitigation_recommendations": mitigations,
            "top_vulnerabilities": shock_result["vulnerabilities"]["top_risk_factors"]
        }
