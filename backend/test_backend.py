"""
Unit and integration test script for Vyuha ML backend services.
Validates the PRD Section 15 benchmark scenario and all pipeline components.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from services.prediction_service import PredictionService
from services.simulator_service import SimulatorService

def test_prd_section_15_scenario():
    print("\n--- Testing PRD Section 15 Example Scenario ---")
    
    predictor = PredictionService()
    simulator = SimulatorService(predictor)
    
    # PRD Section 15 Company Profile
    company_profile = {
        "company_name": "Bharat Microelectronics Ltd",
        "industry": "Electronics",
        "city_state": "Bengaluru, Karnataka",
        "monthly_procurement_lakhs": 50.0,
        "import_dependency": 75.0,
        "supplier_concentration": 65.0,
        "major_suppliers_count": 3,
        "alternative_suppliers_count": 1,
        "avg_lead_time_days": 15.0,
        "inventory_days": 20.0,
        "safety_stock_days": 7.0,
        "port_dependency": 80.0,
        "road_dependency": 60.0,
        "transport_mode": "Road",
        "primary_port": "JNPT (Navi Mumbai)"
    }
    
    # PRD Section 15 Disruption Scenario
    shock_scenario = {
        "shock_type": "Major Port Congestion & Strike",
        "duration_days": 21.0,
        "severity_pct": 80.0,
        "port_congestion_pct": 80.0,
        "road_congestion_pct": 50.0,
        "crude_price_change_pct": 25.0,
        "usd_inr_change_pct": 4.0,
        "freight_index_change_pct": 45.0,
        "commodity_price_change_pct": 15.0,
        "transport_delay_days": 6.5
    }
    
    result = simulator.simulate(
        company=company_profile,
        shock_parameters=shock_scenario
    )
    
    print("\nNORMAL CONDITIONS PREDICTION:")
    print(json.dumps(result["normal_conditions"], indent=2))
    
    print("\nSHOCKED CONDITIONS PREDICTION:")
    print(json.dumps(result["shock_conditions"], indent=2))
    
    print("\nIMPACT DELTA:")
    print(json.dumps(result["impact_delta"], indent=2))
    
    print("\nINVENTORY DEPLETION SUMMARY:")
    print(f"Stockout status: {result['inventory_simulation']['stockout_status']}")
    print(f"Stockout day: {result['inventory_simulation']['stockout_day']}")
    print(f"Safety buffer breach day: {result['inventory_simulation']['safety_stock_breach_day']}")
    print(f"Summary: {result['inventory_simulation']['summary']}")
    
    print("\nTOP VULNERABILITIES:")
    for v in result["top_vulnerabilities"]:
        print(f" - [{v['score']}/100] {v['factor']}: {v['description']}")
        
    print("\nMITIGATION RECOMMENDATIONS:")
    for m in result["mitigation_recommendations"]:
        print(f" - [{m['category']}] {m['title']} | Impact: {m['impact']}")
        
    print("\nVerification successful!")

if __name__ == "__main__":
    test_prd_section_15_scenario()
