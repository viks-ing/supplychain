"""
Multi-Network Pipeline Verification Test for Vyuha ML.
Tests the complete end-to-end flow on 3 required networks:
1. Electronics + Hyderabad
2. Automotive + Chennai
3. Pharmaceuticals + Hyderabad
"""

import os
import sys
import json
from typing import Dict, Any

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference.predict import predict_supply_chain_impact
from features.feature_engineering import CANONICAL_FEATURES


def test_three_required_networks():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    networks_dir = os.path.join(base_dir, "data", "networks")
    schema_path = os.path.join(base_dir, "models", "feature_schema.json")

    with open(schema_path, "r", encoding="utf-8") as f:
        stored_schema = json.load(f)["features"]

    # Verify canonical schema consistency
    assert stored_schema == CANONICAL_FEATURES, "Mismatch between stored schema and CANONICAL_FEATURES!"
    print(f"Verified feature schema integrity ({len(CANONICAL_FEATURES)} features).")

    test_cases = [
        {
            "test_id": "TEST-1",
            "name": "Electronics + Hyderabad",
            "network_file": "electronics_hyderabad.json",
            "shock": {
                "shock_type": "port",
                "target_id": "port_chennai",
                "shock_severity": 0.70,
                "shock_duration_days": 21.0
            },
            "macro": {
                "usd_inr": 84.5,
                "crude_price": 88.0,
                "commodity_price": 105.0,
                "inflation_rate": 5.6,
                "freight_index": 145.0
            }
        },
        {
            "test_id": "TEST-2",
            "name": "Automotive + Chennai",
            "network_file": "automotive_chennai.json",
            "shock": {
                "shock_type": "supplier",
                "target_id": "sup_ecu_import",
                "shock_severity": 0.60,
                "shock_duration_days": 28.0
            },
            "macro": {
                "usd_inr": 83.8,
                "crude_price": 85.0,
                "commodity_price": 108.0,
                "inflation_rate": 5.2,
                "freight_index": 130.0
            }
        },
        {
            "test_id": "TEST-3",
            "name": "Pharmaceuticals + Hyderabad",
            "network_file": "pharma_hyderabad.json",
            "shock": {
                "shock_type": "supplier",
                "target_id": "sup_api_global",
                "shock_severity": 0.80,
                "shock_duration_days": 35.0
            },
            "macro": {
                "usd_inr": 85.2,
                "crude_price": 92.0,
                "commodity_price": 112.0,
                "inflation_rate": 6.1,
                "freight_index": 160.0
            }
        }
    ]

    print(f"\n{'='*70}")
    print("RUNNING END-TO-END PIPELINE VERIFICATION ACROSS 3 NETWORKS")
    print(f"{'='*70}")

    for tc in test_cases:
        net_path = os.path.join(networks_dir, tc["network_file"])
        with open(net_path, "r", encoding="utf-8") as f:
            network_data = json.load(f)

        print(f"\n>>> [{tc['test_id']}] Network: {tc['name']}")
        print(f"    Shock: {tc['shock']['shock_type'].upper()} on '{tc['shock']['target_id']}' (Severity: {tc['shock']['shock_severity']*100:.0f}%, Duration: {tc['shock']['shock_duration_days']} days)")

        # Run complete prediction
        result = predict_supply_chain_impact(
            network_state=network_data,
            shock_state=tc["shock"],
            external_conditions=tc["macro"]
        )

        cost = result["cost_impact_pct"]
        delay = result["delay_days"]
        risk = result["risk_score"]
        tier = result["risk_tier"]
        alt_routes = result["alternative_routes"]
        drivers = result["top_risk_drivers"]

        print(f"    Predictions:")
        print(f"      * Cost Impact:   +{cost:.1f}%")
        print(f"      * Addl Delay:    +{delay:.1f} days")
        print(f"      * Risk Score:    {risk:.1f}/100 [Tier: {tier}]")
        print(f"      * Alt Routes:    {len(alt_routes)} path(s) discovered")
        driver_str = ', '.join([f"{d['feature']} ({d['shap_impact']:+.2f})" for d in drivers[:2]])
        print(f"      * Top Drivers:   {driver_str}")

        # Assertions
        assert 0.0 <= risk <= 100.0, f"Risk score {risk} out of bounds 0-100!"
        assert cost >= 0.0, f"Cost impact {cost} is negative!"
        assert delay >= 0.0, f"Delay {delay} is negative!"
        assert len(result["features"]) == len(CANONICAL_FEATURES), "Feature count mismatch!"

    print(f"\n{'='*70}")
    print("ALL 3 REPRESENTATIVE NETWORKS PASSED PIPELINE VERIFICATION!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    test_three_required_networks()
