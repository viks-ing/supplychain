"""
Generic Extensibility Test Suite for Vyuha ML.
Verifies Section 13 of the PRD:
Tests the entire pipeline (Graph Simulation -> Propagation -> Feature Engineering -> XGBoost -> SHAP)
on completely novel, unseen supply chain networks with arbitrary node names, structures,
and generic attribute shocks. Zero entity-specific or industry-specific if statements required.
"""

import os
import sys
import json
import unittest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference.predict import predict_supply_chain_impact
from features.graph_engine import SupplyChainGraph, propagate_shock
from features.feature_engineering import extract_ml_features, CANONICAL_FEATURES


class TestGenericExtensibility(unittest.TestCase):

    def setUp(self):
        self.macro_conditions = {
            "usd_inr": 83.8,
            "crude_price": 84.5,
            "commodity_price": 104.0,
            "inflation_rate": 5.1,
            "freight_index": 122.0
        }

    def test_novel_four_tier_network_with_congestion_shock(self):
        """
        Test 1: Completely novel 4-tier network:
        Source_Alpha -> Facility_Beta -> Depot_Gamma_1 / Depot_Gamma_2 -> Demand_Omega
        Shocks Facility_Beta with generic congestion attribute increase.
        """
        novel_network = {
            "id": "novel_net_001",
            "name": "Bio-Chemicals Synthesis Network",
            "industry": "Bio-Chemicals",
            "network_depth": 4,
            "nodes": [
                {
                    "id": "source_alpha",
                    "name": "Feedstock Extraction Well",
                    "type": "supplier",
                    "is_primary": True,
                    "share_of_supply": 1.0,
                    "reliability": 0.92,
                    "availability": 0.95,
                    "capacity_utilization": 0.80
                },
                {
                    "id": "facility_beta",
                    "name": "Catalytic Refining Plant",
                    "type": "factory",
                    "capacity_utilization": 0.88,
                    "inventory_days": 18.0,
                    "safety_stock_days": 6.0,
                    "congestion": 15.0,
                    "importance_weight": 0.95
                },
                {
                    "id": "depot_gamma_1",
                    "name": "Solvent Silo Hub A",
                    "type": "warehouse",
                    "capacity_utilization": 0.70,
                    "inventory_days": 24.0,
                    "safety_stock_days": 8.0,
                    "importance_weight": 0.6
                },
                {
                    "id": "depot_gamma_2",
                    "name": "Solvent Silo Hub B",
                    "type": "warehouse",
                    "capacity_utilization": 0.65,
                    "inventory_days": 20.0,
                    "safety_stock_days": 7.0,
                    "importance_weight": 0.6
                },
                {
                    "id": "demand_omega",
                    "name": "Industrial Pharmaceutical Client",
                    "type": "customer",
                    "importance_weight": 1.0
                }
            ],
            "edges": [
                {"id": "link_ab", "source": "source_alpha", "target": "facility_beta", "distance_km": 420.0, "transit_time_days": 2.0, "transport_cost_per_unit": 35.0},
                {"id": "link_bg1", "source": "facility_beta", "target": "depot_gamma_1", "distance_km": 180.0, "transit_time_days": 1.0, "transport_cost_per_unit": 18.0},
                {"id": "link_bg2", "source": "facility_beta", "target": "depot_gamma_2", "distance_km": 260.0, "transit_time_days": 1.5, "transport_cost_per_unit": 22.0},
                {"id": "link_g1o", "source": "depot_gamma_1", "target": "demand_omega", "distance_km": 110.0, "transit_time_days": 0.8, "transport_cost_per_unit": 12.0},
                {"id": "link_g2o", "source": "depot_gamma_2", "target": "demand_omega", "distance_km": 140.0, "transit_time_days": 0.9, "transport_cost_per_unit": 14.0}
            ]
        }

        generic_shock = {
            "node_id": "facility_beta",
            "attribute": "congestion",
            "current_value": 15,
            "new_value": 85,
            "duration_days": 21
        }

        print("\n" + "="*70)
        print("TEST 1: Novel 4-Tier Network with Generic Congestion Attribute Shock")
        print("="*70)
        result = predict_supply_chain_impact(novel_network, generic_shock, self.macro_conditions)

        print(f"Cost Impact: {result['cost_impact_pct']}%")
        print(f"Delay Days:  {result['delay_days']} days")
        print(f"Risk Score:  {result['risk_score']} / 100 ({result['risk_tier']})")
        print(f"Top Risk Drivers:")
        for d in result["top_risk_drivers"][:3]:
            print(f"  * {d['explanation']}")

        # Assertions
        self.assertIn("cost_impact_pct", result)
        self.assertIn("delay_days", result)
        self.assertIn("risk_score", result)
        self.assertIn("risk_tier", result)
        self.assertGreater(result["cost_impact_pct"], 0.0)
        self.assertGreater(result["delay_days"], 0.0)
        self.assertGreater(result["risk_score"], 0.0)
        self.assertEqual(len(result["features"]), len(CANONICAL_FEATURES))

    def test_novel_three_node_lean_network_with_capacity_shock(self):
        """
        Test 2: Minimal 3-node lean network (zero ports, zero warehouses):
        Material_Source_X -> Fabricator_Y -> Final_Customer_Z
        Shocks Material_Source_X with generic capacity drop.
        """
        lean_network = {
            "id": "lean_net_002",
            "name": "Precision Optics Lean Supply Chain",
            "industry": "Optics & Photonics",
            "network_depth": 3,
            "nodes": [
                {
                    "id": "mat_source_x",
                    "name": "Synthetic Quartz Smelter",
                    "type": "supplier",
                    "is_primary": True,
                    "share_of_supply": 1.0,
                    "reliability": 0.94,
                    "availability": 0.96,
                    "capacity_utilization": 0.85
                },
                {
                    "id": "fabricator_y",
                    "name": "Prism Polishing Facility",
                    "type": "factory",
                    "capacity_utilization": 0.82,
                    "inventory_days": 12.0,
                    "safety_stock_days": 4.0,
                    "importance_weight": 0.90
                },
                {
                    "id": "customer_z",
                    "name": "Satellite Payload Integrator",
                    "type": "customer",
                    "importance_weight": 1.0
                }
            ],
            "edges": [
                {"id": "link_xy", "source": "mat_source_x", "target": "fabricator_y", "distance_km": 650.0, "transit_time_days": 3.0, "transport_cost_per_unit": 45.0},
                {"id": "link_yz", "source": "fabricator_y", "target": "customer_z", "distance_km": 300.0, "transit_time_days": 1.5, "transport_cost_per_unit": 25.0}
            ]
        }

        generic_capacity_shock = {
            "node_id": "mat_source_x",
            "attribute": "capacity",
            "current_value": 100,
            "new_value": 25,
            "duration_days": 25
        }

        print("\n" + "="*70)
        print("TEST 2: Minimal 3-Node Lean Network with Generic Capacity Shock")
        print("="*70)
        result = predict_supply_chain_impact(lean_network, generic_capacity_shock, self.macro_conditions)

        print(f"Cost Impact: {result['cost_impact_pct']}%")
        print(f"Delay Days:  {result['delay_days']} days")
        print(f"Risk Score:  {result['risk_score']} / 100 ({result['risk_tier']})")
        print(f"Port Dependency (dynamically computed): {result['features']['port_dependency']}%")
        print(f"Single Point of Failure (dynamically detected): {result['features']['single_point_of_failure']}")

        # Assertions
        self.assertEqual(result["features"]["port_dependency"], 0.0)  # Correctly identifies 0 ports
        self.assertEqual(result["features"]["single_point_of_failure"], 1.0)  # Correctly detects SPOF
        self.assertGreater(result["cost_impact_pct"], 0.0)
        self.assertGreater(result["delay_days"], 0.0)
        self.assertGreater(result["risk_score"], 0.0)

    def test_corridor_edge_transit_shock(self):
        """
        Test 3: Shock applied directly to an Edge (Transit Corridor):
        Transit time increases from 2.0 to 9.0 days due to highway closure.
        """
        multi_route_net = {
            "id": "multi_route_003",
            "name": "Textiles Export Network",
            "industry": "Textiles",
            "nodes": [
                {"id": "spinning_mill", "name": "Coimbatore Spinning Mill", "type": "supplier", "is_primary": True, "share_of_supply": 1.0},
                {"id": "garment_unit", "name": "Tirupur Apparel Unit", "type": "factory", "inventory_days": 14.0},
                {"id": "air_cargo", "name": "Bengaluru Air Gateway", "type": "port", "is_primary": True, "share_of_volume": 0.6},
                {"id": "sea_port", "name": "Tuticorin Container Port", "type": "port", "is_primary": False, "share_of_volume": 0.4},
                {"id": "retail_client", "name": "Global Brand Hub", "type": "customer"}
            ],
            "edges": [
                {"id": "route_mill_unit", "source": "spinning_mill", "target": "garment_unit", "distance_km": 50.0, "transit_time_days": 0.5, "transport_cost_per_unit": 8.0},
                {"id": "route_unit_air", "source": "garment_unit", "target": "air_cargo", "distance_km": 320.0, "transit_time_days": 2.0, "transport_cost_per_unit": 35.0},
                {"id": "route_unit_sea", "source": "garment_unit", "target": "sea_port", "distance_km": 280.0, "transit_time_days": 1.8, "transport_cost_per_unit": 22.0},
                {"id": "route_air_client", "source": "air_cargo", "target": "retail_client", "distance_km": 4000.0, "transit_time_days": 1.0, "transport_cost_per_unit": 120.0},
                {"id": "route_sea_client", "source": "sea_port", "target": "retail_client", "distance_km": 6000.0, "transit_time_days": 16.0, "transport_cost_per_unit": 45.0}
            ]
        }

        edge_shock = {
            "edge_id": "route_unit_air",
            "attribute": "transit_time_days",
            "current_value": 2.0,
            "new_value": 9.0,
            "duration_days": 15
        }

        print("\n" + "="*70)
        print("TEST 3: Direct Edge (Corridor) Transit Time Shock")
        print("="*70)
        result = predict_supply_chain_impact(multi_route_net, edge_shock, self.macro_conditions)

        print(f"Cost Impact: {result['cost_impact_pct']}%")
        print(f"Delay Days:  {result['delay_days']} days")
        print(f"Risk Score:  {result['risk_score']} / 100 ({result['risk_tier']})")
        print(f"Alternate Routes Discovered: {len(result['alternative_routes'])}")

        self.assertGreater(result["cost_impact_pct"], 0.0)
        self.assertGreater(result["delay_days"], 0.0)


if __name__ == "__main__":
    unittest.main()
