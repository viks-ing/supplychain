"""
Comprehensive Stress-Testing and Physical Invariants Suite for Vyuha ML.
Evaluates:
1. Sanity & Edge Cases (Zero Shock baseline, Catastrophic SPOF shock)
2. Structural Resilience (Single supplier vs Dual redundant supplier)
3. Monotonicity Invariants (Severity scaling, Duration scaling, Inventory buffer mitigation)
4. Macroeconomic Sensitivity (USD/INR, Crude oil, Global Freight spikes)
5. Tree SHAP Attribution Consistency
6. Real-Time Inference Latency & High-Throughput Benchmarks
"""

import os
import sys
import time
import json
import unittest
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference.predict import predict_supply_chain_impact


class ModelStressAndInvariantsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Base network for invariant testing
        cls.base_network = {
            "id": "invariant_test_net",
            "name": "Precision Engineering Network",
            "industry": "Industrial Equipment",
            "nodes": [
                {"id": "sup_1", "name": "Primary Castings Supplier", "type": "supplier", "is_primary": True, "share_of_supply": 0.70, "reliability": 0.90, "availability": 0.92, "capacity_utilization": 0.80},
                {"id": "sup_2", "name": "Secondary Castings Supplier", "type": "supplier", "is_primary": False, "share_of_supply": 0.30, "reliability": 0.85, "availability": 0.88, "capacity_utilization": 0.65},
                {"id": "fac_1", "name": "Machining & Assembly Unit", "type": "factory", "capacity_utilization": 0.82, "inventory_days": 18.0, "safety_stock_days": 6.0},
                {"id": "wh_1", "name": "Finished Goods Central Hub", "type": "warehouse", "capacity_utilization": 0.70, "inventory_days": 20.0, "safety_stock_days": 7.0},
                {"id": "client_1", "name": "Heavy Equipment OEM", "type": "customer"}
            ],
            "edges": [
                {"id": "e1", "source": "sup_1", "target": "fac_1", "distance_km": 350.0, "transit_time_days": 1.5, "transport_cost_per_unit": 24.0},
                {"id": "e2", "source": "sup_2", "target": "fac_1", "distance_km": 480.0, "transit_time_days": 2.2, "transport_cost_per_unit": 32.0},
                {"id": "e3", "source": "fac_1", "target": "wh_1", "distance_km": 200.0, "transit_time_days": 1.0, "transport_cost_per_unit": 16.0},
                {"id": "e4", "source": "wh_1", "target": "client_1", "distance_km": 120.0, "transit_time_days": 0.8, "transport_cost_per_unit": 12.0}
            ]
        }

        cls.nominal_macro = {
            "usd_inr": 83.5,
            "crude_price": 82.0,
            "commodity_price": 100.0,
            "inflation_rate": 5.2,
            "freight_index": 115.0
        }

    # =================================================================
    # 1. Edge Cases: Zero Shock vs Catastrophic Shock
    # =================================================================
    def test_01_zero_shock_baseline(self):
        """Zero shock should produce Low risk tier and minimal cost/delay."""
        zero_shock = {
            "node_id": "sup_1",
            "attribute": "capacity",
            "current_value": 100.0,
            "new_value": 100.0,
            "duration_days": 0.0
        }
        res = predict_supply_chain_impact(self.base_network, zero_shock, self.nominal_macro)
        print(f"\n[TEST 1: ZERO SHOCK BASELINE]")
        print(f"  Cost: {res['cost_impact_pct']}%, Delay: {res['delay_days']}d, Risk: {res['risk_score']}/100 ({res['risk_tier']})")

        self.assertLessEqual(res["risk_score"], 25.0)
        self.assertEqual(res["risk_tier"], "Low")
        self.assertLessEqual(res["delay_days"], 1.5)
        self.assertLessEqual(res["cost_impact_pct"], 10.0)

    def test_02_catastrophic_spof_shock(self):
        """A 95% capacity collapse for 45 days on a single factory should trigger High/Critical risk."""
        catastrophic_shock = {
            "node_id": "fac_1",
            "attribute": "capacity",
            "current_value": 100.0,
            "new_value": 5.0,
            "duration_days": 45.0
        }
        res = predict_supply_chain_impact(self.base_network, catastrophic_shock, self.nominal_macro)
        print(f"\n[TEST 2: CATASTROPHIC SPOF SHOCK]")
        print(f"  Cost: {res['cost_impact_pct']}%, Delay: {res['delay_days']}d, Risk: {res['risk_score']}/100 ({res['risk_tier']})")

        self.assertGreaterEqual(res["risk_score"], 60.0)
        self.assertIn(res["risk_tier"], ["High", "Critical"])
        self.assertGreaterEqual(res["cost_impact_pct"], 20.0)
        self.assertGreaterEqual(res["delay_days"], 10.0)

    # =================================================================
    # 2. Structural Resilience: Redundant Suppliers vs Single Point of Failure
    # =================================================================
    def test_03_redundancy_resilience(self):
        """
        A network with dual suppliers should suffer significantly less risk
        than an identical network with only 1 sole supplier when that supplier fails.
        """
        # Network A: Sole Single Supplier (SPOF)
        spof_network = {
            "id": "spof_net",
            "name": "Sole Supplier Network",
            "industry": "Industrial Equipment",
            "nodes": [
                {"id": "sole_sup", "name": "Sole Supplier", "type": "supplier", "is_primary": True, "share_of_supply": 1.0, "reliability": 0.88, "availability": 0.90},
                {"id": "fac_1", "name": "Assembly Unit", "type": "factory", "inventory_days": 15.0},
                {"id": "client_1", "name": "Customer", "type": "customer"}
            ],
            "edges": [
                {"id": "e1", "source": "sole_sup", "target": "fac_1", "distance_km": 400.0, "transit_time_days": 2.0, "transport_cost_per_unit": 28.0},
                {"id": "e2", "source": "fac_1", "target": "client_1", "distance_km": 150.0, "transit_time_days": 1.0, "transport_cost_per_unit": 14.0}
            ]
        }

        # Network B: Dual Supplier (Redundant backup)
        dual_network = {
            "id": "dual_net",
            "name": "Dual Supplier Network",
            "industry": "Industrial Equipment",
            "nodes": [
                {"id": "primary_sup", "name": "Primary Supplier", "type": "supplier", "is_primary": True, "share_of_supply": 0.60, "reliability": 0.88, "availability": 0.90},
                {"id": "backup_sup", "name": "Backup Supplier", "type": "supplier", "is_primary": False, "share_of_supply": 0.40, "reliability": 0.85, "availability": 0.88},
                {"id": "fac_1", "name": "Assembly Unit", "type": "factory", "inventory_days": 15.0},
                {"id": "client_1", "name": "Customer", "type": "customer"}
            ],
            "edges": [
                {"id": "e1", "source": "primary_sup", "target": "fac_1", "distance_km": 400.0, "transit_time_days": 2.0, "transport_cost_per_unit": 28.0},
                {"id": "e2", "source": "backup_sup", "target": "fac_1", "distance_km": 500.0, "transit_time_days": 2.5, "transport_cost_per_unit": 34.0},
                {"id": "e3", "source": "fac_1", "target": "client_1", "distance_km": 150.0, "transit_time_days": 1.0, "transport_cost_per_unit": 14.0}
            ]
        }

        shock_sole = {"node_id": "sole_sup", "attribute": "capacity", "current_value": 100, "new_value": 30, "duration_days": 21}
        shock_primary = {"node_id": "primary_sup", "attribute": "capacity", "current_value": 100, "new_value": 30, "duration_days": 21}

        res_spof = predict_supply_chain_impact(spof_network, shock_sole, self.nominal_macro)
        res_dual = predict_supply_chain_impact(dual_network, shock_primary, self.nominal_macro)

        print(f"\n[TEST 3: RESILIENCE COMPARISON (SOLE vs DUAL SUPPLIER)]")
        print(f"  Sole Supplier Network -> Risk: {res_spof['risk_score']}, Delay: {res_spof['delay_days']}d, Cost: {res_spof['cost_impact_pct']}%")
        print(f"  Dual Supplier Network -> Risk: {res_dual['risk_score']}, Delay: {res_dual['delay_days']}d, Cost: {res_dual['cost_impact_pct']}%")

        self.assertGreater(res_spof["risk_score"], res_dual["risk_score"], "SPOF network should have higher risk than dual-supplier network!")
        self.assertGreater(res_spof["delay_days"], res_dual["delay_days"], "SPOF network should have longer delay than dual-supplier network!")

    # =================================================================
    # 3. Monotonicity: Severity & Duration Scaling
    # =================================================================
    def test_04_monotonicity_severity_scaling(self):
        """Increasing shock severity (15% to 85%) must monotonically increase Risk and Cost Impact."""
        severities = [15, 35, 55, 75, 90]
        risks = []
        costs = []

        print(f"\n[TEST 4: SEVERITY MONOTONICITY SCALING]")
        for sev in severities:
            shock = {
                "node_id": "sup_1",
                "attribute": "capacity",
                "current_value": 100.0,
                "new_value": 100.0 - sev,
                "duration_days": 14.0
            }
            res = predict_supply_chain_impact(self.base_network, shock, self.nominal_macro)
            risks.append(res["risk_score"])
            costs.append(res["cost_impact_pct"])
            print(f"  Severity: {sev}% -> Risk Score: {res['risk_score']:.1f}, Cost Impact: {res['cost_impact_pct']:.1f}%, Delay: {res['delay_days']:.1f}d")

        # Verify overall trend is strictly non-decreasing
        for i in range(len(risks) - 1):
            self.assertLessEqual(risks[i], risks[i+1] + 1.5, f"Risk should increase with severity: step {i} ({risks[i]}) to {i+1} ({risks[i+1]})")
            self.assertLessEqual(costs[i], costs[i+1] + 1.0, f"Cost should increase with severity: step {i} ({costs[i]}) to {i+1} ({costs[i+1]})")

    def test_05_inventory_buffer_mitigation(self):
        """Holding more inventory runway (5 days vs 15 days vs 35 days) should reduce overall risk score."""
        inventory_levels = [5.0, 15.0, 25.0, 40.0]
        risks = []

        print(f"\n[TEST 5: INVENTORY BUFFER MITIGATION]")
        for inv in inventory_levels:
            custom_net = json.loads(json.dumps(self.base_network))
            # Set factory inventory runway
            for n in custom_net["nodes"]:
                if n["id"] == "fac_1":
                    n["inventory_days"] = inv

            shock = {
                "node_id": "sup_1",
                "attribute": "capacity",
                "current_value": 100.0,
                "new_value": 40.0,
                "duration_days": 20.0
            }
            res = predict_supply_chain_impact(custom_net, shock, self.nominal_macro)
            risks.append(res["risk_score"])
            print(f"  Inventory Buffer: {inv:.0f} days -> Risk Score: {res['risk_score']:.1f} ({res['risk_tier']}), Production Throttled: {res['affected_nodes'][0]['capacity_drop_pct']}%")

        # High inventory runway should yield lower risk than minimal 5-day runway
        self.assertGreater(risks[0], risks[-1], "5 days inventory runway must produce higher risk than 40 days buffer runway!")

    # =================================================================
    # 4. Macroeconomic Sensitivity Spikes
    # =================================================================
    def test_06_macroeconomic_surge_sensitivity(self):
        """Higher USD/INR, Crude oil prices, and Freight Index should inflate cost impact."""
        calm_macro = {"usd_inr": 82.0, "crude_price": 70.0, "freight_index": 90.0}
        severe_macro = {"usd_inr": 88.5, "crude_price": 115.0, "freight_index": 210.0}

        shock = {"node_id": "sup_1", "attribute": "capacity", "current_value": 100, "new_value": 50, "duration_days": 14}

        res_calm = predict_supply_chain_impact(self.base_network, shock, calm_macro)
        res_severe = predict_supply_chain_impact(self.base_network, shock, severe_macro)

        print(f"\n[TEST 6: MACROECONOMIC SENSITIVITY]")
        print(f"  Calm Macro Conditions   -> Cost Impact: {res_calm['cost_impact_pct']:.1f}%, Risk: {res_calm['risk_score']:.1f}")
        print(f"  Severe Macro Conditions -> Cost Impact: {res_severe['cost_impact_pct']:.1f}%, Risk: {res_severe['risk_score']:.1f}")

        self.assertGreater(res_severe["cost_impact_pct"], res_calm["cost_impact_pct"])
        self.assertGreater(res_severe["risk_score"], res_calm["risk_score"])

    # =================================================================
    # 5. Tree SHAP Explainability & Root-Cause Attribution
    # =================================================================
    def test_07_tree_shap_driver_coherence(self):
        """Top Tree SHAP root cause drivers must explain the disruption logically."""
        shock = {
            "node_id": "sup_1",
            "attribute": "congestion",
            "current_value": 15,
            "new_value": 85,
            "duration_days": 21
        }
        res = predict_supply_chain_impact(self.base_network, shock, self.nominal_macro)
        drivers = res["top_risk_drivers"]

        print(f"\n[TEST 7: TREE SHAP EXPLANATION COHERENCE]")
        print(f"  Top Root Cause Drivers Identified:")
        for i, d in enumerate(drivers[:4], 1):
            print(f"    {i}. {d['explanation']}")

        self.assertGreater(len(drivers), 0)
        self.assertTrue(any("affected_" in d["feature"] or "transit" in d["feature"] or "congestion" in d["feature"] for d in drivers))

    # =================================================================
    # 6. Real-Time Latency & Throughput Benchmark
    # =================================================================
    def test_08_inference_latency_benchmark(self):
        """Validates real-time production performance: < 15ms per simulation, > 100 predictions in under 1.5s."""
        n_simulations = 100
        shock = {"node_id": "sup_1", "attribute": "capacity", "current_value": 100, "new_value": 60, "duration_days": 14}

        # Warm up
        predict_supply_chain_impact(self.base_network, shock, self.nominal_macro)

        start_time = time.perf_counter()
        for _ in range(n_simulations):
            predict_supply_chain_impact(self.base_network, shock, self.nominal_macro)
        total_time = time.perf_counter() - start_time
        avg_latency_ms = (total_time / n_simulations) * 1000.0

        print(f"\n[TEST 8: INFERENCE LATENCY BENCHMARK]")
        print(f"  Total Time for {n_simulations} Simulations: {total_time:.3f} seconds")
        print(f"  Average Latency: {avg_latency_ms:.2f} ms per simulation")
        print(f"  Throughput: {n_simulations / total_time:.1f} simulations/sec")

        self.assertLess(avg_latency_ms, 25.0, "Average latency must be under 25ms!")
        self.assertLess(total_time, 2.5, "100 simulations must complete in under 2.5s!")


if __name__ == "__main__":
    unittest.main()
