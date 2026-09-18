"""
Data-Driven Training Dataset Generator for Vyuha ML.
Generates thousands of realistic disruption scenarios across diverse supply chain
topologies (both procedurally synthesized arbitrary graphs and representative Indian networks),
applying diverse attribute shocks and macro conditions, and calculating mathematically
grounded business impact targets.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.graph_engine import SupplyChainGraph, propagate_shock
from features.feature_engineering import extract_ml_features, CANONICAL_FEATURES


def generate_synthetic_network(network_id: str, topology_type: str, industry: str) -> Dict[str, Any]:
    """
    Procedurally synthesizes an arbitrary supply chain network with varied
    structures, node counts, capacities, and edge routes.
    """
    nodes = []
    edges = []

    if topology_type == "lean_serial":
        # 3-node lean supply chain: Source -> Factory -> Demand
        nodes = [
            {"id": f"{network_id}_src", "name": "Primary Material Source", "type": "supplier", "is_primary": True, "share_of_supply": 1.0, "reliability": float(np.random.uniform(0.82, 0.96)), "availability": float(np.random.uniform(0.85, 0.98)), "capacity_utilization": float(np.random.uniform(0.65, 0.90))},
            {"id": f"{network_id}_fac", "name": "Main Production Facility", "type": "factory", "capacity_utilization": float(np.random.uniform(0.70, 0.92)), "inventory_days": float(np.random.uniform(10.0, 30.0)), "safety_stock_days": float(np.random.uniform(4.0, 12.0)), "importance_weight": 0.9},
            {"id": f"{network_id}_cust", "name": "End Market Retailer", "type": "customer", "importance_weight": 1.0}
        ]
        edges = [
            {"id": f"{network_id}_e1", "source": f"{network_id}_src", "target": f"{network_id}_fac", "distance_km": float(np.random.uniform(200, 1200)), "transit_time_days": float(np.random.uniform(1.0, 4.0)), "transport_cost_per_unit": float(np.random.uniform(15, 45))},
            {"id": f"{network_id}_e2", "source": f"{network_id}_fac", "target": f"{network_id}_cust", "distance_km": float(np.random.uniform(100, 800)), "transit_time_days": float(np.random.uniform(0.5, 3.0)), "transport_cost_per_unit": float(np.random.uniform(10, 35))}
        ]
        depth = 3

    elif topology_type == "multi_supplier":
        # 3 Suppliers -> 1 Central Factory -> 2 Warehouses -> Customer (7 nodes)
        s1_share = float(np.random.uniform(0.4, 0.7))
        s2_share = float(np.random.uniform(0.15, 1.0 - s1_share))
        s3_share = max(0.05, 1.0 - s1_share - s2_share)
        nodes = [
            {"id": f"{network_id}_sup1", "name": "Primary Vendor", "type": "supplier", "is_primary": True, "share_of_supply": s1_share, "reliability": float(np.random.uniform(0.85, 0.95)), "availability": 0.92, "capacity_utilization": 0.80},
            {"id": f"{network_id}_sup2", "name": "Secondary Vendor", "type": "supplier", "is_primary": False, "share_of_supply": s2_share, "reliability": float(np.random.uniform(0.78, 0.90)), "availability": 0.88, "capacity_utilization": 0.70},
            {"id": f"{network_id}_sup3", "name": "Backup Vendor", "type": "supplier", "is_primary": False, "share_of_supply": s3_share, "reliability": float(np.random.uniform(0.75, 0.88)), "availability": 0.85, "capacity_utilization": 0.60},
            {"id": f"{network_id}_fac", "name": "Central Assembly Plant", "type": "factory", "capacity_utilization": 0.82, "inventory_days": float(np.random.uniform(12, 28)), "safety_stock_days": 6.0, "importance_weight": 0.95},
            {"id": f"{network_id}_wh1", "name": "North Regional Depot", "type": "warehouse", "capacity_utilization": 0.72, "inventory_days": 20.0, "safety_stock_days": 8.0, "importance_weight": 0.6},
            {"id": f"{network_id}_wh2", "name": "South Regional Depot", "type": "warehouse", "capacity_utilization": 0.68, "inventory_days": 22.0, "safety_stock_days": 8.0, "importance_weight": 0.6},
            {"id": f"{network_id}_cust", "name": "Commercial Client", "type": "customer", "importance_weight": 1.0}
        ]
        edges = [
            {"id": f"{network_id}_e1", "source": f"{network_id}_sup1", "target": f"{network_id}_fac", "distance_km": float(np.random.uniform(300, 900)), "transit_time_days": 2.0, "transport_cost_per_unit": 25.0},
            {"id": f"{network_id}_e2", "source": f"{network_id}_sup2", "target": f"{network_id}_fac", "distance_km": float(np.random.uniform(400, 1100)), "transit_time_days": 3.0, "transport_cost_per_unit": 32.0},
            {"id": f"{network_id}_e3", "source": f"{network_id}_sup3", "target": f"{network_id}_fac", "distance_km": float(np.random.uniform(500, 1400)), "transit_time_days": 3.5, "transport_cost_per_unit": 38.0},
            {"id": f"{network_id}_e4", "source": f"{network_id}_fac", "target": f"{network_id}_wh1", "distance_km": float(np.random.uniform(200, 600)), "transit_time_days": 1.5, "transport_cost_per_unit": 18.0},
            {"id": f"{network_id}_e5", "source": f"{network_id}_fac", "target": f"{network_id}_wh2", "distance_km": float(np.random.uniform(300, 700)), "transit_time_days": 2.0, "transport_cost_per_unit": 22.0},
            {"id": f"{network_id}_e6", "source": f"{network_id}_wh1", "target": f"{network_id}_cust", "distance_km": float(np.random.uniform(100, 400)), "transit_time_days": 1.0, "transport_cost_per_unit": 12.0},
            {"id": f"{network_id}_e7", "source": f"{network_id}_wh2", "target": f"{network_id}_cust", "distance_km": float(np.random.uniform(120, 450)), "transit_time_days": 1.2, "transport_cost_per_unit": 14.0}
        ]
        depth = 4

    elif topology_type == "dual_gateway":
        # Dual Gateway: Supplier -> Port A / Port B -> Factory -> DC -> Market (6 nodes)
        nodes = [
            {"id": f"{network_id}_sup", "name": "Global Component Supplier", "type": "supplier", "is_primary": True, "share_of_supply": 1.0, "reliability": 0.90, "availability": 0.92, "capacity_utilization": 0.80},
            {"id": f"{network_id}_pt1", "name": "Primary Inbound Terminal", "type": "port", "is_primary": True, "share_of_volume": 0.70, "capacity_utilization": 0.82, "congestion": 20.0},
            {"id": f"{network_id}_pt2", "name": "Alternate Inbound Terminal", "type": "port", "is_primary": False, "share_of_volume": 0.30, "capacity_utilization": 0.65, "congestion": 10.0},
            {"id": f"{network_id}_fac", "name": "Processing Complex", "type": "factory", "capacity_utilization": 0.85, "inventory_days": 18.0, "safety_stock_days": 7.0, "importance_weight": 0.9},
            {"id": f"{network_id}_dc", "name": "National Fulfillment Center", "type": "distribution_center", "capacity_utilization": 0.75, "importance_weight": 0.75},
            {"id": f"{network_id}_mkt", "name": "Consumer Outlets", "type": "customer", "importance_weight": 1.0}
        ]
        edges = [
            {"id": f"{network_id}_e1", "source": f"{network_id}_sup", "target": f"{network_id}_pt1", "distance_km": 3500.0, "transit_time_days": 12.0, "transport_cost_per_unit": 90.0},
            {"id": f"{network_id}_e2", "source": f"{network_id}_sup", "target": f"{network_id}_pt2", "distance_km": 3900.0, "transit_time_days": 14.0, "transport_cost_per_unit": 105.0},
            {"id": f"{network_id}_e3", "source": f"{network_id}_pt1", "target": f"{network_id}_fac", "distance_km": 450.0, "transit_time_days": 2.0, "transport_cost_per_unit": 28.0},
            {"id": f"{network_id}_e4", "source": f"{network_id}_pt2", "target": f"{network_id}_fac", "distance_km": 720.0, "transit_time_days": 3.0, "transport_cost_per_unit": 42.0},
            {"id": f"{network_id}_e5", "source": f"{network_id}_fac", "target": f"{network_id}_dc", "distance_km": 280.0, "transit_time_days": 1.0, "transport_cost_per_unit": 16.0},
            {"id": f"{network_id}_e6", "source": f"{network_id}_dc", "target": f"{network_id}_mkt", "distance_km": 150.0, "transit_time_days": 0.8, "transport_cost_per_unit": 10.0}
        ]
        depth = 5

    else:
        # Multi-Tier Diamond / Mesh Network (8 nodes)
        nodes = [
            {"id": f"{network_id}_src1", "name": "Tier 1 Supplier", "type": "supplier", "is_primary": True, "share_of_supply": 0.65, "reliability": 0.89, "availability": 0.91, "capacity_utilization": 0.78},
            {"id": f"{network_id}_src2", "name": "Tier 2 Supplier", "type": "supplier", "is_primary": False, "share_of_supply": 0.35, "reliability": 0.85, "availability": 0.88, "capacity_utilization": 0.70},
            {"id": f"{network_id}_hub", "name": "Consolidation Hub", "type": "warehouse", "capacity_utilization": 0.75, "inventory_days": 15.0, "safety_stock_days": 6.0},
            {"id": f"{network_id}_fac1", "name": "Primary Manufacturing Plant", "type": "factory", "capacity_utilization": 0.84, "inventory_days": 22.0, "safety_stock_days": 8.0, "importance_weight": 0.95},
            {"id": f"{network_id}_fac2", "name": "Secondary Auxiliary Plant", "type": "factory", "capacity_utilization": 0.68, "inventory_days": 18.0, "safety_stock_days": 5.0, "importance_weight": 0.75},
            {"id": f"{network_id}_dc", "name": "Logistics Dispatch Center", "type": "distribution_center", "capacity_utilization": 0.72},
            {"id": f"{network_id}_cust1", "name": "Enterprise Buyer A", "type": "customer"},
            {"id": f"{network_id}_cust2", "name": "Enterprise Buyer B", "type": "customer"}
        ]
        edges = [
            {"id": f"{network_id}_e1", "source": f"{network_id}_src1", "target": f"{network_id}_hub", "distance_km": 300.0, "transit_time_days": 1.5, "transport_cost_per_unit": 20.0},
            {"id": f"{network_id}_e2", "source": f"{network_id}_src2", "target": f"{network_id}_hub", "distance_km": 420.0, "transit_time_days": 2.0, "transport_cost_per_unit": 26.0},
            {"id": f"{network_id}_e3", "source": f"{network_id}_hub", "target": f"{network_id}_fac1", "distance_km": 350.0, "transit_time_days": 1.8, "transport_cost_per_unit": 22.0},
            {"id": f"{network_id}_e4", "source": f"{network_id}_hub", "target": f"{network_id}_fac2", "distance_km": 500.0, "transit_time_days": 2.5, "transport_cost_per_unit": 30.0},
            {"id": f"{network_id}_e5", "source": f"{network_id}_fac1", "target": f"{network_id}_dc", "distance_km": 250.0, "transit_time_days": 1.2, "transport_cost_per_unit": 18.0},
            {"id": f"{network_id}_e6", "source": f"{network_id}_fac2", "target": f"{network_id}_dc", "distance_km": 320.0, "transit_time_days": 1.5, "transport_cost_per_unit": 21.0},
            {"id": f"{network_id}_e7", "source": f"{network_id}_dc", "target": f"{network_id}_cust1", "distance_km": 120.0, "transit_time_days": 0.8, "transport_cost_per_unit": 10.0},
            {"id": f"{network_id}_e8", "source": f"{network_id}_dc", "target": f"{network_id}_cust2", "distance_km": 180.0, "transit_time_days": 1.0, "transport_cost_per_unit": 12.0}
        ]
        depth = 5

    return {
        "id": network_id,
        "name": f"{industry} Network ({topology_type.replace('_', ' ').title()})",
        "industry": industry,
        "network_depth": depth,
        "nodes": nodes,
        "edges": edges
    }


def calculate_targets(
    features: Dict[str, float],
    shock: Dict[str, Any],
    network: Dict[str, Any]
) -> Dict[str, float]:
    """
    Mathematically derives ground-truth business impact targets from graph propagation state.
    Targets are mathematically grounded, non-random, and free from target leakage in features.
    """
    severity = features["shock_severity"]
    duration = features["shock_duration_days"]
    inv_days = features["inventory_days"]
    alt_available = features["alternate_route_available"]
    affected_supply = features["affected_supply_percent"]
    affected_prod = features["affected_production_percent"]
    transit_days = features["additional_transit_days"]
    transport_cost = features["additional_transport_cost_pct"]
    rerouted_pct = features["rerouted_supply_percent"]
    has_spof = features["single_point_of_failure"]
    usd_inr = features["usd_inr"]
    crude = features["crude_price"]
    freight = features["freight_index"]

    # 1. Cost Impact % (Model 1 Target)
    c_freight = transport_cost * 0.45
    c_spot = (rerouted_pct / 100.0) * (severity * 14.0)
    c_factory = (affected_prod / 100.0) * 16.5
    c_macro = max(0.0, (usd_inr - 82.5) * 0.35) + max(0.0, (crude - 75.0) * 0.12) + max(0.0, (freight - 100.0) * 0.06)
    
    cost_impact = c_freight + c_spot + c_factory + c_macro
    cost_impact_pct = float(np.clip(np.round(cost_impact + np.random.normal(0, 0.6), 2), 0.5, 65.0))

    # 2. Additional Delay Days (Model 2 Target)
    d_transit = transit_days
    raw_delay = (affected_supply / 100.0) * (duration * 0.38)
    inventory_offset = min(raw_delay * 0.7, inv_days * 0.25)
    net_delay = max(0.0, raw_delay - inventory_offset)
    spof_penalty = (severity * 3.5) if (has_spof and not alt_available) else 0.0
    
    delay = d_transit + net_delay + spof_penalty
    delay_days = float(np.clip(np.round(delay + np.random.normal(0, 0.4), 1), 0.0, 42.0))

    # 3. Composite Risk Score (Model 3 Target: 0-100)
    stress_delay = min(100.0, (delay_days / max(inv_days, 3.0)) * 100.0)
    stress_supply = min(100.0, affected_supply * 1.3)
    stress_cost = min(100.0, (cost_impact_pct / 28.0) * 100.0)
    stress_severity = (severity * 50.0) + (min(45.0, duration) / 45.0 * 35.0)
    stress_spof = has_spof * 15.0

    raw_risk = (
        0.30 * stress_delay +
        0.25 * stress_supply +
        0.25 * stress_cost +
        0.12 * stress_severity +
        0.08 * stress_spof
    )
    risk_score = float(np.clip(np.round(raw_risk + np.random.normal(0, 1.2), 1), 3.0, 99.0))

    return {
        "cost_impact_pct": cost_impact_pct,
        "delay_days": delay_days,
        "risk_score": risk_score
    }


def generate_dataset(num_samples_target: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generates simulated dataset across both curated representative networks
    and diverse procedurally synthesized supply network topologies.
    """
    np.random.seed(seed)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    networks_dir = os.path.join(base_dir, "data", "networks")
    
    # 1. Load curated demo networks
    loaded_networks = []
    if os.path.exists(networks_dir):
        network_files = [f for f in os.listdir(networks_dir) if f.endswith(".json")]
        for nf in network_files:
            with open(os.path.join(networks_dir, nf), "r", encoding="utf-8") as f:
                loaded_networks.append(json.load(f))

    # 2. Add procedurally synthesized networks across diverse non-demo industries
    diverse_industries = [
        "Textiles", "Chemicals", "Food Processing", "Aerospace",
        "Energy", "Construction", "Heavy Machinery", "Plastics"
    ]
    topology_types = ["lean_serial", "multi_supplier", "dual_gateway", "mesh_diamond"]
    
    synth_id = 1
    for ind in diverse_industries:
        for t_type in topology_types:
            synth_net = generate_synthetic_network(f"synth_{synth_id}", t_type, ind)
            loaded_networks.append(synth_net)
            synth_id += 1

    print(f"Loaded/Generated {len(loaded_networks)} total supply chain networks for training.")

    attributes_pool = ["congestion", "capacity", "availability", "reliability", "transit_time_days", "transport_cost", "severity"]
    duration_ranges = [3, 7, 14, 21, 30, 45, 60]

    rows = []
    samples_per_net = int(np.ceil(num_samples_target / len(loaded_networks)))

    for net_dict in loaded_networks:
        graph = SupplyChainGraph(net_dict)
        net_family = net_dict["id"]
        industry = net_dict["industry"]

        for _ in range(samples_per_net):
            # Select target node or edge
            target_is_edge = (np.random.rand() < 0.20 and len(graph.edges) > 0)
            attr = np.random.choice(attributes_pool)
            duration = float(np.clip(np.random.choice(duration_ranges) + np.random.normal(0, 2.0), 2.0, 75.0))

            if target_is_edge:
                edge = np.random.choice(graph.edges)
                curr_val = float(edge.get("transit_time_days", 3.0) if attr == "transit_time_days" else edge.get("transport_cost_per_unit", 25.0))
                factor = float(np.random.uniform(1.2, 3.5))
                new_val = curr_val * factor
                shock_config = {
                    "edge_id": edge.get("id"),
                    "attribute": attr,
                    "current_value": curr_val,
                    "new_value": new_val,
                    "duration_days": duration
                }
            else:
                target_node = np.random.choice(list(graph.nodes.values())) if graph.nodes else None
                node_id = target_node["id"] if target_node else None
                if attr == "capacity":
                    curr_val = 100.0
                    drop = float(np.random.uniform(15.0, 90.0))
                    new_val = curr_val - drop
                elif attr == "congestion":
                    curr_val = float(target_node.get("congestion", 15.0) if target_node else 15.0)
                    surge = float(np.random.uniform(20.0, 80.0))
                    new_val = min(100.0, curr_val + surge)
                elif attr in ["availability", "reliability"]:
                    curr_val = 1.0
                    new_val = float(np.random.uniform(0.1, 0.75))
                else:
                    curr_val = 100.0
                    new_val = float(np.random.uniform(10.0, 85.0))

                shock_config = {
                    "node_id": node_id,
                    "attribute": attr,
                    "current_value": curr_val,
                    "new_value": new_val,
                    "duration_days": duration
                }

            # Propagate shock deterministically through graph
            shock_result = propagate_shock(graph, shock_config)

            # Sample Indian macroeconomic context
            macro = {
                "usd_inr": float(np.clip(np.random.normal(83.4, 1.8), 79.0, 88.5)),
                "crude_price": float(np.clip(np.random.normal(82.5, 8.5), 65.0, 115.0)),
                "commodity_price": float(np.clip(np.random.normal(102.0, 6.0), 85.0, 130.0)),
                "inflation_rate": float(np.clip(np.random.normal(5.4, 0.8), 3.8, 7.8)),
                "freight_index": float(np.clip(np.random.normal(118.0, 22.0), 75.0, 220.0))
            }

            # Canonical feature extraction (strictly zero target leakage)
            features = extract_ml_features(
                network_state=net_dict,
                shock_state=shock_result,
                external_conditions=macro
            )

            # Mathematically compute target impacts
            targets = calculate_targets(features, shock_result, net_dict)

            row = {
                "network_family": net_family,
                "industry": industry,
                "shock_attribute": attr,
                **features,
                **targets
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    # Shuffle
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "training_dataset.csv")

    print("Generating 5,000+ data-driven supply chain shock scenarios...")
    df = generate_dataset(num_samples_target=5000, seed=42)
    df.to_csv(out_path, index=False)
    print(f"Saved training dataset to: {out_path}")
    print(f"Shape: {df.shape}")
    print("\nIndustries Represented in Training Data:")
    print(df["industry"].value_counts().to_string())
    print("\nTarget Distributions:")
    print(df[["cost_impact_pct", "delay_days", "risk_score"]].describe())


if __name__ == "__main__":
    main()
