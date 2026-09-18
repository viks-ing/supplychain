"""
Canonical Feature Engineering Pipeline for Vyuha ML Graph-Driven Architecture.
Derives 46 canonical ML features dynamically from arbitrary supply chain graphs,
shock propagation state, and macroeconomic conditions without hardcoding.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

# Load canonical configuration files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, "config")


def _load_json_config(filename: str, fallback: Any) -> Any:
    path = os.path.join(CONFIG_DIR, filename)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return fallback
    return fallback


NODE_TYPES_CFG = _load_json_config("node_types.json", {})
FEATURE_SCHEMA_CFG = _load_json_config("feature_schema.json", {})
SHOCK_RULES_CFG = _load_json_config("shock_rules.json", {})

CANONICAL_FEATURES: List[str] = FEATURE_SCHEMA_CFG.get("canonical_features", [
    "supplier_count",
    "alternative_supplier_count",
    "port_count",
    "alternative_port_count",
    "factory_count",
    "warehouse_count",
    "distribution_center_count",
    "network_depth",
    "network_edge_count",
    "network_node_count",
    "supplier_concentration",
    "port_dependency",
    "single_point_of_failure",
    "average_route_distance_km",
    "critical_route_distance_km",
    "alternate_route_distance_km",
    "average_supplier_reliability",
    "minimum_supplier_reliability",
    "average_supplier_availability",
    "average_capacity_utilization",
    "maximum_capacity_utilization",
    "factory_capacity_utilization",
    "warehouse_capacity_utilization",
    "inventory_days",
    "safety_stock_days",
    "shock_type_encoded",
    "shock_severity",
    "shock_duration_days",
    "affected_node_importance",
    "affected_capacity_change_pct",
    "affected_availability_change_pct",
    "affected_congestion_change_pct",
    "affected_supply_percent",
    "affected_production_percent",
    "additional_transit_days",
    "additional_transport_cost_pct",
    "alternate_route_available",
    "alternate_route_count",
    "alternate_route_delay_days",
    "alternate_route_cost_increase_pct",
    "rerouted_supply_percent",
    "usd_inr",
    "crude_price",
    "commodity_price",
    "inflation_rate",
    "freight_index"
])

# Role mapping helper built from config
def _build_role_map() -> Dict[str, str]:
    role_map = {}
    roles = NODE_TYPES_CFG.get("node_roles", {})
    for role_name, role_data in roles.items():
        role_map[role_name.lower()] = role_name
        for alias in role_data.get("aliases", []):
            role_map[alias.lower()] = role_name
    return role_map

ROLE_MAP = _build_role_map()

def get_node_role(node: Dict[str, Any]) -> str:
    """Categorizes any arbitrary node into functional structural role."""
    raw_type = str(node.get("type", "general")).lower()
    if raw_type in ROLE_MAP:
        return ROLE_MAP[raw_type]
    # In-degree heuristic if type is arbitrary
    return "general"


def extract_ml_features(
    network_state: Dict[str, Any],
    shock_state: Dict[str, Any],
    external_conditions: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Canonical feature extraction function.
    Converts graph network state, simulated shock state, and macro conditions
    into a standardized feature dictionary with strictly defined units and names.
    Works dynamically for any network topology with zero hardcoding.
    """
    ext = external_conditions or {}

    # Extract nodes and edges
    raw_nodes = network_state.get("nodes", [])
    if isinstance(raw_nodes, dict):
        raw_nodes = list(raw_nodes.values())
    nodes = raw_nodes
    edges = network_state.get("edges", [])

    # Group nodes by functional role dynamically
    source_nodes = [n for n in nodes if get_node_role(n) == "source"]
    if not source_nodes and nodes:
        # Fallback for novel custom networks: nodes with in_degree == 0
        incoming_targets = {e.get("target") for e in edges}
        source_nodes = [n for n in nodes if n.get("id") not in incoming_targets]
        if not source_nodes:
            source_nodes = nodes[:1]

    port_nodes = [n for n in nodes if get_node_role(n) == "transit_gateway"]
    factory_nodes = [n for n in nodes if get_node_role(n) == "processing"]
    warehouse_nodes = [n for n in nodes if get_node_role(n) == "storage"]
    dc_nodes = [n for n in nodes if get_node_role(n) == "distribution"]

    supplier_count = len(source_nodes)
    alt_suppliers = [s for s in source_nodes if not s.get("is_primary", True)]
    alternative_supplier_count = len(alt_suppliers)

    port_count = len(port_nodes)
    alt_ports = [p for p in port_nodes if not p.get("is_primary", True)]
    alternative_port_count = len(alt_ports)

    factory_count = len(factory_nodes)
    warehouse_count = len(warehouse_nodes)
    distribution_center_count = len(dc_nodes)

    network_edge_count = len(edges)
    network_node_count = len(nodes)
    network_depth = float(network_state.get("network_depth", 4.0))

    # Supplier concentration: maximum single source share
    if source_nodes:
        supplier_concentration = max(
            float(s.get("share_of_supply", 1.0 / len(source_nodes))) * 100.0 for s in source_nodes
        )
    else:
        supplier_concentration = 100.0

    # Port dependency: maximum single port share (0.0 if no ports in network)
    if port_nodes:
        port_dependency = max(
            float(p.get("share_of_volume", 1.0 / len(port_nodes))) * 100.0 for p in port_nodes
        )
    else:
        port_dependency = 0.0

    # Single Point of Failure (SPOF)
    # Check if alternative supplier is absent or any critical choke point exists
    has_spof = 1.0 if (alternative_supplier_count == 0 or (port_count > 0 and alternative_port_count == 0)) else 0.0

    # Route distances
    distances = [float(e.get("distance_km", 0.0)) for e in edges if "distance_km" in e]
    average_route_distance_km = float(np.mean(distances)) if distances else 500.0
    critical_route_distance_km = float(shock_state.get("primary_route_distance_km", average_route_distance_km))
    alternate_route_distance_km = float(shock_state.get("alternate_route_distance_km", critical_route_distance_km * 1.25))

    # Operational & Capacity features
    supplier_reliabilities = [float(s.get("reliability", 0.88)) for s in source_nodes]
    average_supplier_reliability = float(np.mean(supplier_reliabilities)) if supplier_reliabilities else 0.85
    minimum_supplier_reliability = float(np.min(supplier_reliabilities)) if supplier_reliabilities else 0.75
    average_supplier_availability = float(np.mean([float(s.get("availability", 0.90)) for s in source_nodes])) if source_nodes else 0.90

    all_utils = [float(n.get("capacity_utilization", 0.75)) * 100.0 for n in nodes if "capacity_utilization" in n]
    average_capacity_utilization = float(np.mean(all_utils)) if all_utils else 75.0
    maximum_capacity_utilization = float(np.max(all_utils)) if all_utils else 90.0

    fac_utils = [float(f.get("capacity_utilization", 0.80)) * 100.0 for f in factory_nodes]
    factory_capacity_utilization = float(np.mean(fac_utils)) if fac_utils else average_capacity_utilization

    wh_utils = [float(w.get("capacity_utilization", 0.70)) * 100.0 for w in warehouse_nodes]
    warehouse_capacity_utilization = float(np.mean(wh_utils)) if wh_utils else average_capacity_utilization

    inv_candidates = factory_nodes + warehouse_nodes if (factory_nodes or warehouse_nodes) else nodes
    inv_days_list = [float(n.get("inventory_days", 20.0)) for n in inv_candidates if "inventory_days" in n]
    inventory_days = float(np.mean(inv_days_list)) if inv_days_list else 20.0

    safety_list = [float(n.get("safety_stock_days", 7.0)) for n in inv_candidates if "safety_stock_days" in n]
    safety_stock_days = float(np.mean(safety_list)) if safety_list else 7.0

    # Shock features
    raw_shock_type = str(shock_state.get("affected_node_type", shock_state.get("shock_type", "supplier"))).lower()
    shock_role = ROLE_MAP.get(raw_shock_type, "general")
    shock_type_cat_map = SHOCK_RULES_CFG.get("shock_type_categories", {
        "source": 0, "transit_gateway": 1, "corridor": 2, "processing": 3, "storage": 4, "distribution": 5, "general": 0
    })
    shock_type_encoded = float(shock_type_cat_map.get(shock_role, 0))

    shock_severity = float(shock_state.get("shock_severity", 0.5))
    shock_duration_days = float(shock_state.get("shock_duration_days", 14.0))
    affected_node_importance = float(shock_state.get("affected_node_importance", 0.6))
    affected_capacity_change_pct = float(shock_state.get("affected_capacity_change_pct", shock_severity * 100.0))
    affected_availability_change_pct = float(shock_state.get("affected_availability_change_pct", shock_severity * 100.0))
    affected_congestion_change_pct = float(shock_state.get("affected_congestion_change_pct", shock_severity * 50.0))
    affected_supply_percent = float(shock_state.get("affected_supply_percent", shock_severity * 40.0))
    affected_production_percent = float(shock_state.get("affected_production_percent", shock_severity * 35.0))

    # Route / Alternative Features
    additional_transit_days = float(shock_state.get("additional_transit_days", 0.0))
    additional_transport_cost_pct = float(shock_state.get("additional_transport_cost_pct", 0.0))
    alternate_route_available = float(shock_state.get("alternate_route_available", 1.0))
    alternate_route_count = float(shock_state.get("alternate_route_count", 1.0))
    alternate_route_delay_days = float(shock_state.get("alternate_route_delay_days", additional_transit_days))
    alternate_route_cost_increase_pct = float(shock_state.get("alternate_route_cost_increase_pct", additional_transport_cost_pct))
    rerouted_supply_percent = float(shock_state.get("rerouted_supply_percent", 50.0))

    # External / Macroeconomic Features
    usd_inr = float(ext.get("usd_inr", 83.5))
    crude_price = float(ext.get("crude_price", 82.0))
    commodity_price = float(ext.get("commodity_price", 100.0))
    inflation_rate = float(ext.get("inflation_rate", 5.2))
    freight_index = float(ext.get("freight_index", 115.0))

    # Assemble dictionary strictly matching CANONICAL_FEATURES
    feature_dict = {
        "supplier_count": float(supplier_count),
        "alternative_supplier_count": float(alternative_supplier_count),
        "port_count": float(port_count),
        "alternative_port_count": float(alternative_port_count),
        "factory_count": float(factory_count),
        "warehouse_count": float(warehouse_count),
        "distribution_center_count": float(distribution_center_count),
        "network_depth": float(network_depth),
        "network_edge_count": float(network_edge_count),
        "network_node_count": float(network_node_count),
        "supplier_concentration": round(float(supplier_concentration), 2),
        "port_dependency": round(float(port_dependency), 2),
        "single_point_of_failure": float(has_spof),
        "average_route_distance_km": round(float(average_route_distance_km), 2),
        "critical_route_distance_km": round(float(critical_route_distance_km), 2),
        "alternate_route_distance_km": round(float(alternate_route_distance_km), 2),
        "average_supplier_reliability": round(float(average_supplier_reliability), 3),
        "minimum_supplier_reliability": round(float(minimum_supplier_reliability), 3),
        "average_supplier_availability": round(float(average_supplier_availability), 3),
        "average_capacity_utilization": round(float(average_capacity_utilization), 2),
        "maximum_capacity_utilization": round(float(maximum_capacity_utilization), 2),
        "factory_capacity_utilization": round(float(factory_capacity_utilization), 2),
        "warehouse_capacity_utilization": round(float(warehouse_capacity_utilization), 2),
        "inventory_days": round(float(inventory_days), 2),
        "safety_stock_days": round(float(safety_stock_days), 2),
        "shock_type_encoded": float(shock_type_encoded),
        "shock_severity": round(float(shock_severity), 3),
        "shock_duration_days": round(float(shock_duration_days), 2),
        "affected_node_importance": round(float(affected_node_importance), 3),
        "affected_capacity_change_pct": round(float(affected_capacity_change_pct), 2),
        "affected_availability_change_pct": round(float(affected_availability_change_pct), 2),
        "affected_congestion_change_pct": round(float(affected_congestion_change_pct), 2),
        "affected_supply_percent": round(float(affected_supply_percent), 2),
        "affected_production_percent": round(float(affected_production_percent), 2),
        "additional_transit_days": round(float(additional_transit_days), 2),
        "additional_transport_cost_pct": round(float(additional_transport_cost_pct), 2),
        "alternate_route_available": float(alternate_route_available),
        "alternate_route_count": float(alternate_route_count),
        "alternate_route_delay_days": round(float(alternate_route_delay_days), 2),
        "alternate_route_cost_increase_pct": round(float(alternate_route_cost_increase_pct), 2),
        "rerouted_supply_percent": round(float(rerouted_supply_percent), 2),
        "usd_inr": round(float(usd_inr), 2),
        "crude_price": round(float(crude_price), 2),
        "commodity_price": round(float(commodity_price), 2),
        "inflation_rate": round(float(inflation_rate), 2),
        "freight_index": round(float(freight_index), 2)
    }

    return feature_dict


def feature_dict_to_array(feature_dict: Dict[str, float]) -> np.ndarray:
    """Converts feature dictionary to a NumPy vector with strictly guaranteed canonical ordering."""
    return np.array([feature_dict[feat] for feat in CANONICAL_FEATURES], dtype=np.float32).reshape(1, -1)


def feature_dict_to_dataframe(feature_dict: Dict[str, float]) -> pd.DataFrame:
    """Converts feature dictionary to a single-row DataFrame with exact canonical feature order."""
    return pd.DataFrame([{feat: feature_dict[feat] for feat in CANONICAL_FEATURES}])
