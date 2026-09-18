"""
Generic Deterministic Supply Chain Graph Engine for Vyuha ML.
Models arbitrary supply chain graphs (nodes, edges, capacities, routes),
discovers topology dynamically (sources, sinks, depth, cut-vertices/SPOFs),
and deterministically propagates attribute shocks without hardcoding.
"""

from typing import Dict, Any, List, Optional, Tuple, Set
import os
import json
import copy
import math


def load_config(filename: str) -> Dict[str, Any]:
    """Loads a configuration JSON file from ml/config/."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(base_dir, "config", filename)
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


class SupplyChainGraph:
    """
    Graph representation of any supply chain network.
    Contains arbitrary nodes and directed edges. Automatically discovers
    topological properties (sources, sinks, depth, articulation points).
    """

    def __init__(self, network_data: Optional[Dict[str, Any]] = None):
        self.network_name: str = "Supply Chain Network"
        self.industry: str = "General"
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []
        self.adjacency: Dict[str, List[Dict[str, Any]]] = {}
        self.in_degree: Dict[str, int] = {}
        self.out_degree: Dict[str, int] = {}

        if network_data:
            self.load_from_dict(network_data)

    def load_from_dict(self, data: Dict[str, Any]):
        """Populates graph from arbitrary dictionary with nodes and edges."""
        self.network_name = data.get("name", self.network_name)
        self.industry = data.get("industry", self.industry)

        raw_nodes = data.get("nodes", [])
        if isinstance(raw_nodes, dict):
            raw_nodes = list(raw_nodes.values())
        self.nodes = {n["id"]: copy.deepcopy(n) for n in raw_nodes if "id" in n}

        self.edges = copy.deepcopy(data.get("edges", []))
        self._build_graph_structures()

    def _build_graph_structures(self):
        """Constructs adjacency list and degree maps."""
        self.adjacency = {node_id: [] for node_id in self.nodes}
        self.in_degree = {node_id: 0 for node_id in self.nodes}
        self.out_degree = {node_id: 0 for node_id in self.nodes}

        for edge in self.edges:
            src = edge.get("source")
            tgt = edge.get("target")
            if src in self.adjacency and tgt in self.nodes:
                self.adjacency[src].append(edge)
                self.out_degree[src] = self.out_degree.get(src, 0) + 1
                self.in_degree[tgt] = self.in_degree.get(tgt, 0) + 1

    def get_sources(self) -> List[Dict[str, Any]]:
        """Identifies origin/supply nodes dynamically (in-degree == 0 or role alias)."""
        zero_in = [n for n_id, n in self.nodes.items() if self.in_degree.get(n_id, 0) == 0]
        if zero_in:
            return zero_in
        # Fallback for cyclic graphs: nodes with net positive out-degree
        net_out = [n for n_id, n in self.nodes.items() if self.out_degree.get(n_id, 0) > self.in_degree.get(n_id, 0)]
        return net_out if net_out else list(self.nodes.values())[:1]

    def get_sinks(self) -> List[Dict[str, Any]]:
        """Identifies terminal/demand nodes dynamically (out-degree == 0 or role alias)."""
        zero_out = [n for n_id, n in self.nodes.items() if self.out_degree.get(n_id, 0) == 0]
        if zero_out:
            return zero_out
        # Fallback for cyclic graphs: nodes with net positive in-degree
        net_in = [n for n_id, n in self.nodes.items() if self.in_degree.get(n_id, 0) > self.out_degree.get(n_id, 0)]
        return net_in if net_in else list(self.nodes.values())[-1:]

    def get_nodes_by_type(self, node_type: str) -> List[Dict[str, Any]]:
        """Filters nodes matching specific type string."""
        return [n for n in self.nodes.values() if str(n.get("type", "")).lower() == node_type.lower()]

    def find_all_paths(
        self,
        start: str,
        target: str,
        visited: Optional[Set[str]] = None,
        max_depth: int = 12
    ) -> List[List[str]]:
        """Finds all simple directed paths between two nodes."""
        if visited is None:
            visited = set()
        visited.add(start)
        if start == target:
            return [[start]]
        if len(visited) > max_depth:
            return []

        paths = []
        for edge in self.adjacency.get(start, []):
            nxt = edge["target"]
            if nxt not in visited and nxt in self.nodes:
                sub_paths = self.find_all_paths(nxt, target, visited.copy(), max_depth)
                for sp in sub_paths:
                    paths.append([start] + sp)
        return paths

    def find_all_source_to_sink_paths(self) -> List[List[str]]:
        """Discovers all feasible end-to-end paths from any source to any sink."""
        sources = self.get_sources()
        sinks = self.get_sinks()
        all_paths = []
        for src in sources:
            for snk in sinks:
                if src["id"] != snk["id"]:
                    paths = self.find_all_paths(src["id"], snk["id"])
                    all_paths.extend(paths)
        return all_paths

    def get_network_depth(self) -> int:
        """Calculates topological network depth (longest simple path length in hops)."""
        paths = self.find_all_source_to_sink_paths()
        if paths:
            return max(len(p) for p in paths)
        return max(1, len(self.nodes))

    def detect_single_points_of_failure(self) -> Set[str]:
        """
        Dynamically detects articulation points / cut-vertices.
        A node is an SPOF if removing it severs all paths between any source-sink pair.
        """
        sources = [s["id"] for s in self.get_sources()]
        sinks = [k["id"] for k in self.get_sinks()]
        baseline_paths = self.find_all_source_to_sink_paths()

        if not baseline_paths:
            return set()

        spof_nodes = set()
        # Test each internal node
        for node_id in self.nodes:
            # Check if all paths from at least one source to all reachable sinks pass through node_id
            for src in sources:
                for snk in sinks:
                    src_snk_paths = [p for p in baseline_paths if p[0] == src and p[-1] == snk]
                    if src_snk_paths:
                        # If EVERY path between this source and sink passes through node_id, it is an SPOF
                        if all(node_id in p for p in src_snk_paths):
                            spof_nodes.add(node_id)
        return spof_nodes

    def calculate_path_metrics(self, path: List[str]) -> Dict[str, float]:
        """Computes cumulative distance, transit days, and transport cost along a path."""
        total_dist = 0.0
        total_time = 0.0
        total_cost = 0.0
        for i in range(len(path) - 1):
            src, tgt = path[i], path[i + 1]
            edge = next((e for e in self.adjacency.get(src, []) if e.get("target") == tgt), None)
            if edge:
                total_dist += float(edge.get("distance_km", 0.0))
                total_time += float(edge.get("transit_time_days", 1.0))
                total_cost += float(edge.get("transport_cost_per_unit", 10.0))
        return {
            "distance_km": total_dist,
            "transit_days": total_time,
            "cost_per_unit": total_cost
        }


def propagate_shock(
    graph: SupplyChainGraph,
    shock: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Deterministically propagates a disruption shock through the supply chain graph.
    Accepts:
      - node_id (or target_id)
      - edge_id (optional)
      - attribute (congestion, capacity, availability, reliability, transit_time_days, transport_cost, severity)
      - current_value, new_value
      - duration_days
    Works with ANY arbitrary topology and computes physical consequences dynamically.
    """
    rules_cfg = load_config("shock_rules.json")
    attr_mappings = rules_cfg.get("attribute_shock_mappings", {})

    # 1. Resolve Target Node or Edge
    target_node_id = shock.get("node_id", shock.get("target_id"))
    target_edge_id = shock.get("edge_id")
    attribute = str(shock.get("attribute", "severity")).lower()
    duration_days = float(shock.get("duration_days", shock.get("shock_duration_days", 14.0)))

    affected_node = None
    affected_edge = None

    if target_node_id and target_node_id in graph.nodes:
        affected_node = graph.nodes[target_node_id]
    elif target_edge_id:
        affected_edge = next((e for e in graph.edges if e.get("id") == target_edge_id), None)

    # Fallback to type or primary node if target not specified
    if not affected_node and not affected_edge:
        raw_type = shock.get("shock_type", "supplier").lower()
        matching = graph.get_nodes_by_type(raw_type)
        if matching:
            affected_node = next((n for n in matching if n.get("is_primary", True)), matching[0])
        elif graph.nodes:
            sources = graph.get_sources()
            affected_node = sources[0] if sources else list(graph.nodes.values())[0]

    affected_id = affected_node["id"] if affected_node else (affected_edge.get("id", "") if affected_edge else "")
    affected_name = affected_node.get("name", affected_id) if affected_node else (affected_edge.get("name", affected_id) if affected_edge else "System Element")
    node_type = str(affected_node.get("type", "general")) if affected_node else "corridor"

    # 2. Derive Effective Disruption Severity from Generic Attribute Shock
    curr_val = shock.get("current_value")
    new_val = shock.get("new_value")

    # If legacy shock_severity is passed directly
    if curr_val is None and new_val is None:
        raw_sev = float(shock.get("shock_severity", 0.5))
        effective_severity = max(0.05, min(1.0, raw_sev))
        capacity_change_pct = effective_severity * 100.0
        availability_change_pct = effective_severity * 100.0
        congestion_change_pct = float(shock.get("congestion_increase_pct", effective_severity * 50.0))
        direct_transit_delay = 0.0
        direct_cost_surge_pct = 0.0
    else:
        curr_val = float(curr_val if curr_val is not None else 100.0)
        new_val = float(new_val if new_val is not None else 50.0)
        
        # Calculate attribute-specific perturbations
        if attribute == "capacity":
            capacity_change_pct = max(0.0, min(100.0, (curr_val - new_val) / max(curr_val, 1e-6) * 100.0))
            effective_severity = capacity_change_pct / 100.0
            availability_change_pct = capacity_change_pct * 0.8
            congestion_change_pct = capacity_change_pct * 0.6
            direct_transit_delay = 0.0
            direct_cost_surge_pct = 0.0
        elif attribute == "congestion":
            congestion_change_pct = max(0.0, min(100.0, new_val - curr_val))
            effective_severity = congestion_change_pct / 100.0
            capacity_change_pct = congestion_change_pct * 0.7
            availability_change_pct = congestion_change_pct * 0.5
            direct_transit_delay = (congestion_change_pct / 100.0) * (duration_days * 0.15)
            direct_cost_surge_pct = (congestion_change_pct / 100.0) * 25.0
        elif attribute in ["availability", "reliability"]:
            availability_change_pct = max(0.0, min(100.0, (curr_val - new_val) / max(curr_val, 1e-6) * 100.0))
            effective_severity = availability_change_pct / 100.0
            capacity_change_pct = availability_change_pct * 0.85
            congestion_change_pct = availability_change_pct * 0.4
            direct_transit_delay = 0.0
            direct_cost_surge_pct = 0.0
        elif attribute == "transit_time_days":
            added_days = max(0.0, new_val - curr_val)
            direct_transit_delay = added_days
            effective_severity = min(1.0, added_days / 14.0)
            capacity_change_pct = effective_severity * 40.0
            availability_change_pct = effective_severity * 40.0
            congestion_change_pct = effective_severity * 60.0
            direct_cost_surge_pct = effective_severity * 30.0
        elif attribute in ["transport_cost", "transport_cost_per_unit", "freight_cost"]:
            cost_diff = max(0.0, new_val - curr_val)
            direct_cost_surge_pct = (cost_diff / max(curr_val, 1e-6)) * 100.0
            effective_severity = min(1.0, direct_cost_surge_pct / 100.0)
            capacity_change_pct = 0.0
            availability_change_pct = 0.0
            congestion_change_pct = 0.0
            direct_transit_delay = 0.0
        else:
            # General severity
            effective_severity = max(0.05, min(1.0, (new_val / 100.0 if new_val > 1.0 else new_val)))
            capacity_change_pct = effective_severity * 100.0
            availability_change_pct = effective_severity * 100.0
            congestion_change_pct = effective_severity * 50.0
            direct_transit_delay = 0.0
            direct_cost_surge_pct = 0.0

    node_importance = float(affected_node.get("importance_weight", 0.6)) if affected_node else 0.5
    supply_share = float(affected_node.get("share_of_supply", affected_node.get("share_of_volume", 0.5))) if affected_node else 0.5

    # 3. Dynamic Path Discovery & Alternative Feasibility Analysis
    all_paths = graph.find_all_source_to_sink_paths()
    if not all_paths and graph.nodes:
        # Single edge or direct nodes fallback
        all_paths = [[n_id] for n_id in graph.nodes]

    # Baseline primary path (shortest/lowest cost path among all valid source-to-sink paths)
    baseline_path = all_paths[0] if all_paths else []
    baseline_metrics = graph.calculate_path_metrics(baseline_path) if baseline_path else {
        "distance_km": 500.0, "transit_days": 2.5, "cost_per_unit": 30.0
    }

    # Discover alternative paths that completely bypass the shocked node / edge
    feasible_alternates = []
    for p in all_paths:
        if affected_node and affected_id in p:
            continue
        if affected_edge and any(p[i] == affected_edge.get("source") and p[i+1] == affected_edge.get("target") for i in range(len(p)-1)):
            continue
        feasible_alternates.append(p)

    if feasible_alternates:
        alt_metrics = graph.calculate_path_metrics(feasible_alternates[0])
        alternate_available = 1
        alt_count = len(feasible_alternates)
        additional_transit_days = max(0.0, alt_metrics["transit_days"] - baseline_metrics["transit_days"] + direct_transit_delay)
        additional_transport_cost_pct = max(0.0, ((alt_metrics["cost_per_unit"] - baseline_metrics["cost_per_unit"]) / max(baseline_metrics["cost_per_unit"], 1.0)) * 100.0 + direct_cost_surge_pct)
        alt_route_dist = alt_metrics["distance_km"]
        # Rerouting volume capability
        rerouted_supply_percent = min(100.0, supply_share * (1.0 - effective_severity * 0.35) * 100.0)
    else:
        # No alternative paths exist (node is a bottleneck or SPOF)
        alternate_available = 0
        alt_count = 0
        additional_transit_days = direct_transit_delay + (effective_severity * duration_days * 0.5)
        additional_transport_cost_pct = direct_cost_surge_pct + (effective_severity * 35.0)
        alt_route_dist = baseline_metrics["distance_km"]
        rerouted_supply_percent = 0.0

    # 4. Supply & Production Bottleneck Calculation
    # Unmitigated lost supply percentage
    affected_supply_percent = max(0.0, (supply_share * effective_severity * 100.0) - (rerouted_supply_percent * 0.5))

    # Inventory buffer runway
    inv_days = float(affected_node.get("inventory_days", 18.0) if affected_node else 18.0)
    inventory_cushion = max(0.0, inv_days - duration_days)
    production_throttle = effective_severity * (1.0 - min(1.0, inventory_cushion / max(inv_days, 1.0)))
    affected_production_percent = round(float(production_throttle * 100.0), 2)

    return {
        "affected_node": affected_name,
        "affected_node_id": affected_id,
        "affected_node_type": node_type,
        "affected_node_importance": round(node_importance, 3),
        "shock_attribute": attribute,
        "shock_severity": round(effective_severity, 3),
        "shock_duration_days": round(duration_days, 2),
        "affected_capacity_change_pct": round(capacity_change_pct, 2),
        "affected_availability_change_pct": round(availability_change_pct, 2),
        "affected_congestion_change_pct": round(congestion_change_pct, 2),
        "affected_supply_percent": round(affected_supply_percent, 2),
        "affected_production_percent": round(affected_production_percent, 2),
        "additional_transit_days": round(additional_transit_days, 2),
        "additional_transport_cost_pct": round(additional_transport_cost_pct, 2),
        "alternate_route_available": alternate_available,
        "alternate_route_count": alt_count,
        "alternate_route_delay_days": round(additional_transit_days, 2),
        "alternate_route_cost_increase_pct": round(additional_transport_cost_pct, 2),
        "rerouted_supply_percent": round(rerouted_supply_percent, 2),
        "primary_route_distance_km": round(baseline_metrics["distance_km"], 2),
        "alternate_route_distance_km": round(alt_route_dist, 2),
        "available_alternate_routes": feasible_alternates[:3]
    }
