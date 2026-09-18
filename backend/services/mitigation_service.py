"""
Mitigation Recommendation Engine for Vyuha ML.
Dynamically resolves supply chain interventions, alternate ports, and multimodal
routing from configuration catalogs and model vulnerability rankings.
"""

import os
import json
from typing import Dict, Any, List

def load_corridors_config() -> Dict[str, Any]:
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "logistics_corridors.json"
    )
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {}

def generate_mitigations(
    company: Dict[str, Any],
    shock: Dict[str, Any],
    vulnerabilities: Dict[str, Any],
    risk_score: float,
    stockout_day: int = None
) -> List[Dict[str, Any]]:
    corridors = load_corridors_config().get("port_alternatives", {})
    
    import_dep = float(company.get("import_dependency", 50.0))
    supplier_conc = float(company.get("supplier_concentration", 50.0))
    alt_suppliers = int(company.get("alternative_suppliers_count", 2))
    port_dep = float(company.get("port_dependency", 50.0))
    primary_port = str(company.get("primary_port", "JNPT (Navi Mumbai)"))
    road_dep = float(company.get("road_dependency", 60.0))
    inventory_days = float(company.get("inventory_days", 20.0))
    
    port_congestion = float(shock.get("port_congestion_pct", 25.0))
    usd_change = float(shock.get("usd_inr_change_pct", 0.0))
    crude_change = float(shock.get("crude_price_change_pct", 0.0))
    
    mitigations = []
    
    # 1. Port Congestion & Disruption
    port_info = corridors.get(primary_port, {
        "alternate": "Nearest Regional Container Terminal",
        "delay_savings_days": "3 - 6 days",
        "rail_corridor": "Dedicated Freight Corridor"
    })
    
    if port_dep > 45 or port_congestion > 50:
        alt_name = port_info["alternate"]
        savings = port_info["delay_savings_days"]
        mitigations.append({
            "category": "Immediate (0-7 Days)",
            "title": f"Evaluate Port Rerouting to {alt_name}",
            "description": f"Primary gateway port ({primary_port}) is facing elevated congestion ({port_congestion}%). Divert inbound customs bonded bills of lading to {alt_name} to bypass container dwell time.",
            "impact": f"Reduces maritime delay by {savings}",
            "effort": "Medium",
            "risk_type": "Port & Maritime"
        })
        
    # 2. Inventory Buffer & Imminent Stockout
    inv_risk = vulnerabilities.get("dimensions", {}).get("inventory_risk", 0)
    if stockout_day is not None or inventory_days < 22 or inv_risk > 60:
        extra_days = 10 if inventory_days < 15 else 7
        urgency = "URGENT: " if stockout_day and stockout_day <= 15 else ""
        mitigations.append({
            "category": "Immediate (0-7 Days)",
            "title": f"{urgency}Advance Buffer Stocking (+{extra_days} Days)",
            "description": f"Current inventory ({inventory_days} days) risks line starvation before delayed shipments arrive. Issue emergency purchase orders or expedite air-cargo for critical A-class components.",
            "impact": "Eliminates stockout vulnerability window",
            "effort": "High",
            "risk_type": "Inventory"
        })

    # 3. Supplier Concentration & Single Sourcing
    if supplier_conc > 50 or alt_suppliers <= 1:
        mitigations.append({
            "category": "Tactical (1-4 Weeks)",
            "title": "Activate Domestic Indian Backup Sourcing",
            "description": f"Supplier concentration is high ({supplier_conc}%) with only {alt_suppliers} alternative vendor(s). Qualify tier-2 domestic MSME suppliers in regional manufacturing clusters (Pune, Coimbatore, Rajkot, Peenya).",
            "impact": "Diversifies critical component flow & reduces single-vendor markup",
            "effort": "Medium",
            "risk_type": "Supplier"
        })
        
    # 4. Currency / USD-INR Volatility
    if import_dep > 40 and usd_change > 2.0:
        mitigations.append({
            "category": "Tactical (1-4 Weeks)",
            "title": "Execute USD/INR Forward Contracts & Rupee Trade Invoicing",
            "description": f"With {import_dep}% import exposure and USD/INR appreciating by {usd_change}%, hedge next 60-90 days import receivables via forward booking with authorized dealer banks or explore rupee-settled bilateral mechanisms.",
            "impact": "Protects procurement budget against further currency slide",
            "effort": "Low",
            "risk_type": "Financial/Forex"
        })
        
    # 5. Road Freight & Fuel Surcharges
    if road_dep > 55 or crude_change > 15.0:
        corridor_rail = port_info.get("rail_corridor", "Electrified Rail Corridor")
        mitigations.append({
            "category": "Tactical (1-4 Weeks)",
            "title": f"Shift Long-Haul Freight to {corridor_rail}",
            "description": f"Road transport dependency ({road_dep}%) leaves operations exposed to diesel price hikes and highway congestions. Shift bulk container movements to CONCOR rail cargo on {corridor_rail}.",
            "impact": "Cuts transit cost by 12-18% and improves transit predictability",
            "effort": "Medium",
            "risk_type": "Logistics"
        })

    # 6. Strategic Structural Resilience
    if risk_score > 60 or import_dep > 65:
        mitigations.append({
            "category": "Strategic (>1 Month)",
            "title": "Localize Tier-1 Sub-assemblies (Atmanirbhar Sourcing)",
            "description": "Establish long-term vendor development programs in Indian industrial corridors. Transition from imported completely built units (CBU) to semi-knocked-down (SKD) local assembly to lower import tariffs and freight exposures.",
            "impact": "Permanently reduces baseline risk score by 20-30 points",
            "effort": "High",
            "risk_type": "Strategic"
        })
        
    if len(mitigations) < 3:
        mitigations.append({
            "category": "Tactical (1-4 Weeks)",
            "title": "Establish Real-Time Milestone Tracking with Logistics Service Providers (LSPs)",
            "description": "Enforce mandatory GPS / Fastag tracking and daily exception reporting from 3PL logistics partners to detect transit bottlenecks before delivery SLA breaches occur.",
            "impact": "Improves supply chain visibility by 40%",
            "effort": "Low",
            "risk_type": "Visibility"
        })
        
    return mitigations
