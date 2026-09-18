"""
Generates Curated Representative Demonstration Supply Networks for Vyuha ML.
Covers 5 core Indian industrial sectors with regional hub variations.
"""

import json
import os
from typing import Dict, Any, List


def create_demo_networks() -> Dict[str, Dict[str, Any]]:
    networks = {}

    # 1. Electronics - Hyderabad Hub
    networks["electronics_hyderabad"] = {
        "id": "electronics_hyderabad",
        "name": "Representative / Demonstration Supply Network: Consumer Electronics (Hyderabad)",
        "industry": "Electronics",
        "region": "Telangana & Andhra Pradesh",
        "network_depth": 5,
        "nodes": [
            {"id": "sup_chip_primary", "name": "Primary Semiconductor Foundry (Taiwan/S.Korea)", "type": "supplier", "is_primary": True, "share_of_supply": 0.70, "reliability": 0.94, "availability": 0.95},
            {"id": "sup_pcb_alt", "name": "Domestic PCB & Passive Supplier (Bengaluru)", "type": "supplier", "is_primary": False, "share_of_supply": 0.30, "reliability": 0.88, "availability": 0.92},
            {"id": "port_chennai", "name": "Chennai Port Gateway", "type": "port", "is_primary": True, "share_of_volume": 0.75, "capacity_utilization": 0.78, "availability": 0.92},
            {"id": "port_krishnapatnam", "name": "Krishnapatnam Port (Alternative)", "type": "port", "is_primary": False, "share_of_volume": 0.25, "capacity_utilization": 0.55, "availability": 0.95},
            {"id": "factory_hyd", "name": "Fab City Electronic Assembly Plant (Hyderabad)", "type": "factory", "capacity_utilization": 0.82, "inventory_days": 18.0, "safety_stock_days": 6.0, "importance_weight": 0.9},
            {"id": "wh_secunderabad", "name": "Central Finished Goods Hub (Secunderabad)", "type": "warehouse", "capacity_utilization": 0.70, "inventory_days": 22.0, "safety_stock_days": 8.0},
            {"id": "dc_south", "name": "South India Distribution Center", "type": "distribution_center", "capacity_utilization": 0.75}
        ],
        "edges": [
            {"id": "e_sea_chennai", "source": "sup_chip_primary", "target": "port_chennai", "distance_km": 4200, "transit_time_days": 12.0, "transport_cost_per_unit": 45.0, "mode": "Sea", "is_primary": True},
            {"id": "e_sea_krishna", "source": "sup_chip_primary", "target": "port_krishnapatnam", "distance_km": 4350, "transit_time_days": 13.0, "transport_cost_per_unit": 52.0, "mode": "Sea", "is_alternate": True},
            {"id": "e_road_blr_hyd", "source": "sup_pcb_alt", "target": "factory_hyd", "distance_km": 570, "transit_time_days": 1.5, "transport_cost_per_unit": 22.0, "mode": "Road", "is_alternate": True},
            {"id": "e_chennai_hyd", "source": "port_chennai", "target": "factory_hyd", "distance_km": 630, "transit_time_days": 2.0, "transport_cost_per_unit": 35.0, "mode": "Road", "is_primary": True},
            {"id": "e_krishna_hyd", "source": "port_krishnapatnam", "target": "factory_hyd", "distance_km": 460, "transit_time_days": 1.5, "transport_cost_per_unit": 38.0, "mode": "Road", "is_alternate": True},
            {"id": "e_fac_wh", "source": "factory_hyd", "target": "wh_secunderabad", "distance_km": 45, "transit_time_days": 0.2, "transport_cost_per_unit": 6.0, "mode": "Road", "is_primary": True},
            {"id": "e_wh_dc", "source": "wh_secunderabad", "target": "dc_south", "distance_km": 380, "transit_time_days": 1.0, "transport_cost_per_unit": 18.0, "mode": "Road", "is_primary": True}
        ]
    }

    # 2. Electronics - Bengaluru Hub
    networks["electronics_bengaluru"] = {
        "id": "electronics_bengaluru",
        "name": "Representative / Demonstration Supply Network: Smart Hardware (Bengaluru)",
        "industry": "Electronics",
        "region": "Karnataka",
        "network_depth": 5,
        "nodes": [
            {"id": "sup_display_primary", "name": "OLED Module Supplier (East Asia)", "type": "supplier", "is_primary": True, "share_of_supply": 0.65, "reliability": 0.95, "availability": 0.94},
            {"id": "sup_enclosure_alt", "name": "Precision Metal Supplier (Hosur)", "type": "supplier", "is_primary": False, "share_of_supply": 0.35, "reliability": 0.91, "availability": 0.96},
            {"id": "port_chennai_blr", "name": "Chennai Port", "type": "port", "is_primary": True, "share_of_volume": 0.80, "capacity_utilization": 0.82, "availability": 0.90},
            {"id": "port_mangalore", "name": "New Mangalore Port (Alternative)", "type": "port", "is_primary": False, "share_of_volume": 0.20, "capacity_utilization": 0.50, "availability": 0.96},
            {"id": "factory_blr", "name": "Electronics City Manufacturing Facility (Bengaluru)", "type": "factory", "capacity_utilization": 0.85, "inventory_days": 16.0, "safety_stock_days": 5.0, "importance_weight": 0.92},
            {"id": "wh_whitefield", "name": "Whitefield Fulfilment Hub", "type": "warehouse", "capacity_utilization": 0.78, "inventory_days": 20.0, "safety_stock_days": 7.0},
            {"id": "dc_pan_india", "name": "National Air/Road Dispatch Center", "type": "distribution_center", "capacity_utilization": 0.80}
        ],
        "edges": [
            {"id": "e_sea_chn_blr", "source": "sup_display_primary", "target": "port_chennai_blr", "distance_km": 4100, "transit_time_days": 11.0, "transport_cost_per_unit": 42.0, "mode": "Sea", "is_primary": True},
            {"id": "e_sea_mng_blr", "source": "sup_display_primary", "target": "port_mangalore", "distance_km": 4800, "transit_time_days": 14.0, "transport_cost_per_unit": 49.0, "mode": "Sea", "is_alternate": True},
            {"id": "e_hosur_blr", "source": "sup_enclosure_alt", "target": "factory_blr", "distance_km": 40, "transit_time_days": 0.2, "transport_cost_per_unit": 8.0, "mode": "Road", "is_alternate": True},
            {"id": "e_chn_fac_blr", "source": "port_chennai_blr", "target": "factory_blr", "distance_km": 350, "transit_time_days": 1.2, "transport_cost_per_unit": 24.0, "mode": "Road", "is_primary": True},
            {"id": "e_mng_fac_blr", "source": "port_mangalore", "target": "factory_blr", "distance_km": 360, "transit_time_days": 1.5, "transport_cost_per_unit": 28.0, "mode": "Road", "is_alternate": True},
            {"id": "e_blr_wh", "source": "factory_blr", "target": "wh_whitefield", "distance_km": 30, "transit_time_days": 0.1, "transport_cost_per_unit": 5.0, "mode": "Road", "is_primary": True},
            {"id": "e_wh_pan", "source": "wh_whitefield", "target": "dc_pan_india", "distance_km": 25, "transit_time_days": 0.1, "transport_cost_per_unit": 4.0, "mode": "Road", "is_primary": True}
        ]
    }

    # 3. Automotive - Chennai Hub
    networks["automotive_chennai"] = {
        "id": "automotive_chennai",
        "name": "Representative / Demonstration Supply Network: Passenger Vehicle Assembly (Chennai)",
        "industry": "Automotive",
        "region": "Tamil Nadu Auto Cluster",
        "network_depth": 5,
        "nodes": [
            {"id": "sup_tier1_engine", "name": "Powertrain & Gearbox Supplier (Coimbatore)", "type": "supplier", "is_primary": True, "share_of_supply": 0.60, "reliability": 0.96, "availability": 0.95},
            {"id": "sup_ecu_import", "name": "ECU & Semiconductor Modules (Germany/Japan)", "type": "supplier", "is_primary": True, "share_of_supply": 0.40, "reliability": 0.92, "availability": 0.90},
            {"id": "port_ennore", "name": "Kamarajar (Ennore) Automobile Port", "type": "port", "is_primary": True, "share_of_volume": 0.85, "capacity_utilization": 0.76, "availability": 0.94},
            {"id": "port_chennai_auto", "name": "Chennai Seaport (Alternative)", "type": "port", "is_primary": False, "share_of_volume": 0.15, "capacity_utilization": 0.85, "availability": 0.90},
            {"id": "factory_sriperumbudur", "name": "Vehicle Mega Assembly Plant (Sriperumbudur)", "type": "factory", "capacity_utilization": 0.88, "inventory_days": 12.0, "safety_stock_days": 4.0, "importance_weight": 0.95},
            {"id": "wh_oragadam", "name": "Finished Vehicle Yard (Oragadam)", "type": "warehouse", "capacity_utilization": 0.82, "inventory_days": 15.0, "safety_stock_days": 5.0},
            {"id": "dc_dealers_south", "name": "Regional Dealership Consolidation Center", "type": "distribution_center", "capacity_utilization": 0.70}
        ],
        "edges": [
            {"id": "e_cbe_chn", "source": "sup_tier1_engine", "target": "factory_sriperumbudur", "distance_km": 500, "transit_time_days": 1.5, "transport_cost_per_unit": 65.0, "mode": "Road", "is_primary": True},
            {"id": "e_sea_ecu", "source": "sup_ecu_import", "target": "port_ennore", "distance_km": 7200, "transit_time_days": 18.0, "transport_cost_per_unit": 85.0, "mode": "Sea", "is_primary": True},
            {"id": "e_sea_ecu_alt", "source": "sup_ecu_import", "target": "port_chennai_auto", "distance_km": 7150, "transit_time_days": 18.5, "transport_cost_per_unit": 90.0, "mode": "Sea", "is_alternate": True},
            {"id": "e_ennore_srip", "source": "port_ennore", "target": "factory_sriperumbudur", "distance_km": 65, "transit_time_days": 0.3, "transport_cost_per_unit": 15.0, "mode": "Road", "is_primary": True},
            {"id": "e_chn_srip", "source": "port_chennai_auto", "target": "factory_sriperumbudur", "distance_km": 55, "transit_time_days": 0.4, "transport_cost_per_unit": 18.0, "mode": "Road", "is_alternate": True},
            {"id": "e_srip_orag", "source": "factory_sriperumbudur", "target": "wh_oragadam", "distance_km": 20, "transit_time_days": 0.1, "transport_cost_per_unit": 8.0, "mode": "Road", "is_primary": True},
            {"id": "e_orag_dealers", "source": "wh_oragadam", "target": "dc_dealers_south", "distance_km": 400, "transit_time_days": 1.0, "transport_cost_per_unit": 45.0, "mode": "Road", "is_primary": True}
        ]
    }

    # 4. Automotive - Pune Hub
    networks["automotive_pune"] = {
        "id": "automotive_pune",
        "name": "Representative / Demonstration Supply Network: Commercial Vehicle & Auto Components (Pune)",
        "industry": "Automotive",
        "region": "Maharashtra Auto Corridor",
        "network_depth": 5,
        "nodes": [
            {"id": "sup_chassis_pune", "name": "Chassis & Frame Manufacturer (Aurangabad)", "type": "supplier", "is_primary": True, "share_of_supply": 0.65, "reliability": 0.94, "availability": 0.93},
            {"id": "sup_castings_kolhapur", "name": "Foundry & Castings Supplier (Kolhapur)", "type": "supplier", "is_primary": False, "share_of_supply": 0.35, "reliability": 0.90, "availability": 0.91},
            {"id": "port_jnpt_pune", "name": "JNPT Navi Mumbai Container Port", "type": "port", "is_primary": True, "share_of_volume": 0.85, "capacity_utilization": 0.80, "availability": 0.91},
            {"id": "port_mumbai_old", "name": "Mumbai Port Trust (Alternative)", "type": "port", "is_primary": False, "share_of_volume": 0.15, "capacity_utilization": 0.65, "availability": 0.88},
            {"id": "factory_chakan", "name": "Chakan Auto Assembly Plant (Pune)", "type": "factory", "capacity_utilization": 0.86, "inventory_days": 14.0, "safety_stock_days": 4.5, "importance_weight": 0.93},
            {"id": "wh_talegaon", "name": "Talegaon Logistics Park", "type": "warehouse", "capacity_utilization": 0.75, "inventory_days": 16.0, "safety_stock_days": 5.0},
            {"id": "dc_west_dealers", "name": "Western India Dealership Hub", "type": "distribution_center", "capacity_utilization": 0.72}
        ],
        "edges": [
            {"id": "e_aur_pune", "source": "sup_chassis_pune", "target": "factory_chakan", "distance_km": 230, "transit_time_days": 0.8, "transport_cost_per_unit": 35.0, "mode": "Road", "is_primary": True},
            {"id": "e_kol_pune", "source": "sup_castings_kolhapur", "target": "factory_chakan", "distance_km": 240, "transit_time_days": 0.8, "transport_cost_per_unit": 32.0, "mode": "Road", "is_alternate": True},
            {"id": "e_jnpt_chakan", "source": "port_jnpt_pune", "target": "factory_chakan", "distance_km": 135, "transit_time_days": 0.6, "transport_cost_per_unit": 20.0, "mode": "Road", "is_primary": True},
            {"id": "e_mbpt_chakan", "source": "port_mumbai_old", "target": "factory_chakan", "distance_km": 160, "transit_time_days": 0.8, "transport_cost_per_unit": 25.0, "mode": "Road", "is_alternate": True},
            {"id": "e_chakan_tale", "source": "factory_chakan", "target": "wh_talegaon", "distance_km": 35, "transit_time_days": 0.1, "transport_cost_per_unit": 6.0, "mode": "Road", "is_primary": True},
            {"id": "e_tale_west", "source": "wh_talegaon", "target": "dc_west_dealers", "distance_km": 320, "transit_time_days": 0.9, "transport_cost_per_unit": 38.0, "mode": "Road", "is_primary": True}
        ]
    }

    # 5. Pharmaceuticals - Hyderabad Hub
    networks["pharma_hyderabad"] = {
        "id": "pharma_hyderabad",
        "name": "Representative / Demonstration Supply Network: Formulations & Bulk Drugs (Genome Valley Hyderabad)",
        "industry": "Pharmaceuticals",
        "region": "Telangana Pharma Cluster",
        "network_depth": 5,
        "nodes": [
            {"id": "sup_api_global", "name": "Global Active Pharmaceutical Ingredient (API) Supplier", "type": "supplier", "is_primary": True, "share_of_supply": 0.70, "reliability": 0.91, "availability": 0.92},
            {"id": "sup_excipient_local", "name": "Domestic Excipients & Solvents Supplier (Vizag)", "type": "supplier", "is_primary": False, "share_of_supply": 0.30, "reliability": 0.89, "availability": 0.95},
            {"id": "port_jnpt_pharma", "name": "JNPT Navi Mumbai Sea Gateway", "type": "port", "is_primary": True, "share_of_volume": 0.70, "capacity_utilization": 0.82, "availability": 0.90},
            {"id": "port_vizag_pharma", "name": "Visakhapatnam Port (Alternative)", "type": "port", "is_primary": False, "share_of_volume": 0.30, "capacity_utilization": 0.60, "availability": 0.94},
            {"id": "factory_genome_valley", "name": "Genome Valley Formulations Facility (Hyderabad)", "type": "factory", "capacity_utilization": 0.84, "inventory_days": 28.0, "safety_stock_days": 10.0, "importance_weight": 0.94},
            {"id": "wh_cold_chain_hyd", "name": "Temperature-Controlled Central Cold Storage (Shamshabad)", "type": "warehouse", "capacity_utilization": 0.76, "inventory_days": 35.0, "safety_stock_days": 12.0},
            {"id": "dc_hospital_supply", "name": "Hospitals & Institutional Pharmacy Network", "type": "distribution_center", "capacity_utilization": 0.70}
        ],
        "edges": [
            {"id": "e_sea_jnpt_pharma", "source": "sup_api_global", "target": "port_jnpt_pharma", "distance_km": 5400, "transit_time_days": 15.0, "transport_cost_per_unit": 60.0, "mode": "Sea", "is_primary": True},
            {"id": "e_sea_vizag_pharma", "source": "sup_api_global", "target": "port_vizag_pharma", "distance_km": 4900, "transit_time_days": 13.5, "transport_cost_per_unit": 55.0, "mode": "Sea", "is_alternate": True},
            {"id": "e_vizag_sup_hyd", "source": "sup_excipient_local", "target": "factory_genome_valley", "distance_km": 620, "transit_time_days": 1.8, "transport_cost_per_unit": 28.0, "mode": "Road", "is_alternate": True},
            {"id": "e_jnpt_genome", "source": "port_jnpt_pharma", "target": "factory_genome_valley", "distance_km": 720, "transit_time_days": 2.2, "transport_cost_per_unit": 42.0, "mode": "Road", "is_primary": True},
            {"id": "e_vizag_genome", "source": "port_vizag_pharma", "target": "factory_genome_valley", "distance_km": 640, "transit_time_days": 1.9, "transport_cost_per_unit": 38.0, "mode": "Road", "is_alternate": True},
            {"id": "e_genome_cold", "source": "factory_genome_valley", "target": "wh_cold_chain_hyd", "distance_km": 60, "transit_time_days": 0.2, "transport_cost_per_unit": 12.0, "mode": "Road", "is_primary": True},
            {"id": "e_cold_hospitals", "source": "wh_cold_chain_hyd", "target": "dc_hospital_supply", "distance_km": 350, "transit_time_days": 1.0, "transport_cost_per_unit": 30.0, "mode": "Road", "is_primary": True}
        ]
    }

    # 6. Pharmaceuticals - Gujarat Hub
    networks["pharma_gujarat"] = {
        "id": "pharma_gujarat",
        "name": "Representative / Demonstration Supply Network: Sterile Injectables (Vadodara & Ahmedabad)",
        "industry": "Pharmaceuticals",
        "region": "Gujarat Pharma Corridor",
        "network_depth": 5,
        "nodes": [
            {"id": "sup_vial_primary", "name": "Specialty Glass Vial & Polymer Supplier (Mumbai)", "type": "supplier", "is_primary": True, "share_of_supply": 0.65, "reliability": 0.95, "availability": 0.94},
            {"id": "sup_api_chem", "name": "Chemical Intermediate Supplier (Ankleshwar)", "type": "supplier", "is_primary": False, "share_of_supply": 0.35, "reliability": 0.92, "availability": 0.96},
            {"id": "port_mundra_pharma", "name": "Mundra Port (Gujarat)", "type": "port", "is_primary": True, "share_of_volume": 0.80, "capacity_utilization": 0.74, "availability": 0.95},
            {"id": "port_hazira_pharma", "name": "Hazira Port Surat (Alternative)", "type": "port", "is_primary": False, "share_of_volume": 0.20, "capacity_utilization": 0.58, "availability": 0.96},
            {"id": "factory_sanand", "name": "Sterile Injectable Manufacturing Plant (Sanand)", "type": "factory", "capacity_utilization": 0.82, "inventory_days": 25.0, "safety_stock_days": 9.0, "importance_weight": 0.92},
            {"id": "wh_changodar", "name": "Changodar Central Pharma Warehouse", "type": "warehouse", "capacity_utilization": 0.74, "inventory_days": 30.0, "safety_stock_days": 11.0},
            {"id": "dc_national_retail", "name": "National Pharmacy Distributor Network", "type": "distribution_center", "capacity_utilization": 0.75}
        ],
        "edges": [
            {"id": "e_mum_sanand", "source": "sup_vial_primary", "target": "factory_sanand", "distance_km": 520, "transit_time_days": 1.4, "transport_cost_per_unit": 32.0, "mode": "Road", "is_primary": True},
            {"id": "e_ank_sanand", "source": "sup_api_chem", "target": "factory_sanand", "distance_km": 210, "transit_time_days": 0.6, "transport_cost_per_unit": 16.0, "mode": "Road", "is_alternate": True},
            {"id": "e_mundra_sanand", "source": "port_mundra_pharma", "target": "factory_sanand", "distance_km": 340, "transit_time_days": 1.0, "transport_cost_per_unit": 26.0, "mode": "Road", "is_primary": True},
            {"id": "e_hazira_sanand", "source": "port_hazira_pharma", "target": "factory_sanand", "distance_km": 280, "transit_time_days": 0.8, "transport_cost_per_unit": 22.0, "mode": "Road", "is_alternate": True},
            {"id": "e_sanand_wh", "source": "factory_sanand", "target": "wh_changodar", "distance_km": 25, "transit_time_days": 0.1, "transport_cost_per_unit": 5.0, "mode": "Road", "is_primary": True},
            {"id": "e_changodar_dc", "source": "wh_changodar", "target": "dc_national_retail", "distance_km": 480, "transit_time_days": 1.2, "transport_cost_per_unit": 36.0, "mode": "Road", "is_primary": True}
        ]
    }

    # 7. Consumer Goods (FMCG) - Mumbai Hub
    networks["consumer_goods_mumbai"] = {
        "id": "consumer_goods_mumbai",
        "name": "Representative / Demonstration Supply Network: FMCG Packaged Goods (Bhiwandi & Mumbai)",
        "industry": "Consumer Goods",
        "region": "MMR & Western India",
        "network_depth": 5,
        "nodes": [
            {"id": "sup_packaging_vapi", "name": "Corrugated Packaging Supplier (Vapi)", "type": "supplier", "is_primary": True, "share_of_supply": 0.60, "reliability": 0.93, "availability": 0.94},
            {"id": "sup_oils_kandla", "name": "Edible Oils & Fragrances (Kandla)", "type": "supplier", "is_primary": False, "share_of_supply": 0.40, "reliability": 0.90, "availability": 0.92},
            {"id": "port_jnpt_fmcg", "name": "JNPT Container Terminal", "type": "port", "is_primary": True, "share_of_volume": 0.80, "capacity_utilization": 0.78, "availability": 0.92},
            {"id": "port_hazira_fmcg", "name": "Hazira Port (Surat Alternate)", "type": "port", "is_primary": False, "share_of_volume": 0.20, "capacity_utilization": 0.60, "availability": 0.95},
            {"id": "factory_bhiwandi", "name": "Mega FMCG Processing Plant (Bhiwandi)", "type": "factory", "capacity_utilization": 0.85, "inventory_days": 15.0, "safety_stock_days": 5.0, "importance_weight": 0.90},
            {"id": "wh_panvel", "name": "Panvel High-Speed Regional Fulfillment Center", "type": "warehouse", "capacity_utilization": 0.80, "inventory_days": 18.0, "safety_stock_days": 6.0},
            {"id": "dc_modern_retail", "name": "Modern Trade & Supermarket Distribution Hub", "type": "distribution_center", "capacity_utilization": 0.78}
        ],
        "edges": [
            {"id": "e_vapi_bhiwandi", "source": "sup_packaging_vapi", "target": "factory_bhiwandi", "distance_km": 145, "transit_time_days": 0.5, "transport_cost_per_unit": 18.0, "mode": "Road", "is_primary": True},
            {"id": "e_kandla_bhiwandi", "source": "sup_oils_kandla", "target": "factory_bhiwandi", "distance_km": 780, "transit_time_days": 2.2, "transport_cost_per_unit": 46.0, "mode": "Road", "is_alternate": True},
            {"id": "e_jnpt_bhiwandi", "source": "port_jnpt_fmcg", "target": "factory_bhiwandi", "distance_km": 65, "transit_time_days": 0.3, "transport_cost_per_unit": 14.0, "mode": "Road", "is_primary": True},
            {"id": "e_hazira_bhiwandi", "source": "port_hazira_fmcg", "target": "factory_bhiwandi", "distance_km": 250, "transit_time_days": 0.8, "transport_cost_per_unit": 28.0, "mode": "Road", "is_alternate": True},
            {"id": "e_bhiwandi_panvel", "source": "factory_bhiwandi", "target": "wh_panvel", "distance_km": 50, "transit_time_days": 0.2, "transport_cost_per_unit": 9.0, "mode": "Road", "is_primary": True},
            {"id": "e_panvel_retail", "source": "wh_panvel", "target": "dc_modern_retail", "distance_km": 300, "transit_time_days": 0.8, "transport_cost_per_unit": 32.0, "mode": "Road", "is_primary": True}
        ]
    }

    # 8. Consumer Goods - Delhi NCR Hub
    networks["consumer_goods_delhi"] = {
        "id": "consumer_goods_delhi",
        "name": "Representative / Demonstration Supply Network: Personal Care & Home Essentials (Delhi NCR)",
        "industry": "Consumer Goods",
        "region": "Northern India Logistics Arc",
        "network_depth": 5,
        "nodes": [
            {"id": "sup_paper_rudrapur", "name": "Packaging & Carton Mills (Rudrapur)", "type": "supplier", "is_primary": True, "share_of_supply": 0.65, "reliability": 0.94, "availability": 0.95},
            {"id": "sup_chem_baddi", "name": "Surfactant & Fragrance Base (Baddi HP)", "type": "supplier", "is_primary": False, "share_of_supply": 0.35, "reliability": 0.91, "availability": 0.93},
            {"id": "port_icd_tughlakabad", "name": "ICD Tughlakabad Inland Rail Terminal", "type": "port", "is_primary": True, "share_of_volume": 0.85, "capacity_utilization": 0.82, "availability": 0.89},
            {"id": "port_icd_dadri", "name": "ICD Dadri (Western DFC Terminal)", "type": "port", "is_primary": False, "share_of_volume": 0.15, "capacity_utilization": 0.62, "availability": 0.94},
            {"id": "factory_haridwar", "name": "Integrated FMCG Manufacturing Complex (Haridwar)", "type": "factory", "capacity_utilization": 0.86, "inventory_days": 16.0, "safety_stock_days": 5.5, "importance_weight": 0.91},
            {"id": "wh_manesar", "name": "Manesar Mega Distribution Warehouse", "type": "warehouse", "capacity_utilization": 0.78, "inventory_days": 18.0, "safety_stock_days": 6.5},
            {"id": "dc_north_retail", "name": "North India Kirana & Retail Network", "type": "distribution_center", "capacity_utilization": 0.75}
        ],
        "edges": [
            {"id": "e_rudra_hari", "source": "sup_paper_rudrapur", "target": "factory_haridwar", "distance_km": 190, "transit_time_days": 0.6, "transport_cost_per_unit": 20.0, "mode": "Road", "is_primary": True},
            {"id": "e_baddi_hari", "source": "sup_chem_baddi", "target": "factory_haridwar", "distance_km": 210, "transit_time_days": 0.7, "transport_cost_per_unit": 24.0, "mode": "Road", "is_alternate": True},
            {"id": "e_icd_tkd_hari", "source": "port_icd_tughlakabad", "target": "factory_haridwar", "distance_km": 220, "transit_time_days": 0.8, "transport_cost_per_unit": 26.0, "mode": "Road", "is_primary": True},
            {"id": "e_icd_dadri_hari", "source": "port_icd_dadri", "target": "factory_haridwar", "distance_km": 205, "transit_time_days": 0.7, "transport_cost_per_unit": 25.0, "mode": "Road", "is_alternate": True},
            {"id": "e_hari_manesar", "source": "factory_haridwar", "target": "wh_manesar", "distance_km": 250, "transit_time_days": 0.8, "transport_cost_per_unit": 28.0, "mode": "Road", "is_primary": True},
            {"id": "e_manesar_retail", "source": "wh_manesar", "target": "dc_north_retail", "distance_km": 280, "transit_time_days": 0.9, "transport_cost_per_unit": 32.0, "mode": "Road", "is_primary": True}
        ]
    }

    # 9. Industrial Equipment - Coimbatore Hub
    networks["industrial_equipment_coimbatore"] = {
        "id": "industrial_equipment_coimbatore",
        "name": "Representative / Demonstration Supply Network: Pumps, Motors & Heavy Engineering (Coimbatore)",
        "industry": "Industrial Equipment",
        "region": "Western Tamil Nadu",
        "network_depth": 5,
        "nodes": [
            {"id": "sup_specialty_steel", "name": "Specialty Forgings & Alloy Steel (Salem)", "type": "supplier", "is_primary": True, "share_of_supply": 0.65, "reliability": 0.95, "availability": 0.96},
            {"id": "sup_copper_wiring", "name": "Enamelled Copper Winding Wire (Hosur)", "type": "supplier", "is_primary": False, "share_of_supply": 0.35, "reliability": 0.92, "availability": 0.93},
            {"id": "port_cochin", "name": "Vallarpadam Terminal (Cochin Port)", "type": "port", "is_primary": True, "share_of_volume": 0.75, "capacity_utilization": 0.70, "availability": 0.93},
            {"id": "port_tuticorin", "name": "V.O. Chidambaranar Port (Tuticorin)", "type": "port", "is_primary": False, "share_of_volume": 0.25, "capacity_utilization": 0.62, "availability": 0.94},
            {"id": "factory_coimbatore", "name": "Industrial Pump & Motor Works (Coimbatore)", "type": "factory", "capacity_utilization": 0.84, "inventory_days": 24.0, "safety_stock_days": 8.0, "importance_weight": 0.91},
            {"id": "wh_coimbatore", "name": "Central Spares & Sub-Assembly Yard", "type": "warehouse", "capacity_utilization": 0.76, "inventory_days": 28.0, "safety_stock_days": 9.0},
            {"id": "dc_infra_projects", "name": "Infrastructure & EPC Project Sites Hub", "type": "distribution_center", "capacity_utilization": 0.72}
        ],
        "edges": [
            {"id": "e_salem_cbe", "source": "sup_specialty_steel", "target": "factory_coimbatore", "distance_km": 165, "transit_time_days": 0.5, "transport_cost_per_unit": 22.0, "mode": "Road", "is_primary": True},
            {"id": "e_hosur_cbe", "source": "sup_copper_wiring", "target": "factory_coimbatore", "distance_km": 340, "transit_time_days": 1.0, "transport_cost_per_unit": 38.0, "mode": "Road", "is_alternate": True},
            {"id": "e_cochin_cbe", "source": "port_cochin", "target": "factory_coimbatore", "distance_km": 190, "transit_time_days": 0.7, "transport_cost_per_unit": 28.0, "mode": "Road", "is_primary": True},
            {"id": "e_tuti_cbe", "source": "port_tuticorin", "target": "factory_coimbatore", "distance_km": 360, "transit_time_days": 1.1, "transport_cost_per_unit": 42.0, "mode": "Road", "is_alternate": True},
            {"id": "e_cbe_wh", "source": "factory_coimbatore", "target": "wh_coimbatore", "distance_km": 20, "transit_time_days": 0.1, "transport_cost_per_unit": 6.0, "mode": "Road", "is_primary": True},
            {"id": "e_wh_infra", "source": "wh_coimbatore", "target": "dc_infra_projects", "distance_km": 600, "transit_time_days": 1.8, "transport_cost_per_unit": 70.0, "mode": "Road", "is_primary": True}
        ]
    }

    # 10. Industrial Equipment - Ahmedabad Hub
    networks["industrial_equipment_ahmedabad"] = {
        "id": "industrial_equipment_ahmedabad",
        "name": "Representative / Demonstration Supply Network: Heavy Machinery & Valves (Ahmedabad & Sanand)",
        "industry": "Industrial Equipment",
        "region": "Gujarat Heavy Engineering Hub",
        "network_depth": 5,
        "nodes": [
            {"id": "sup_castings_rajkot", "name": "Heavy Castings & Foundry Cluster (Rajkot)", "type": "supplier", "is_primary": True, "share_of_supply": 0.60, "reliability": 0.94, "availability": 0.95},
            {"id": "sup_hydraulics_pune", "name": "High-Pressure Hydraulic Valves (Pune)", "type": "supplier", "is_primary": False, "share_of_supply": 0.40, "reliability": 0.92, "availability": 0.92},
            {"id": "port_mundra_ind", "name": "Mundra Port Engineering Terminal", "type": "port", "is_primary": True, "share_of_volume": 0.75, "capacity_utilization": 0.72, "availability": 0.95},
            {"id": "port_hazira_ind", "name": "Hazira Heavy Bulk Terminal (Surat)", "type": "port", "is_primary": False, "share_of_volume": 0.25, "capacity_utilization": 0.58, "availability": 0.96},
            {"id": "factory_vatva", "name": "Heavy Machinery & Turbine Fabrication (Vatva)", "type": "factory", "capacity_utilization": 0.85, "inventory_days": 26.0, "safety_stock_days": 8.5, "importance_weight": 0.93},
            {"id": "wh_aslali", "name": "Aslali Central Machinery Depot", "type": "warehouse", "capacity_utilization": 0.78, "inventory_days": 30.0, "safety_stock_days": 10.0},
            {"id": "dc_power_mining", "name": "Power & Mining Industrial Dispatch", "type": "distribution_center", "capacity_utilization": 0.70}
        ],
        "edges": [
            {"id": "e_raj_vatva", "source": "sup_castings_rajkot", "target": "factory_vatva", "distance_km": 215, "transit_time_days": 0.7, "transport_cost_per_unit": 28.0, "mode": "Road", "is_primary": True},
            {"id": "e_pune_vatva", "source": "sup_hydraulics_pune", "target": "factory_vatva", "distance_km": 640, "transit_time_days": 1.9, "transport_cost_per_unit": 68.0, "mode": "Road", "is_alternate": True},
            {"id": "e_mundra_vatva", "source": "port_mundra_ind", "target": "factory_vatva", "distance_km": 370, "transit_time_days": 1.1, "transport_cost_per_unit": 38.0, "mode": "Road", "is_primary": True},
            {"id": "e_hazira_vatva", "source": "port_hazira_ind", "target": "factory_vatva", "distance_km": 270, "transit_time_days": 0.8, "transport_cost_per_unit": 32.0, "mode": "Road", "is_alternate": True},
            {"id": "e_vatva_aslali", "source": "factory_vatva", "target": "wh_aslali", "distance_km": 15, "transit_time_days": 0.1, "transport_cost_per_unit": 6.0, "mode": "Road", "is_primary": True},
            {"id": "e_aslali_power", "source": "wh_aslali", "target": "dc_power_mining", "distance_km": 680, "transit_time_days": 2.0, "transport_cost_per_unit": 85.0, "mode": "Road", "is_primary": True}
        ]
    }

    return networks


def save_networks_to_disk():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(base_dir, "data", "networks")
    os.makedirs(target_dir, exist_ok=True)

    networks = create_demo_networks()
    for net_id, net_data in networks.items():
        file_path = os.path.join(target_dir, f"{net_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(net_data, f, indent=2)
        print(f"Saved demo network: {file_path}")

    print(f"\nGenerated {len(networks)} representative networks across 5 core industries successfully.")


if __name__ == "__main__":
    save_networks_to_disk()
