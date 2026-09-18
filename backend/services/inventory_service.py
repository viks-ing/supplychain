"""
Inventory Depletion Simulation Service for Vyuha ML.
Projects day-by-day inventory runways under normal and disrupted conditions,
calculating the exact day of depletion / stockout danger point.
"""

from typing import Dict, Any, List

def simulate_inventory_depletion(
    company: Dict[str, Any],
    lead_time_delay_days: float,
    disruption_duration_days: float,
    horizon_days: int = 45
) -> Dict[str, Any]:
    inventory_days = float(company.get("inventory_days", 20.0))
    safety_stock_days = float(company.get("safety_stock_days", 6.0))
    avg_lead_time = float(company.get("avg_lead_time_days", 15.0))
    
    # Daily burn rate is 1.0 "days worth of inventory" per day
    # Replenishment under Normal conditions: orders arrive periodically every avg_lead_time
    # Replenishment under Shock conditions: orders arrive delayed by lead_time_delay_days
    
    normal_curve: List[float] = []
    shocked_curve: List[float] = []
    
    current_norm = inventory_days
    current_shock = inventory_days
    
    stockout_day = None
    safety_stock_breach_day = None
    
    # Batches order cycle assumption: replenishment arrives approximately every avg_lead_time days
    cycle_length = max(7, int(round(avg_lead_time)))
    batch_size = cycle_length # restores cycle_length days worth of stock
    
    for day in range(1, horizon_days + 1):
        # 1. Normal trajectory
        current_norm -= 1.0
        # Scheduled arrival
        if day % cycle_length == 0:
            current_norm += batch_size
        normal_curve.append(round(max(0.0, current_norm), 1))
        
        # 2. Shock trajectory
        current_shock -= 1.0
        
        # In a disruption, shipments are delayed by lead_time_delay_days
        # If disruption is active, delivery is postponed
        effective_arrival_day = cycle_length + int(round(lead_time_delay_days))
        
        if day == effective_arrival_day and day > disruption_duration_days * 0.7:
            # Partial or delayed shipment arrives
            current_shock += batch_size * 0.85
        elif day > effective_arrival_day and (day - effective_arrival_day) % cycle_length == 0:
            # Subsequent staggered replenishment
            current_shock += batch_size
            
        current_shock_bounded = max(0.0, current_shock)
        shocked_curve.append(round(current_shock_bounded, 1))
        
        # Track breach points
        if current_shock_bounded <= safety_stock_days and safety_stock_breach_day is None:
            safety_stock_breach_day = day
            
        if current_shock_bounded <= 0.01 and stockout_day is None:
            stockout_day = day

    # Determine status
    if stockout_day is not None:
        stockout_status = "CRITICAL_STOCKOUT"
        message = f"Critical stockout projected on Day {stockout_day}! Incoming shipments will arrive too late to prevent line stoppage."
    elif safety_stock_breach_day is not None:
        stockout_status = "SAFETY_STOCK_DEPLETED"
        message = f"Safety stock buffer breached on Day {safety_stock_breach_day}. Operations will rely on emergency reserves."
    else:
        stockout_status = "BUFFER_MAINTAINED"
        message = "Inventory buffer remains above safety stock thresholds throughout the projected disruption."
        
    trajectory = []
    for d in range(horizon_days):
        trajectory.append({
            "day": d + 1,
            "normal_inventory": normal_curve[d],
            "shocked_inventory": shocked_curve[d],
            "safety_stock": safety_stock_days
        })
        
    return {
        "stockout_day": stockout_day,
        "safety_stock_breach_day": safety_stock_breach_day,
        "stockout_status": stockout_status,
        "summary": message,
        "initial_inventory_days": inventory_days,
        "safety_stock_days": safety_stock_days,
        "horizon_days": horizon_days,
        "trajectory": trajectory
    }
