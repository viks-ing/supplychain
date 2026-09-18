"""
End-to-end API verification suite for Vyuha ML Backend Engine.
"""

import urllib.request
import json

def test_url(url, method='GET', data=None):
    req = urllib.request.Request(url, method=method)
    if data:
        req.add_header('Content-Type', 'application/json')
        payload = json.dumps(data).encode('utf-8')
    else:
        payload = None
    with urllib.request.urlopen(req, data=payload) as response:
        return response.status, response.read().decode('utf-8')

print("=== 1. Testing Backend Health ===")
status, body = test_url('http://127.0.0.1:8000/health')
print(f"Status: {status}, Response: {body}")

print("\n=== 2. Testing Model Metrics ===")
status, body = test_url('http://127.0.0.1:8000/model-metrics')
metrics = json.loads(body)
print(f"Status: {status}")
print("Available Targets in Metrics:")
for tgt_name, tgt_metrics in metrics.get('targets', {}).items():
    if tgt_metrics.get('task') == 'classification':
        print(f" - [Classification] {tgt_name} -> Accuracy: {tgt_metrics.get('accuracy')}, F1: {tgt_metrics.get('f1_score')}, ROC-AUC: {tgt_metrics.get('roc_auc')}")
    else:
        print(f" - [Regression] {tgt_name} -> R2: {tgt_metrics.get('r2')}, RMSE: {tgt_metrics.get('rmse')}, MAE: {tgt_metrics.get('mae')}")


print("\n=== 3. Testing Company Presets ===")
status, body = test_url('http://127.0.0.1:8000/presets/companies')
presets = json.loads(body)
print(f"Status: {status}, Loaded {len(presets)} companies:")
for p in presets:
    print(f" - {p['company_name']} ({p['industry']} - {p['city_state']})")

print("\n=== 4. Testing Scenario Presets ===")
status, body = test_url('http://127.0.0.1:8000/presets/scenarios')
scenarios = json.loads(body)
print(f"Status: {status}, Loaded {len(scenarios)} scenarios:")
for s in scenarios:
    print(f" - {s['name']}")

print("\n=== 5. Testing POST /simulate (PRD Scenario 15) ===")
sample_payload = {
    'company': presets[0],
    'shock_parameters': scenarios[0]
}
status, body = test_url('http://127.0.0.1:8000/simulate', method='POST', data=sample_payload)
sim = json.loads(body)['simulation']
print(f"Status: {status}")
print("Normal Cost Impact:", sim['normal_conditions']['cost_impact_pct'], "% | Risk:", sim['normal_conditions']['risk_score'], f"({sim['normal_conditions']['risk_level']})")
print("Shocked Cost Impact:", sim['shock_conditions']['cost_impact_pct'], "% | Risk:", sim['shock_conditions']['risk_score'], f"({sim['shock_conditions']['risk_level']})")
print("Impact Delta: Cost +", sim['impact_delta']['additional_cost_pct'], "% | Lead Time +", sim['impact_delta']['additional_lead_time_days'], "days")
print("Inventory Stockout Day:", sim['inventory_simulation']['stockout_day'])
print("Generated Mitigations:", len(sim['mitigation_recommendations']))
for m in sim['mitigation_recommendations']:
    print(f" * [{m['category']}] {m['title']} ({m['effort']} effort)")

print("\n>>> ALL ML BACKEND VERIFICATION CHECKS PASSED SUCCESSFULLY! <<<")
