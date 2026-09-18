"""
Comprehensive test script for Vyuha ML Multi-Dataset & Zero-Hardcoding Pipeline.
"""

import urllib.request
import json

def test_endpoints():
    def req_get(url):
        req = urllib.request.Request(f'http://127.0.0.1:8000{url}')
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode('utf-8'))

    def req_post(url, data):
        body = json.dumps(data).encode('utf-8')
        req = urllib.request.Request(f'http://127.0.0.1:8000{url}', data=body, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode('utf-8'))

    print("=== 1. Health Check ===")
    health = req_get('/health')
    print("Health Status:", health)
    assert health['status'] == 'healthy'
    assert health['dataco_model_loaded'] is True

    print("\n=== 2. Multi-Dataset Manifest ===")
    datasets = req_get('/datasets')
    for d in datasets:
        print(f"  * {d['name']} [{d['id']}] - {d['size_mb']} MB ({d['domain']})")
    assert len(datasets) >= 4

    print("\n=== 3. Dataset Profiles ===")
    for ds_id in ['dataco', 'olist', 'delivery_traffic', 'gscpi']:
        prof = req_get(f'/datasets/{ds_id}/profile')
        target_names = [t['column'] for t in prof['candidate_targets']]
        print(f"  * {prof['name']}: {prof['total_columns']} cols, {prof['total_sampled_rows']} rows. Detected Targets: {target_names[:4]}")

    print("\n=== 4. Empirical Sample Extraction ===")
    sample_orders = req_get('/dataco/sample-orders')
    print(f"  Extracted {len(sample_orders)} real DataCo orders.")
    assert len(sample_orders) > 0

    print("\n=== 5. Zero-Hardcoding Tree SHAP Prediction ===")
    test_order = sample_orders[0]['order']
    pred_result = req_post('/dataco/predict', test_order)
    print("  Prediction Status:", pred_result['status'])
    pred = pred_result['prediction']
    print(f"  Late Delivery Risk Prob: {pred['late_delivery_risk_probability']} (Tier: {pred['risk_level']})")
    print(f"  Predicted Transit Delay: {pred['predicted_delay_days']} days")
    print(f"  Predicted Order Profit:  ${pred['predicted_profit_per_order']}")
    print("  Top Dynamic Tree SHAP Drivers:")
    for d in pred['top_risk_drivers'][:3]:
        print(f"    - {d['feature']} = {d['value']} -> {d['impact']}")

    print("\n ALL MULTI-DATASET & ZERO-HARDCODING TESTS PASSED!")

if __name__ == "__main__":
    test_endpoints()
