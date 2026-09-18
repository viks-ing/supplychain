"""
Verification script for Kaggle DataCo Smart Supply Chain ML endpoints.
"""

import urllib.request
import json

def request_json(url, method='GET', data=None):
    req = urllib.request.Request(url, method=method)
    if data:
        req.add_header('Content-Type', 'application/json')
        payload = json.dumps(data).encode('utf-8')
    else:
        payload = None
    with urllib.request.urlopen(req, data=payload) as response:
        return response.status, json.loads(response.read().decode('utf-8'))

print("=== 1. Testing Health Check ===")
status, res = request_json('http://127.0.0.1:8000/health')
print(f"Status: {status}")
print(f"Engine status: {res}")
assert res["dataco_model_loaded"] is True, "DataCo model not loaded!"

print("\n=== 2. Testing DataCo Model Evaluation Metrics ===")
status, metrics = request_json('http://127.0.0.1:8000/dataco/model-metrics')
print(f"Status: {status}")
print(f"Dataset: {metrics['dataset_name']} ({metrics['dataset_size']:,} rows)")
print(f"Features: {metrics['features_count']} features ({', '.join(metrics['feature_names'][:6])}...)")

late_metrics = metrics['models']['late_delivery_risk']
print("\nXGBoost Late Delivery Risk Classifier:")
print(f" - Accuracy:  {late_metrics['accuracy'] * 100:.2f}%")
print(f" - ROC-AUC:   {late_metrics['roc_auc']:.4f}")
print(f" - Precision: {late_metrics['precision']:.4f}")
print(f" - Recall:    {late_metrics['recall']:.4f}")
print(f" - F1-Score:  {late_metrics['f1_score']:.4f}")

delay_metrics = metrics['models']['delivery_delay_days']
print("\nXGBoost Delivery Delay Regressor:")
print(f" - RMSE: {delay_metrics['rmse']} days")
print(f" - MAE:  {delay_metrics['mae']} days")
print(f" - R2:   {delay_metrics['r2']}")

print("\nTop 5 Predictive Features for Late Delivery Risk:")
for fi in metrics['feature_importances']['late_delivery_risk'][:5]:
    print(f" * {fi['feature']}: {fi['importance'] * 100:.1f}%")

print("\n=== 3. Testing Sample Orders Retrieval ===")
status, samples = request_json('http://127.0.0.1:8000/dataco/sample-orders')
print(f"Status: {status}, Retrieved {len(samples)} sample orders:")
for s in samples:
    print(f" - [{s['id']}] {s['name']}")

print("\n=== 4. Testing POST /dataco/predict on Sample Orders ===")
for s in samples:
    status, pred = request_json('http://127.0.0.1:8000/dataco/predict', method='POST', data=s['order'])
    p = pred['prediction']
    print(f"\nOrder: {s['name']}")
    print(f"  Summary: {p['order_summary']}")
    print(f"  Late Delivery Risk: {p['late_delivery_risk_score']}% ({p['risk_level']}) | Probability: {p['late_delivery_risk_probability']}")
    print(f"  Predicted Transit Delay: {p['predicted_delay_days']} day(s)")
    print(f"  Predicted Benefit/Profit: ${p['predicted_profit_per_order']}")
    print(f"  Mathematical Tree SHAP Drivers ({len(p['top_risk_drivers'])}):")
    for d in p['top_risk_drivers']:
        print(f"   • {d['feature']} = {d['value']} -> {d['impact']}")

print("\n>>> ALL KAGGLE DATACO ML VERIFICATIONS PASSED SUCCESSFULLY! <<<")
