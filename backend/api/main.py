"""
FastAPI Application for Vyuha ML.
Exposes endpoints for prediction, scenario simulation, model benchmarks, and presets.
"""

import os
import sys
import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.schemas import PredictRequest, SimulateRequest, CompanyProfile, IndianConditions, DataCoOrderRequest
from services.prediction_service import PredictionService
from services.simulator_service import SimulatorService
from services.dataco_prediction_service import DataCoPredictionService

app = FastAPI(
    title="Vyuha ML - Supply Chain Risk & Shock Prediction Engine",
    description="India-Focused Company Supply Chain Risk & Shock Prediction Simulator API + Kaggle DataCo ML Engine",
    version="2.0.0"
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML Services
try:
    predictor = PredictionService()
    simulator = SimulatorService(predictor)
except Exception as e:
    print(f"Warning: ML models could not be loaded immediately: {e}")
    predictor = None
    simulator = None

try:
    dataco_predictor = DataCoPredictionService()
except Exception as e:
    print(f"Warning: DataCo ML models could not be loaded immediately: {e}")
    dataco_predictor = None

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Vyuha ML Engine",
        "version": "2.0.0",
        "model_loaded": predictor is not None,
        "dataco_model_loaded": dataco_predictor is not None
    }

@app.post("/predict")
def predict_risk(request: PredictRequest):
    if predictor is None:
        raise HTTPException(status_code=500, detail="ML Models are not loaded.")
    
    company_dict = request.company.model_dump()
    cond_dict = request.conditions.model_dump() if request.conditions else {}
    
    try:
        result = predictor.predict(company_dict, cond_dict)
        return {
            "company_name": request.company.company_name,
            "industry": request.company.industry,
            "prediction": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/simulate")
def simulate_scenario(request: SimulateRequest):
    if simulator is None:
        raise HTTPException(status_code=500, detail="Simulation Engine is not loaded.")
        
    company_dict = request.company.model_dump()
    shock_dict = request.shock_parameters.model_dump()
    base_dict = request.baseline_conditions.model_dump() if request.baseline_conditions else None
    
    try:
        result = simulator.simulate(company_dict, shock_dict, base_dict)
        return {
            "company_name": request.company.company_name,
            "simulation": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

@app.get("/model-metrics")
def get_model_metrics():
    artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "artifacts")
    metrics_path = os.path.join(artifacts_dir, "evaluation_metrics.json")
    
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Model metrics have not been generated yet.")
        
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    return metrics

@app.get("/presets/companies")
def get_company_presets():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "companies.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return []

@app.get("/presets/scenarios")
def get_scenario_presets():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "scenarios.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return []

# =====================================================================
# Kaggle DataCo Smart Supply Chain Endpoints
# =====================================================================

@app.post("/dataco/predict")
def predict_dataco_order(order: DataCoOrderRequest):
    if dataco_predictor is None:
        raise HTTPException(status_code=500, detail="DataCo ML Models are not loaded.")
    try:
        order_dict = order.model_dump(by_alias=True)
        result = dataco_predictor.predict(order_dict)
        return {
            "status": "success",
            "model_type": "XGBoost (Kaggle DataCo Smart Supply Chain)",
            "prediction": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DataCo prediction failed: {str(e)}")

@app.get("/dataco/model-metrics")
def get_dataco_model_metrics():
    artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "artifacts")
    metrics_path = os.path.join(artifacts_dir, "dataco_evaluation_metrics.json")
    
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="DataCo model metrics have not been generated yet.")
        
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    return metrics

from data.dataset_catalog import DatasetCatalog

catalog = DatasetCatalog()

@app.get("/datasets")
def list_available_datasets():
    """Lists all registered open supply chain datasets and availability status."""
    return catalog.list_datasets()

@app.get("/datasets/{dataset_id}/profile")
def get_dataset_profile(dataset_id: str):
    """Profiles dataset schema, column counts, and candidate targets dynamically."""
    try:
        return catalog.get_dataset_profile(dataset_id)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/datasets/{dataset_id}/samples")
def get_dataset_samples(dataset_id: str, n: int = 3):
    """Extracts empirical real-world sample records with zero hardcoding."""
    try:
        return catalog.sample_records(dataset_id, n_samples=n)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/dataco/sample-orders")
def get_dataco_sample_orders():
    """Dynamically extracts real sample orders from the DataCo dataset."""
    try:
        raw_samples = catalog.sample_records("dataco", n_samples=4)
        formatted = []
        for i, s in enumerate(raw_samples, 1):
            formatted.append({
                "id": f"dataco_order_{s.get('Order Id', i)}",
                "name": f"DataCo Order #{s.get('Order Id', i)} ({s.get('Shipping Mode', 'Standard')} - {s.get('Market', 'Global')})",
                "order": s
            })
        return formatted
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract samples: {str(e)}")


