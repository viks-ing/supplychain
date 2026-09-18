# Vyuha ML (v2.0) — Dynamic Machine Learning Backend Engine
## Zero-Hardcoding Architecture (Kaggle DataCo + Indian Logistics Edition)

Vyuha ML is a fully data-driven, mathematically explainable AI/ML supply chain prediction backend. **No heuristics, feature weights, column mappings, or sample orders are hardcoded.**

---

## 1. Zero-Hardcoding Design Principles

### A. Dynamic Feature Discovery & Distribution Learning
- Feature types (numerical vs categorical) are discovered dynamically from dataframe schemas using `pd.api.types`.
- Target variables and post-delivery leaky fields are dynamically filtered.
- Numerical medians and categorical modes are learned and stored during `fit_transform()`. During inference, any missing field is automatically imputed from the learned statistical distribution.

### B. Mathematical Tree SHAP Attribution (Zero Rule-Based Heuristics)
- Feature attributions and root-cause risk drivers are **not** evaluated through manual `if-else` rules.
- The engine uses native XGBoost Tree SHAP (`booster.predict(dmat, pred_contribs=True)`).
- Every prediction reports the exact directional margin contribution of each feature towards delay risk (e.g. `Shipping Mode = First Class -> SHAP +3.225 (increased delay risk)` or `order_hour = 6 -> SHAP -1.997 (reduced delay risk)`).

### C. Externalized Configurations & Real-Data Sampling
- Sample orders are sampled dynamically from `DataCoSupplyChainDataset.csv` (180,519 rows).
- Company profiles, shock scenarios, and logistics port corridors are loaded dynamically from `backend/config/` (`companies.json`, `scenarios.json`, `logistics_corridors.json`).

---

## 2. Directory Layout

```
backend/
├── data/
│   ├── DataCoSupplyChainDataset.csv # 180,519-row Kaggle DataCo dataset (~91.5 MB)
│   ├── generator.py                 # Indian industrial dataset generator
│   └── indian_supply_chain.csv      # Calibrated Indian supply chain dataset
├── config/
│   ├── companies.json               # External company profiles catalog
│   ├── scenarios.json               # External disruption scenarios catalog
│   └── logistics_corridors.json     # Dynamic port corridors & rail connections
├── models/
│   ├── train_dataco.py              # Kaggle DataCo XGBoost training script
│   ├── train.py                     # Indian supply chain training script (Ridge, RF, XGBoost)
│   └── artifacts/                   # Serialized production models & metrics
│       ├── dataco_late_delivery_xgb.joblib
│       ├── dataco_delay_days_xgb.joblib
│       ├── dataco_profit_xgb.joblib
│       ├── dataco_feature_pipeline.joblib
│       ├── dataco_evaluation_metrics.json
│       ├── cost_impact_pct_xgb.joblib
│       ├── lead_time_impact_days_xgb.joblib
│       ├── risk_score_xgb.joblib
│       ├── feature_pipeline.joblib
│       └── evaluation_metrics.json
├── services/
│   ├── dataco_feature_engineering.py# Dynamic feature pipeline with distribution learning
│   ├── dataco_prediction_service.py # Tree SHAP inference engine for DataCo orders
│   ├── feature_engineering.py       # Domain feature pipeline for Indian companies
│   ├── prediction_service.py        # Tree SHAP multi-target inference engine
│   ├── simulator_service.py         # What-If scenario comparison engine
│   ├── vulnerability_service.py     # SHAP-driven 5-factor risk decomposition
│   ├── inventory_service.py         # 45-day daily inventory runway simulation
│   └── mitigation_service.py        # Dynamic corridor & playbook mitigation engine
├── api/
│   ├── schemas.py                   # Pydantic validation schemas
│   └── main.py                      # FastAPI application
├── test_dataco.py                   # DataCo ML test suite
├── test_backend.py                  # Indian supply chain scenario test suite
├── verify_endpoints.py              # End-to-end verification suite
├── requirements.txt                 # Dependencies
└── run.py                           # FastAPI launcher (http://127.0.0.1:8000)
```

---

## 3. Kaggle DataCo XGBoost Benchmarks (36,104 Test Orders)

- **Late Delivery Risk Classifier (`XGBClassifier`)**:
  - Accuracy: **74.04%**
  - ROC-AUC: **0.8427**
  - Precision: **0.8516**
  - F1-Score: **0.7293**
- **Delivery Delay Days Regressor (`XGBRegressor`)**:
  - MAE: **0.938 days**
  - RMSE: **1.210 days**
  - R²: **0.3331**

### Top Global Features Learned:
1. `Days for shipment (scheduled)`: **49.0%**
2. `Shipping Mode`: **28.8%**
3. `Type`: **3.4%**
4. `order_hour`: **3.0%**
5. `Customer Country`: **0.8%**

---

## 4. Running & Verification

```bash
# Start FastAPI service
python backend/run.py

# Run DataCo Tree SHAP verification
python backend/test_dataco.py

# Run Indian supply chain verification
python backend/verify_endpoints.py
```
- API Endpoint: `http://127.0.0.1:8000`
- Interactive Swagger UI: `http://127.0.0.1:8000/docs`
