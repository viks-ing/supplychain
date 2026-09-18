"""
Dynamic Multi-Dataset Catalog and Profiler for Supply Chain & Logistics.
Discovers, profiles, and samples from all registered open supply chain datasets
with zero hardcoding of schemas or heuristic rules.
"""

import os
import re
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class DatasetCatalog:
    """
    Catalog of supply chain and logistics disruption datasets.
    Provides automated profiling, candidate target detection, and sample extraction.
    """

    def __init__(self, data_dir: Optional[str] = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = data_dir or os.path.join(base_dir, "data")
        
        # Manifest of known dataset files with metadata descriptions
        self.registry = {
            "dataco": {
                "id": "dataco",
                "filename": "DataCoSupplyChainDataset.csv",
                "name": "Kaggle DataCo Smart Supply Chain",
                "description": "Global supply chain transactions with shipment scheduling, delivery risks, sales, and order profit.",
                "domain": "Enterprise Supply Chain & Fulfillment"
            },
            "olist": {
                "id": "olist",
                "filename": "olist_delivery_risk.csv",
                "name": "Brazilian E-Commerce (Olist)",
                "description": "Commercial marketplace orders with delivery delay hours, freight values, seller grades, and product dimensions.",
                "domain": "E-Commerce Freight & Logistics Delay"
            },
            "delivery_traffic": {
                "id": "delivery_traffic",
                "filename": "delivery_traffic_weather.csv",
                "name": "Delivery Time with Traffic & Weather",
                "description": "Last-mile courier delivery durations with road traffic density, weather conditions, vehicle types, and GPS coordinates.",
                "domain": "Last-Mile & Weather/Traffic Disruptions"
            },
            "gscpi": {
                "id": "gscpi",
                "filename": "gscpi_monthly_disruption_index.csv",
                "name": "Federal Reserve GSCPI Disruption Index",
                "description": "Global Supply Chain Pressure Index monthly time-series measuring cross-border transport and maritime bottlenecks.",
                "domain": "Macroeconomic Supply Chain Stress"
            }
        }

    def list_datasets(self) -> List[Dict[str, Any]]:
        """Lists all registered datasets with existence and size stats."""
        catalog_list = []
        for key, meta in self.registry.items():
            path = os.path.join(self.data_dir, meta["filename"])
            exists = os.path.exists(path)
            size_mb = round(os.path.getsize(path) / (1024 * 1024), 2) if exists else 0.0
            
            catalog_list.append({
                "id": meta["id"],
                "name": meta["name"],
                "filename": meta["filename"],
                "description": meta["description"],
                "domain": meta["domain"],
                "available": exists,
                "size_mb": size_mb
            })
        return catalog_list

    def get_dataset_profile(self, dataset_id: str, nrows: int = 5000) -> Dict[str, Any]:
        """Profiles a specific dataset dynamically without hardcoding."""
        if dataset_id not in self.registry:
            raise ValueError(f"Unknown dataset '{dataset_id}'. Available: {list(self.registry.keys())}")
            
        filename = self.registry[dataset_id]["filename"]
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file '{filename}' not found at {path}")
            
        df = pd.read_csv(path, encoding="latin1", nrows=nrows)
        
        # Dynamically discover column types
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in df.columns if c not in num_cols]
        
        # Dynamically identify candidate targets (risk, delay, cost, time, profit)
        target_pattern = r"(?i)(risk|late|delay|time|profit|cost|price|taken|delivery)"
        candidate_targets = []
        for col in df.columns:
            if re.search(target_pattern, col):
                is_binary = df[col].nunique(dropna=True) <= 2
                task = "classification" if is_binary else "regression"
                candidate_targets.append({
                    "column": col,
                    "task": task,
                    "unique_values": int(df[col].nunique(dropna=True))
                })
                
        return {
            "id": dataset_id,
            "name": self.registry[dataset_id]["name"],
            "total_sampled_rows": len(df),
            "total_columns": len(df.columns),
            "numerical_columns_count": len(num_cols),
            "categorical_columns_count": len(cat_cols),
            "candidate_targets": candidate_targets,
            "sample_columns": list(df.columns)[:15]
        }

    def sample_records(self, dataset_id: str, n_samples: int = 3) -> List[Dict[str, Any]]:
        """Extracts representative empirical sample records from any dataset."""
        if dataset_id not in self.registry:
            raise ValueError(f"Unknown dataset '{dataset_id}'.")
            
        path = os.path.join(self.data_dir, self.registry[dataset_id]["filename"])
        df = pd.read_csv(path, encoding="latin1", nrows=1000)
        sample_df = df.sample(n=min(n_samples, len(df)), random_state=42)
        
        # Convert NaN to None for strict JSON compliance
        clean_records = []
        for r in sample_df.to_dict(orient="records"):
            clean_r = {}
            for k, v in r.items():
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    clean_r[k] = None
                elif pd.isna(v):
                    clean_r[k] = None
                else:
                    clean_r[k] = v
            clean_records.append(clean_r)
        return clean_records
