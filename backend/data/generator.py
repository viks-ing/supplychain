"""
Dynamic Data Loader and Empirical Sampler for Kaggle DataCo Supply Chain Dataset.
Replaces static synthetic heuristic generation with automated, data-driven empirical sampling
directly from the 180,519-record DataCo dataset with zero hardcoded formulas or profiles.
"""

import os
import argparse
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class DynamicDataSampler:
    """
    Data-driven loader and empirical sampler for supply chain datasets.
    Derives real data distributions dynamically without hardcoded constants.
    """

    def __init__(self, data_path: Optional[str] = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_path = data_path or os.path.join(base_dir, "data", "DataCoSupplyChainDataset.csv")
        self._df: Optional[pd.DataFrame] = None

    def load_data(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """Loads the real supply chain dataset with dynamic schema caching."""
        if self._df is None or (nrows is not None and len(self._df) < nrows):
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Supply chain dataset not found at: {self.data_path}")
            print(f"Loading data from {self.data_path}...")
            self._df = pd.read_csv(self.data_path, encoding="latin1", nrows=nrows)
        return self._df

    def get_column_summary(self) -> Dict[str, Any]:
        """Dynamically computes statistical summaries for all numeric and categorical columns."""
        df = self.load_data()
        summary = {
            "total_records": len(df),
            "columns_count": len(df.columns),
            "numerical_summary": df.describe().to_dict(),
            "categorical_uniques": {
                col: int(df[col].nunique(dropna=True))
                for col in df.select_dtypes(include=["object", "category"]).columns
            }
        }
        return summary

    def sample_real_orders(
        self,
        n_samples: int = 5,
        stratify_by: Optional[str] = "Shipping Mode"
    ) -> List[Dict[str, Any]]:
        """
        Extracts representative real-world orders stratified by the given column.
        Zero hardcoded order fields or synthetic formulas.
        """
        df = self.load_data()
        samples = []
        
        if stratify_by and stratify_by in df.columns:
            groups = df.groupby(stratify_by)
            for name, group in groups:
                row = group.sample(n=1, random_state=42).iloc[0]
                samples.append(row.to_dict())
                if len(samples) >= n_samples:
                    break
        else:
            sample_df = df.sample(n=min(n_samples, len(df)), random_state=42)
            samples = sample_df.to_dict(orient="records")
            
        return samples

    def generate_bootstrap_sample(self, n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
        """
        Generates empirical bootstrap samples from the real Kaggle DataCo distribution
        without imposing any arbitrary or hardcoded mathematical formulas.
        """
        df = self.load_data()
        bootstrap_df = df.sample(n=n_samples, replace=True, random_state=random_state)
        return bootstrap_df.reset_index(drop=True)


def load_dataset(data_path: Optional[str] = None) -> pd.DataFrame:
    """Convenience function to load the primary dataset."""
    sampler = DynamicDataSampler(data_path=data_path)
    return sampler.load_data()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DataCo Dynamic Data Loader & Empirical Sampler")
    parser.add_argument("--samples", type=int, default=5, help="Number of real sample orders to extract")
    parser.add_argument("--stratify", type=str, default="Shipping Mode", help="Column to stratify by")
    args = parser.parse_args()
    
    sampler = DynamicDataSampler()
    sample_records = sampler.sample_real_orders(n_samples=args.samples, stratify_by=args.stratify)
    print(f"\nExtracted {len(sample_records)} empirical orders from real DataCo dataset (Stratified by '{args.stratify}'):")
    for i, rec in enumerate(sample_records, 1):
        order_id = rec.get("Order Id", i)
        ship_mode = rec.get("Shipping Mode", "N/A")
        market = rec.get("Market", "N/A")
        late_risk = rec.get("Late_delivery_risk", "N/A")
        print(f"  [{i}] Order #{order_id} | Mode: {ship_mode} | Market: {market} | Late Delivery Risk: {late_risk}")
