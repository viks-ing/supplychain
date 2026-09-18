"""
Kaggle DataCo Dynamic Feature Engineering Pipeline.
Uses DynamicFeaturePipeline for automated schema discovery, zero hardcoding,
and empirical distribution learning.
"""

from services.feature_engineering import DynamicFeaturePipeline, DynamicFeaturePipeline as DataCoFeaturePipeline

__all__ = ["DynamicFeaturePipeline", "DataCoFeaturePipeline"]
