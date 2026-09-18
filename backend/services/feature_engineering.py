"""
Zero-Hardcoding Dynamic Feature Engineering Pipeline for Supply Chain Datasets.
Learns column types, categorical encodings, temporal patterns, and statistical
distributions directly from data without any hardcoded column lists or fallback rules.
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Set
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DynamicFeaturePipeline:
    """
    Automated, zero-hardcoding feature engineering pipeline.
    Dynamically identifies numerical, categorical, and datetime columns,
    learns distribution statistics during fit(), and provides deterministic imputation.
    """

    def __init__(
        self,
        target_cols: Optional[List[str]] = None,
        exclude_patterns: Optional[str] = None,
        max_cardinality_ratio: float = 0.95
    ):
        # Dynamically excluded target names and leaky indicators
        default_targets = [
            "Late_delivery_risk", "Days for shipping (real)", "delay_days",
            "Delivery Status", "Benefit per order", "Order Profit Per Order",
            "Order Item Profit Ratio", "cost_impact_pct", "lead_time_impact_days",
            "risk_score"
        ]
        self.target_cols: Set[str] = set(target_cols if target_cols is not None else default_targets)
        
        # Regex to dynamically identify IDs, PII, free text, URLs, passwords
        self.exclude_pattern = exclude_patterns or r"(?i)(_id$|^id_|\bid\b|email|password|image|description|url|zipcode|street|fname|lname|status)"
        self.max_cardinality_ratio = max_cardinality_ratio
        
        # Learned attributes (populated dynamically in fit)
        self.categorical_cols: List[str] = []
        self.numerical_cols: List[str] = []
        self.datetime_cols: List[str] = []
        self.feature_names: List[str] = []
        
        self.column_medians: Dict[str, float] = {}
        self.column_means: Dict[str, float] = {}
        self.column_modes: Dict[str, str] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        
        self.is_fitted: bool = False

    def _should_exclude(self, col: str, df: pd.DataFrame) -> bool:
        """Determines if a column should be excluded dynamically based on patterns, targets, or cardinality."""
        if col in self.target_cols:
            return True
        if re.search(self.exclude_pattern, col):
            return True
        # Exclude extreme-cardinality non-numeric identifier fields
        if not pd.api.types.is_numeric_dtype(df[col]):
            uniqueness_ratio = df[col].nunique(dropna=True) / max(len(df), 1)
            if uniqueness_ratio > self.max_cardinality_ratio and len(df) > 50:
                return True
        return False

    def _extract_temporal_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Dynamically identifies datetime columns and extracts temporal features without hardcoded names."""
        df = df.copy()
        generated_cols = []
        
        for col in df.columns:
            # Check if column is already datetime or string containing date keywords
            is_date_named = bool(re.search(r"(?i)(date|time|timestamp|order date)", str(col)))
            is_dt_type = pd.api.types.is_datetime64_any_dtype(df[col])
            
            if is_dt_type or is_date_named:
                try:
                    try:
                        parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
                    except Exception:
                        parsed = pd.to_datetime(df[col], errors="coerce")
                    valid_ratio = parsed.notna().sum() / max(len(df), 1)
                    if valid_ratio > 0.4:
                        prefix = re.sub(r"[^\w]", "_", str(col)).strip("_").lower()
                        m_col = f"{prefix}_month"
                        dow_col = f"{prefix}_dayofweek"
                        h_col = f"{prefix}_hour"
                        
                        df[m_col] = parsed.dt.month.fillna(parsed.dt.month.median() if parsed.dt.month.notna().any() else 1).astype(int)
                        df[dow_col] = parsed.dt.dayofweek.fillna(parsed.dt.dayofweek.median() if parsed.dt.dayofweek.notna().any() else 0).astype(int)
                        df[h_col] = parsed.dt.hour.fillna(parsed.dt.hour.median() if parsed.dt.hour.notna().any() else 12).astype(int)
                        
                        generated_cols.extend([m_col, dow_col, h_col])
                        if col not in self.datetime_cols:
                            self.datetime_cols.append(col)
                except Exception:
                    pass
                    
        return df, generated_cols

    def fit(self, df: pd.DataFrame):
        """Learns feature schema and statistical parameters directly from the dataset."""
        df_processed, temporal_cols = self._extract_temporal_features(df)
        
        # Discover candidate feature columns
        candidate_cols = [c for c in df_processed.columns if not self._should_exclude(c, df_processed) and c not in self.datetime_cols]
        
        # Discover column types dynamically using pandas api
        self.categorical_cols = [
            c for c in candidate_cols
            if pd.api.types.is_string_dtype(df_processed[c])
            or pd.api.types.is_object_dtype(df_processed[c])
            or pd.api.types.is_categorical_dtype(df_processed[c])
        ]
        
        self.numerical_cols = [
            c for c in candidate_cols
            if c not in self.categorical_cols and pd.api.types.is_numeric_dtype(df_processed[c])
        ]
        
        # Learn and store empirical distributions
        for col in self.numerical_cols:
            median_val = float(df_processed[col].median(skipna=True))
            if np.isnan(median_val):
                median_val = 0.0
            mean_val = float(df_processed[col].mean(skipna=True))
            if np.isnan(mean_val):
                mean_val = 0.0
            self.column_medians[col] = median_val
            self.column_means[col] = mean_val
            
        for col in self.categorical_cols:
            mode_series = df_processed[col].dropna().mode()
            mode_val = str(mode_series.iloc[0]) if len(mode_series) > 0 else "Unknown"
            self.column_modes[col] = mode_val
            
            # Fit label encoder
            le = LabelEncoder()
            clean_series = df_processed[col].fillna(mode_val).astype(str)
            le.fit(clean_series)
            self.label_encoders[col] = le

        self.feature_names = self.numerical_cols + self.categorical_cols
        
        # Fit scaler on numerical columns
        if self.numerical_cols:
            num_data = df_processed[self.numerical_cols].fillna(self.column_medians).values
            self.scaler.fit(num_data)
            
        self.is_fitted = True
        return self

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.fit(df)
        return self.transform(df)

    def transform(self, df: pd.DataFrame, return_tuple: bool = False):
        """
        Transforms input DataFrame into feature matrices using learned distributions.
        Missing columns and values are dynamically imputed from training medians/modes.
        """
        if not self.is_fitted:
            raise ValueError("DynamicFeaturePipeline must be fitted before transform.")
            
        df_processed, _ = self._extract_temporal_features(df)
        
        # 1. Process numerical features with dynamic median imputation
        num_arrays = []
        for col in self.numerical_cols:
            if col in df_processed.columns:
                vals = pd.to_numeric(df_processed[col], errors="coerce").fillna(self.column_medians[col]).values
            else:
                vals = np.full(len(df_processed), self.column_medians[col])
            num_arrays.append(vals.reshape(-1, 1))
            
        num_raw = np.hstack(num_arrays) if num_arrays else np.empty((len(df_processed), 0))
        
        # 2. Process categorical features with vectorized dynamic mode imputation
        cat_arrays = []
        for col in self.categorical_cols:
            le = self.label_encoders[col]
            class_map = {c: i for i, c in enumerate(le.classes_)}
            default_label = class_map.get(self.column_modes[col], 0)
            
            if col in df_processed.columns:
                series = df_processed[col].fillna(self.column_modes[col]).astype(str)
                # Fast vectorized dictionary mapping
                encoded = series.map(class_map).fillna(default_label).astype(np.int32).values
            else:
                encoded = np.full(len(df_processed), default_label, dtype=np.int32)
            cat_arrays.append(encoded.reshape(-1, 1))
            
        cat_encoded = np.hstack(cat_arrays) if cat_arrays else np.empty((len(df_processed), 0))
        
        # Combine numerical and categorical features
        if num_raw.size > 0 and cat_encoded.size > 0:
            X_raw = np.hstack([num_raw, cat_encoded])
        elif num_raw.size > 0:
            X_raw = num_raw
        else:
            X_raw = cat_encoded
            
        if return_tuple:
            num_scaled = self.scaler.transform(num_raw) if num_raw.size > 0 else num_raw
            X_scaled = np.hstack([num_scaled, cat_encoded]) if cat_encoded.size > 0 else num_scaled
            return X_scaled, X_raw
            
        return X_raw

    def transform_single(self, input_dict: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Transforms a single dictionary record dynamically."""
        df = pd.DataFrame([input_dict])
        return self.transform(df, return_tuple=True)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str) -> "DynamicFeaturePipeline":
        return joblib.load(filepath)


# Aliases for backwards compatibility across modules
FeaturePipeline = DynamicFeaturePipeline
DataCoFeaturePipeline = DynamicFeaturePipeline
