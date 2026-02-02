"""
Feature extraction from observation metadata.

Provides utilities to convert MAST observation metadata into feature vectors
suitable for machine learning.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any


class MetadataFeaturizer:
    """
    Extract and encode features from observation metadata.
    
    Handles various data types:
    - Numeric features (exposure time, etc.)
    - Categorical features (instrument, filters, etc.)
    - Text features (target name, proposal description)
    
    Parameters
    ----------
    categorical_columns : list of str, optional
        Columns to treat as categorical.
    numeric_columns : list of str, optional
        Columns to treat as numeric.
    """
    
    def __init__(
        self,
        categorical_columns: Optional[List[str]] = None,
        numeric_columns: Optional[List[str]] = None
    ):
        self.categorical_columns = categorical_columns or []
        self.numeric_columns = numeric_columns or []
        
        # Learned vocabularies
        self.vocabularies_ = {}
        self.feature_names_ = []
    
    def fit(self, df: pd.DataFrame) -> "MetadataFeaturizer":
        """
        Fit the featurizer on observation metadata.
        
        Parameters
        ----------
        df : pd.DataFrame
            Observation metadata.
            
        Returns
        -------
        self : MetadataFeaturizer
        """
        # Build vocabularies for categorical columns
        for col in self.categorical_columns:
            if col in df.columns:
                self.vocabularies_[col] = df[col].fillna("missing").unique().tolist()
        
        # Build feature names
        self.feature_names_ = []
        
        for col in self.numeric_columns:
            if col in df.columns:
                self.feature_names_.append(f"num_{col}")
        
        for col in self.categorical_columns:
            if col in df.columns:
                vocab = self.vocabularies_[col]
                for value in vocab:
                    self.feature_names_.append(f"cat_{col}_{value}")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform observation metadata to feature vectors.
        
        Parameters
        ----------
        df : pd.DataFrame
            Observation metadata.
            
        Returns
        -------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        """
        features = []
        
        # Numeric features
        for col in self.numeric_columns:
            if col in df.columns:
                values = pd.to_numeric(df[col], errors='coerce').fillna(0).values
                features.append(values.reshape(-1, 1))
        
        # Categorical features (one-hot encoding)
        for col in self.categorical_columns:
            if col in df.columns:
                vocab = self.vocabularies_.get(col, [])
                encoded = np.zeros((len(df), len(vocab)))
                
                col_values = df[col].fillna("missing").values
                for i, value in enumerate(col_values):
                    if value in vocab:
                        j = vocab.index(value)
                        encoded[i, j] = 1.0
                
                features.append(encoded)
        
        if features:
            X = np.hstack(features)
        else:
            X = np.zeros((len(df), 0))
        
        return X
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform observation metadata.
        
        Parameters
        ----------
        df : pd.DataFrame
            Observation metadata.
            
        Returns
        -------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        """
        return self.fit(df).transform(df)


def get_default_featurizer() -> MetadataFeaturizer:
    """
    Get a default featurizer with common HST metadata columns.
    
    Returns
    -------
    featurizer : MetadataFeaturizer
        Configured featurizer.
    """
    return MetadataFeaturizer(
        numeric_columns=[
            "t_exptime",  # Exposure time
            "t_min",      # Start time
            "t_max",      # End time
        ],
        categorical_columns=[
            "instrument_name",
            "filters",
            "dataproduct_type",
            "obs_collection",
        ]
    )
