"""
Advanced feature engineering for astronomy data.

Provides advanced transformation and feature generation techniques.
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for astronomical observations.
    
    Combines multiple transformation techniques:
    - Polynomial features for interaction terms
    - PCA for dimensionality reduction
    - Statistical aggregations
    - Domain-specific astronomy features
    
    Parameters
    ----------
    polynomial_degree : int, default=2
        Degree of polynomial features.
    n_pca_components : int, optional
        Number of PCA components. If None, no PCA is applied.
    scale_features : bool, default=True
        Whether to standardize features.
    """
    
    def __init__(
        self,
        polynomial_degree: int = 2,
        n_pca_components: Optional[int] = None,
        scale_features: bool = True
    ):
        self.polynomial_degree = polynomial_degree
        self.n_pca_components = n_pca_components
        self.scale_features = scale_features
        
        self.scaler_ = StandardScaler() if scale_features else None
        self.poly_ = PolynomialFeatures(
            degree=polynomial_degree,
            include_bias=False
        ) if polynomial_degree > 1 else None
        self.pca_ = PCA(n_components=n_pca_components) if n_pca_components else None
        
        self.feature_names_ = []
        self.is_fitted_ = False
    
    def fit(self, X: np.ndarray) -> "AdvancedFeatureEngineer":
        """
        Fit the feature engineer.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features.
            
        Returns
        -------
        self : AdvancedFeatureEngineer
        """
        X = np.asarray(X)
        
        # Scale features
        if self.scaler_:
            X = self.scaler_.fit_transform(X)
        
        # Generate polynomial features
        if self.poly_:
            X = self.poly_.fit_transform(X)
        
        # Apply PCA
        if self.pca_:
            self.pca_.fit(X)
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features.
            
        Returns
        -------
        X_transformed : np.ndarray
            Transformed features.
        """
        if not self.is_fitted_:
            raise ValueError("AdvancedFeatureEngineer must be fitted before transform")
        
        X = np.asarray(X)
        
        # Scale features
        if self.scaler_:
            X = self.scaler_.transform(X)
        
        # Generate polynomial features
        if self.poly_:
            X = self.poly_.transform(X)
        
        # Apply PCA
        if self.pca_:
            X = self.pca_.transform(X)
        
        return X
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform features.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features.
            
        Returns
        -------
        X_transformed : np.ndarray
            Transformed features.
        """
        return self.fit(X).transform(X)


def create_time_features(df: pd.DataFrame, time_column: str = 't_min') -> pd.DataFrame:
    """
    Create time-based features from observation timestamps.
    
    Parameters
    ----------
    df : pd.DataFrame
        Observation metadata with time column.
    time_column : str, default='t_min'
        Name of the time column.
        
    Returns
    -------
    df_features : pd.DataFrame
        DataFrame with additional time features.
    """
    df = df.copy()
    
    if time_column in df.columns:
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
            df[time_column] = pd.to_datetime(df[time_column], unit='s', errors='coerce')
        
        # Extract time components
        df['obs_year'] = df[time_column].dt.year
        df['obs_month'] = df[time_column].dt.month
        df['obs_day_of_year'] = df[time_column].dt.dayofyear
        df['obs_hour'] = df[time_column].dt.hour
        
        # Cyclical encoding for month
        df['obs_month_sin'] = np.sin(2 * np.pi * df['obs_month'] / 12)
        df['obs_month_cos'] = np.cos(2 * np.pi * df['obs_month'] / 12)
    
    return df


def create_exposure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create exposure-based features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Observation metadata with exposure time.
        
    Returns
    -------
    df_features : pd.DataFrame
        DataFrame with additional exposure features.
    """
    df = df.copy()
    
    if 't_exptime' in df.columns:
        # Log transform exposure time
        df['log_exptime'] = np.log1p(df['t_exptime'])
        
        # Binned exposure categories
        df['exptime_category'] = pd.cut(
            df['t_exptime'],
            bins=[0, 100, 300, 1000, np.inf],
            labels=['short', 'medium', 'long', 'very_long']
        )
    
    return df


def create_instrument_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create instrument-specific features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Observation metadata with instrument information.
        
    Returns
    -------
    df_features : pd.DataFrame
        DataFrame with additional instrument features.
    """
    df = df.copy()
    
    if 'instrument_name' in df.columns:
        # Extract main instrument (before /)
        df['main_instrument'] = df['instrument_name'].str.split('/').str[0]
        
        # Extract detector (after /)
        df['detector'] = df['instrument_name'].str.split('/').str[1]
        
        # Flag for specific instruments
        df['is_acs'] = df['instrument_name'].str.contains('ACS', na=False)
        df['is_wfc3'] = df['instrument_name'].str.contains('WFC3', na=False)
    
    return df
