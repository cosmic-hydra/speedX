"""Tests for feature extraction."""
import numpy as np
import pandas as pd
import pytest
from speedx.features import MetadataFeaturizer, get_default_featurizer


@pytest.fixture
def sample_metadata():
    """Create sample observation metadata."""
    return pd.DataFrame({
        "obs_id": ["obs_1", "obs_2", "obs_3", "obs_4"],
        "instrument_name": ["ACS/WFC", "ACS/WFC", "WFC3/UVIS", "WFC3/IR"],
        "filters": ["F606W", "F814W", "F606W", "F110W"],
        "t_exptime": [100.0, 200.0, 150.0, 300.0],
        "dataproduct_type": ["image", "image", "image", "spectrum"],
        "obs_collection": ["HST", "HST", "HST", "HST"]
    })


def test_featurizer_initialization():
    """Test featurizer initialization."""
    featurizer = MetadataFeaturizer(
        categorical_columns=["instrument_name", "filters"],
        numeric_columns=["t_exptime"]
    )
    
    assert featurizer.categorical_columns == ["instrument_name", "filters"]
    assert featurizer.numeric_columns == ["t_exptime"]


def test_featurizer_fit(sample_metadata):
    """Test featurizer fitting."""
    featurizer = MetadataFeaturizer(
        categorical_columns=["instrument_name"],
        numeric_columns=["t_exptime"]
    )
    
    featurizer.fit(sample_metadata)
    
    # Check vocabulary
    assert "instrument_name" in featurizer.vocabularies_
    vocab = featurizer.vocabularies_["instrument_name"]
    assert "ACS/WFC" in vocab
    assert "WFC3/UVIS" in vocab
    assert "WFC3/IR" in vocab


def test_featurizer_transform(sample_metadata):
    """Test featurizer transformation."""
    featurizer = MetadataFeaturizer(
        categorical_columns=["instrument_name"],
        numeric_columns=["t_exptime"]
    )
    
    featurizer.fit(sample_metadata)
    X = featurizer.transform(sample_metadata)
    
    # Check shape
    n_numeric = 1
    n_categorical = len(featurizer.vocabularies_["instrument_name"])
    expected_features = n_numeric + n_categorical
    
    assert X.shape == (4, expected_features)


def test_featurizer_fit_transform(sample_metadata):
    """Test combined fit and transform."""
    featurizer = MetadataFeaturizer(
        categorical_columns=["instrument_name"],
        numeric_columns=["t_exptime"]
    )
    
    X = featurizer.fit_transform(sample_metadata)
    
    assert X.shape[0] == 4
    assert X.shape[1] > 0


def test_featurizer_handles_missing_values():
    """Test that featurizer handles missing values."""
    df = pd.DataFrame({
        "instrument_name": ["ACS/WFC", None, "WFC3/UVIS"],
        "t_exptime": [100.0, None, 200.0]
    })
    
    featurizer = MetadataFeaturizer(
        categorical_columns=["instrument_name"],
        numeric_columns=["t_exptime"]
    )
    
    X = featurizer.fit_transform(df)
    
    # Should handle NaN without errors
    assert X.shape == (3, 4)  # 1 numeric + 3 categorical (including "missing")
    assert np.isfinite(X).all()


def test_featurizer_numeric_only():
    """Test featurizer with only numeric columns."""
    df = pd.DataFrame({
        "t_exptime": [100.0, 200.0, 150.0],
        "t_min": [0.0, 1.0, 2.0],
        "t_max": [10.0, 11.0, 12.0]
    })
    
    featurizer = MetadataFeaturizer(
        numeric_columns=["t_exptime", "t_min", "t_max"]
    )
    
    X = featurizer.fit_transform(df)
    
    assert X.shape == (3, 3)
    np.testing.assert_array_almost_equal(X[:, 0], [100.0, 200.0, 150.0])


def test_featurizer_categorical_only():
    """Test featurizer with only categorical columns."""
    df = pd.DataFrame({
        "instrument_name": ["ACS/WFC", "WFC3/UVIS", "ACS/WFC"],
        "filters": ["F606W", "F814W", "F606W"]
    })
    
    featurizer = MetadataFeaturizer(
        categorical_columns=["instrument_name", "filters"]
    )
    
    X = featurizer.fit_transform(df)
    
    # Should have one-hot encoding
    assert X.shape[0] == 3
    # Each categorical column contributes its vocabulary size
    assert X.shape[1] > 0


def test_featurizer_one_hot_encoding():
    """Test that categorical features are one-hot encoded correctly."""
    df = pd.DataFrame({
        "category": ["A", "B", "A", "C"]
    })
    
    featurizer = MetadataFeaturizer(categorical_columns=["category"])
    X = featurizer.fit_transform(df)
    
    # Should have 3 features (A, B, C)
    assert X.shape == (4, 3)
    
    # Check one-hot encoding
    # First row should be [1, 0, 0] (A)
    # Second row should be [0, 1, 0] (B)
    assert X[0].sum() == 1  # Only one category is active
    assert X[1].sum() == 1


def test_featurizer_unknown_categorical_at_transform():
    """Test handling of unknown categorical values during transform."""
    df_train = pd.DataFrame({
        "category": ["A", "B", "A"]
    })
    
    df_test = pd.DataFrame({
        "category": ["A", "C"]  # C is unknown
    })
    
    featurizer = MetadataFeaturizer(categorical_columns=["category"])
    featurizer.fit(df_train)
    X = featurizer.transform(df_test)
    
    # Unknown category should result in all zeros
    assert X[0].sum() == 1  # A is known
    assert X[1].sum() == 0  # C is unknown


def test_featurizer_empty_dataframe():
    """Test featurizer with empty dataframe."""
    df = pd.DataFrame({
        "instrument_name": [],
        "t_exptime": []
    })
    
    featurizer = MetadataFeaturizer(
        categorical_columns=["instrument_name"],
        numeric_columns=["t_exptime"]
    )
    
    X = featurizer.fit_transform(df)
    
    assert X.shape == (0, 1)  # 1 numeric feature, 0 categorical values


def test_get_default_featurizer():
    """Test default featurizer configuration."""
    featurizer = get_default_featurizer()
    
    assert isinstance(featurizer, MetadataFeaturizer)
    assert "t_exptime" in featurizer.numeric_columns
    assert "instrument_name" in featurizer.categorical_columns


def test_default_featurizer_on_data(sample_metadata):
    """Test default featurizer on sample data."""
    featurizer = get_default_featurizer()
    X = featurizer.fit_transform(sample_metadata)
    
    assert X.shape[0] == 4
    assert X.shape[1] > 0
    
    # Check that features are reasonable
    assert np.isfinite(X).all()
