"""Tests for FITS utilities."""
import numpy as np
import pytest
from pathlib import Path
from astropy.io import fits
from speedx.fits_utils import (
    extract_fits_sketch,
    extract_header_info,
    sketch_to_features
)
import tempfile


@pytest.fixture
def synthetic_fits():
    """Create a synthetic FITS file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        fits_path = f.name
    
    # Create synthetic image data
    np.random.seed(42)
    data = np.random.randn(100, 100).astype(np.float32) * 100 + 500
    
    # Add some NaN and Inf values
    data[0, 0] = np.nan
    data[1, 1] = np.inf
    
    # Create FITS file
    hdu = fits.PrimaryHDU(data)
    hdu.header['TELESCOP'] = 'HST'
    hdu.header['INSTRUME'] = 'ACS'
    hdu.header['DETECTOR'] = 'WFC'
    hdu.header['FILTER'] = 'F606W'
    hdu.header['EXPTIME'] = 100.0
    hdu.header['DATE-OBS'] = '2023-01-01'
    hdu.header['TARGNAME'] = 'NGC1234'
    
    hdul = fits.HDUList([hdu])
    hdul.writeto(fits_path, overwrite=True)
    
    yield fits_path
    
    # Cleanup
    Path(fits_path).unlink()


def test_extract_fits_sketch(synthetic_fits):
    """Test FITS sketch extraction."""
    sketch = extract_fits_sketch(synthetic_fits)
    
    # Check required fields
    assert "shape" in sketch
    assert "dtype" in sketch
    assert "mean" in sketch
    assert "std" in sketch
    assert "median" in sketch
    assert "min" in sketch
    assert "max" in sketch
    assert "q25" in sketch
    assert "q75" in sketch
    assert "n_nan" in sketch
    assert "n_inf" in sketch
    
    # Check values
    assert sketch["shape"] == [100, 100]
    # dtype can be 'float32' or '>f4' depending on endianness
    assert 'float32' in sketch["dtype"] or 'f4' in sketch["dtype"]
    assert sketch["n_nan"] == 1
    assert sketch["n_inf"] == 1
    
    # Check statistics are reasonable
    assert 400 < sketch["mean"] < 600
    assert 50 < sketch["std"] < 150
    assert sketch["min"] < sketch["q25"] < sketch["median"] < sketch["q75"] < sketch["max"]


def test_extract_fits_sketch_reproducibility(synthetic_fits):
    """Test that sketch extraction is reproducible."""
    np.random.seed(42)
    sketch1 = extract_fits_sketch(synthetic_fits)
    
    np.random.seed(42)
    sketch2 = extract_fits_sketch(synthetic_fits)
    
    # Should be identical with same random seed
    assert sketch1["mean"] == sketch2["mean"]
    assert sketch1["std"] == sketch2["std"]
    assert sketch1["median"] == sketch2["median"]


def test_extract_fits_sketch_large_image():
    """Test sketch extraction with sampling for large images."""
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        fits_path = f.name
    
    try:
        # Create large synthetic image
        np.random.seed(42)
        data = np.random.randn(2000, 2000).astype(np.float32)
        
        hdu = fits.PrimaryHDU(data)
        hdul = fits.HDUList([hdu])
        hdul.writeto(fits_path, overwrite=True)
        
        # Extract with sampling
        sketch = extract_fits_sketch(fits_path, max_pixels=100000)
        
        assert sketch["shape"] == [2000, 2000]
        # Statistics should still be reasonable due to sampling
        assert -0.5 < sketch["mean"] < 0.5
        assert 0.8 < sketch["std"] < 1.2
        
    finally:
        Path(fits_path).unlink()


def test_extract_header_info(synthetic_fits):
    """Test FITS header extraction."""
    header_info = extract_header_info(synthetic_fits)
    
    # Check extracted fields
    assert header_info["TELESCOP"] == "HST"
    assert header_info["INSTRUME"] == "ACS"
    assert header_info["DETECTOR"] == "WFC"
    assert header_info["FILTER"] == "F606W"
    assert header_info["EXPTIME"] == 100.0
    assert header_info["DATE-OBS"] == "2023-01-01"
    assert header_info["TARGNAME"] == "NGC1234"


def test_extract_header_info_specific_keys(synthetic_fits):
    """Test header extraction with specific keys."""
    header_info = extract_header_info(
        synthetic_fits,
        keys=["TELESCOP", "FILTER"]
    )
    
    assert len(header_info) == 2
    assert header_info["TELESCOP"] == "HST"
    assert header_info["FILTER"] == "F606W"


def test_extract_header_info_missing_keys(synthetic_fits):
    """Test header extraction with missing keys."""
    header_info = extract_header_info(
        synthetic_fits,
        keys=["NONEXISTENT", "TELESCOP"]
    )
    
    assert header_info["NONEXISTENT"] is None
    assert header_info["TELESCOP"] == "HST"


def test_sketch_to_features(synthetic_fits):
    """Test conversion of sketch to feature vector."""
    sketch = extract_fits_sketch(synthetic_fits)
    features = sketch_to_features(sketch)
    
    # Check feature vector
    assert isinstance(features, np.ndarray)
    assert features.dtype == np.float32
    assert features.shape == (11,)  # 9 stats + 2 shape features
    
    # All features should be finite
    assert np.isfinite(features).all()


def test_sketch_to_features_with_nans():
    """Test feature conversion handles NaN values."""
    sketch = {
        "mean": np.nan,
        "std": 1.0,
        "median": np.nan,
        "min": 0.0,
        "max": 10.0,
        "q25": 2.0,
        "q75": 8.0,
        "n_nan": 100,
        "n_inf": 0,
        "shape": [100, 100]
    }
    
    features = sketch_to_features(sketch)
    
    # NaN values should be converted to 0
    assert np.isfinite(features).all()
    assert features[0] == 0.0  # mean was NaN
    assert features[2] == 0.0  # median was NaN


def test_extract_fits_sketch_3d():
    """Test sketch extraction from 3D data."""
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        fits_path = f.name
    
    try:
        # Create 3D data (e.g., IFU cube)
        data = np.random.randn(10, 50, 50).astype(np.float32)
        
        hdu = fits.PrimaryHDU(data)
        hdul = fits.HDUList([hdu])
        hdul.writeto(fits_path, overwrite=True)
        
        sketch = extract_fits_sketch(fits_path)
        
        # Should extract middle slice
        assert len(sketch["shape"]) == 2
        assert sketch["shape"] == [50, 50]
        
    finally:
        Path(fits_path).unlink()


def test_extract_fits_sketch_empty_hdu():
    """Test sketch extraction from FITS with no image data."""
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        fits_path = f.name
    
    try:
        # Create FITS with only header (no data)
        hdu = fits.PrimaryHDU()
        hdul = fits.HDUList([hdu])
        hdul.writeto(fits_path, overwrite=True)
        
        with pytest.raises(RuntimeError, match="No image data found"):
            extract_fits_sketch(fits_path)
        
    finally:
        Path(fits_path).unlink()
