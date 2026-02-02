"""
FITS file utilities for extracting cheap robust sketch statistics.

Provides functions to extract statistical summaries from FITS files
without loading entire data arrays into memory when possible.
"""
import numpy as np
from astropy.io import fits
from typing import Dict, Any, Optional, List
import warnings


def extract_fits_sketch(
    fits_path: str,
    max_pixels: int = 1_000_000
) -> Dict[str, Any]:
    """
    Extract sketch statistics from a FITS file.
    
    Computes cheap, robust statistics that characterize the image
    without requiring full data loading. Uses sampling for large images.
    
    Parameters
    ----------
    fits_path : str
        Path to FITS file.
    max_pixels : int, optional
        Maximum number of pixels to sample for statistics.
        Default is 1 million.
        
    Returns
    -------
    sketch : dict
        Dictionary containing sketch statistics:
        - shape: Image shape
        - dtype: Data type
        - mean: Mean pixel value
        - std: Standard deviation
        - median: Median pixel value
        - min: Minimum pixel value
        - max: Maximum pixel value
        - q25: 25th percentile
        - q75: 75th percentile
        - n_nan: Number of NaN pixels (from sample)
        - n_inf: Number of Inf pixels (from sample)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        try:
            with fits.open(fits_path, memmap=True) as hdul:
                # Find first image HDU
                data = None
                for hdu in hdul:
                    if isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU)):
                        if hdu.data is not None and len(hdu.data.shape) > 0:
                            data = hdu.data
                            break
                
                if data is None:
                    raise RuntimeError("No image data found in FITS file")
                
                # Get shape and dtype
                shape = data.shape
                dtype = str(data.dtype)
                
                # Flatten if multidimensional
                if len(shape) > 2:
                    # Take middle slice for multi-dimensional data
                    middle_idx = tuple(s // 2 for s in shape[:-2])
                    data = data[middle_idx]
                    shape = data.shape
                
                # Sample if too large
                total_pixels = np.prod(shape)
                
                if total_pixels > max_pixels:
                    # Random sampling
                    n_samples = max_pixels
                    indices = np.random.choice(
                        total_pixels,
                        size=n_samples,
                        replace=False
                    )
                    flat_data = data.flatten()
                    sample = flat_data[indices]
                else:
                    sample = data.flatten()
                
                # Handle masked arrays
                if np.ma.is_masked(sample):
                    sample = sample.compressed()
                
                # Filter out non-finite values for statistics
                finite_mask = np.isfinite(sample)
                finite_sample = sample[finite_mask]
                
                # Compute statistics
                sketch = {
                    "shape": list(shape),
                    "dtype": dtype,
                    "n_nan": int(np.isnan(sample).sum()),
                    "n_inf": int(np.isinf(sample).sum()),
                }
                
                if len(finite_sample) > 0:
                    sketch.update({
                        "mean": float(np.mean(finite_sample)),
                        "std": float(np.std(finite_sample)),
                        "median": float(np.median(finite_sample)),
                        "min": float(np.min(finite_sample)),
                        "max": float(np.max(finite_sample)),
                        "q25": float(np.percentile(finite_sample, 25)),
                        "q75": float(np.percentile(finite_sample, 75)),
                    })
                else:
                    # All NaN/Inf
                    sketch.update({
                        "mean": np.nan,
                        "std": np.nan,
                        "median": np.nan,
                        "min": np.nan,
                        "max": np.nan,
                        "q25": np.nan,
                        "q75": np.nan,
                    })
                
                return sketch
                
        except Exception as e:
            raise RuntimeError(f"Failed to extract FITS sketch: {e}")


def extract_header_info(
    fits_path: str,
    keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Extract header information from FITS file.
    
    Parameters
    ----------
    fits_path : str
        Path to FITS file.
    keys : list of str, optional
        Specific header keys to extract. If None, extracts common keys.
        
    Returns
    -------
    header_info : dict
        Dictionary of header key-value pairs.
    """
    if keys is None:
        # Common HST header keys
        keys = [
            "TELESCOP",
            "INSTRUME",
            "DETECTOR",
            "FILTER",
            "EXPTIME",
            "DATE-OBS",
            "TARGNAME",
        ]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        try:
            with fits.open(fits_path) as hdul:
                header = hdul[0].header
                
                header_info = {}
                for key in keys:
                    if key in header:
                        header_info[key] = header[key]
                    else:
                        header_info[key] = None
                
                return header_info
                
        except Exception as e:
            raise RuntimeError(f"Failed to extract header info: {e}")


def sketch_to_features(sketch: Dict[str, Any]) -> np.ndarray:
    """
    Convert sketch statistics to a feature vector.
    
    Parameters
    ----------
    sketch : dict
        Sketch statistics from extract_fits_sketch.
        
    Returns
    -------
    features : np.ndarray of shape (n_features,)
        Feature vector.
    """
    # Extract numeric features
    feature_keys = [
        "mean", "std", "median", "min", "max", "q25", "q75",
        "n_nan", "n_inf"
    ]
    
    features = []
    for key in feature_keys:
        value = sketch.get(key, 0.0)
        if value is None or not np.isfinite(value):
            value = 0.0
        features.append(float(value))
    
    # Add shape features (log scale)
    shape = sketch.get("shape", [1, 1])
    if len(shape) >= 2:
        features.append(np.log1p(shape[0]))
        features.append(np.log1p(shape[1]))
    else:
        features.append(0.0)
        features.append(0.0)
    
    return np.array(features, dtype=np.float32)
