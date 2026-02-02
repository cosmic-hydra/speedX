# speedX

A high-throughput science-content classifier for HST data using the HST/MAST API.

## Overview

speedX is a Python package designed for fast inference on large astronomy datasets. It provides:

- **RABoostClassifier**: A custom retrieval-augmented boosting classifier implemented from scratch
- **MAST Integration**: Query HST observations and download data products via the public MAST API
- **Feature Extraction**: Extract features from observation metadata and FITS files
- **CLI Tools**: Command-line interface for querying, training, and inference

## Installation

### From source

```bash
git clone https://github.com/cosmic-hydra/speedX.git
cd speedX
pip install -e .
```

### Development installation

For development with testing dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Query HST Observations

Query observation metadata from MAST and save to a local file:

```bash
speedx query-metadata --instrument "ACS/WFC" --max-records 100 -o observations.parquet
```

### Fetch Sample Data

Download a small sample of observation products:

```bash
speedx fetch-sample --instrument "WFC3/UVIS" -n 10 -o ./data/samples
```

### Extract FITS Statistics

Extract sketch statistics from a FITS file:

```bash
speedx sketch-fits observation.fits -o sketch.json
```

### Train a Classifier

Train the RABoostClassifier on observation metadata:

```bash
# First, create a labels file (CSV with obs_id and label columns)
speedx train -m observations.parquet -l labels.csv -o model.pkl
```

### Make Predictions

Predict labels for new observations:

```bash
speedx predict -m new_observations.parquet --model model.pkl -o predictions.csv
```

## Architecture

### RABoostClassifier

The RABoostClassifier combines three key techniques:

1. **Embedding Generation**: Uses random projection to create compact feature representations
2. **kNN Retrieval**: Finds similar training examples to augment features with context
3. **Boosting Ensemble**: Trains multiple weak learners (decision stumps) with adaptive weighting

This design enables fast inference while maintaining good performance on astronomy classification tasks.

### MAST Integration

The MAST client provides minimal, efficient access to HST data:

- **Query Building**: Flexible filtering by instrument, target, proposal, etc.
- **Metadata Caching**: Save query results to parquet/CSV for offline use
- **Product Download**: Fetch FITS files and other data products
- **Rate Limiting**: Built-in respect for API rate limits

The client uses the public MAST API and requires no authentication.

### Feature Extraction

#### Metadata Features

The `MetadataFeaturizer` converts observation metadata into feature vectors:

- Numeric features: exposure time, observation dates, etc.
- Categorical features: instrument, filter, data product type (one-hot encoded)

#### FITS Features

The `extract_fits_sketch` function computes cheap, robust statistics from FITS files:

- Basic statistics: mean, std, median, min, max, percentiles
- Data quality: NaN count, Inf count
- Efficient sampling for large images (avoids loading full arrays)

## CLI Reference

### `speedx query-metadata`

Query HST observation metadata from MAST.

```bash
speedx query-metadata [OPTIONS]

Options:
  -i, --instrument TEXT      Filter by instrument name
  -t, --target TEXT         Filter by target name
  -p, --proposal TEXT       Filter by proposal ID
  -n, --max-records INTEGER Maximum records to retrieve (default: 1000)
  -o, --output TEXT         Output file path (required)
  -f, --format [parquet|csv] Output format (default: parquet)
```

### `speedx fetch-sample`

Fetch sample observations and download products.

```bash
speedx fetch-sample [OPTIONS]

Options:
  -i, --instrument TEXT     Filter by instrument name
  -n, --n-samples INTEGER  Number of samples (default: 10)
  -o, --output-dir TEXT    Output directory (default: ./data/samples)
```

### `speedx train`

Train a RABoostClassifier on observation metadata.

```bash
speedx train [OPTIONS]

Options:
  -m, --metadata TEXT       Path to metadata file (required)
  -l, --labels TEXT         Path to labels CSV (required)
  -o, --output TEXT         Output model path (required)
  --n-estimators INTEGER    Number of boosting rounds (default: 10)
  --k-neighbors INTEGER     Number of neighbors for retrieval (default: 5)
```

### `speedx predict`

Predict labels using a trained model.

```bash
speedx predict [OPTIONS]

Options:
  -m, --metadata TEXT   Path to metadata file (required)
  --model TEXT          Path to trained model (required)
  -o, --output TEXT     Output predictions path (required)
```

### `speedx sketch-fits`

Extract sketch statistics from a FITS file.

```bash
speedx sketch-fits FITS_PATH [OPTIONS]

Options:
  -o, --output TEXT  Output JSON path (optional, prints to stdout if omitted)
```

## Python API

### Basic Usage

```python
from speedx import RABoostClassifier, MASTClient
from speedx.features import get_default_featurizer
import numpy as np

# Query observations
client = MASTClient()
observations = client.query_observations(
    instrument="ACS/WFC",
    max_records=100
)

# Extract features
featurizer = get_default_featurizer()
X = featurizer.fit_transform(observations)

# Create synthetic labels for demonstration
y = np.random.randint(0, 3, size=len(observations))

# Train classifier
clf = RABoostClassifier(
    n_estimators=10,
    k_neighbors=5,
    random_state=42
)
clf.fit(X, y)

# Predict
y_pred = clf.predict(X)
y_proba = clf.predict_proba(X)
```

### FITS Processing

```python
from speedx.fits_utils import extract_fits_sketch, sketch_to_features

# Extract sketch statistics
sketch = extract_fits_sketch("observation.fits")

# Convert to feature vector
features = sketch_to_features(sketch)
```

## Testing

Run the test suite:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=speedx --cov-report=html
```

## MAST API Integration

speedX uses the public MAST API to query HST observations and download data products. Key features:

- **No Authentication Required**: Uses public API endpoints
- **Flexible Filtering**: Filter by instrument, target, proposal, dates, etc.
- **Efficient Pagination**: Automatically handles large result sets
- **Local Caching**: Save metadata to parquet/CSV for offline use
- **Minimal I/O**: Design for I/O minimization with sampling and streaming

### Example Queries

Query ACS/WFC observations of a specific target:
```python
client = MASTClient()
obs = client.query_observations(
    instrument="ACS/WFC",
    target_name="NGC1234",
    max_records=50
)
```

Get data products for an observation:
```python
products = client.get_product_list("hst_12345_01_acs_wfc_f606w")
```

Download a product:
```python
file_path = client.download_product(
    product_uri="mast:HST/product/file.fits",
    output_dir="./data"
)
```

## Performance

speedX is designed for fast inference:

- **RABoostClassifier**: Lightweight boosting with kNN retrieval
- **FITS Sketching**: Sampling-based statistics avoid loading full arrays
- **Efficient Storage**: Parquet format for compressed metadata storage

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or pull request on GitHub.

## Citation

If you use speedX in your research, please cite:

```
@software{speedx,
  title = {speedX: High-throughput science-content classifier for HST data},
  author = {cosmic-hydra},
  year = {2026},
  url = {https://github.com/cosmic-hydra/speedX}
}
```