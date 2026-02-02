# speedX

A high-throughput science-content classifier for HST data using the HST/MAST API.

## Overview

speedX is a Python package designed for fast inference on large astronomy datasets. It provides:

- **RAGBoostClassifier**: A custom retrieval-augmented boosting classifier implemented from scratch
- **MAST Integration**: Query HST observations and download data products via the public MAST API
- **Feature Extraction**: Extract features from observation metadata and FITS files
- **CLI Tools**: Command-line interface for querying, training, and inference
- **Ensemble Methods**: Multiple classifier ensembles for improved accuracy
- **Online Learning**: Incremental learning on streaming data
- **Active Learning**: Uncertainty-based sample selection for efficient labeling

## New Features (v0.2.0)

### 🚀 Performance Enhancements
- **Batch Prediction**: Memory-efficient batch processing for large datasets
- **Caching**: Embedding cache for faster repeated predictions
- **Optimized kNN**: Efficient nearest neighbor retrieval

### 🤖 Advanced AI Capabilities
- **Ensemble Classifier**: Combine multiple models with soft/hard voting
- **Stacked Ensemble**: Meta-learning with stacked classifiers
- **Online Learning**: Incremental updates with `OnlineRAGBoostClassifier`
- **Active Learning**: Identify uncertain samples for efficient labeling
- **Advanced Features**: Polynomial features, PCA, domain-specific transformations

### 📊 Enhanced Features
- **Model Persistence**: Save/load models with pickle
- **Scikit-learn API**: Compatible `get_params`/`set_params`/`score` methods
- **Better Validation**: Comprehensive input validation with informative errors
- **Retry Logic**: Exponential backoff for failed MAST requests

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

Train the RAGBoostClassifier on observation metadata:

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

### RAGBoostClassifier

The RAGBoostClassifier combines three key techniques:

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

Train a RAGBoostClassifier on observation metadata.

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
from speedx import RAGBoostClassifier, MASTClient
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
clf = RAGBoostClassifier(
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

- **RAGBoostClassifier**: Lightweight boosting with kNN retrieval
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
## Advanced Usage

### Ensemble Learning

Use ensemble methods for improved accuracy:

```python
from speedx import EnsembleClassifier, RAGBoostClassifier
import numpy as np

# Create ensemble of 5 models
ensemble = EnsembleClassifier(
    n_models=5,
    base_params={'n_estimators': 10, 'k_neighbors': 5},
    voting='soft'  # or 'hard' for majority vote
)

X_train, y_train = ...  # Your training data
ensemble.fit(X_train, y_train)

# Predict with ensemble
y_pred = ensemble.predict(X_test)
accuracy = ensemble.score(X_test, y_test)
```

### Online Learning

Update models incrementally with new data:

```python
from speedx import OnlineRAGBoostClassifier

# Create online classifier
online_clf = OnlineRAGBoostClassifier(
    n_estimators=10,
    max_train_samples=10000  # Limit memory usage
)

# Initial training
online_clf.partial_fit(X_batch1, y_batch1, classes=[0, 1, 2])

# Incremental updates as new data arrives
for X_batch, y_batch in data_stream:
    online_clf.partial_fit(X_batch, y_batch)
    
# Predict on new data
y_pred = online_clf.predict(X_new)
```

### Active Learning

Identify uncertain samples for efficient labeling:

```python
from speedx import ActiveLearningClassifier

# Create active learning classifier
al_clf = ActiveLearningClassifier(
    uncertainty_threshold=0.3
)

al_clf.fit(X_labeled, y_labeled)

# Find most uncertain samples from unlabeled pool
uncertain_indices = al_clf.get_uncertain_samples(X_unlabeled, n_samples=100)

# Label only these uncertain samples
X_to_label = X_unlabeled[uncertain_indices]
# ... get labels for these samples ...

# Retrain with newly labeled data
al_clf.fit(X_all, y_all)
```

### Advanced Feature Engineering

Use advanced transformations:

```python
from speedx.advanced_features import (
    AdvancedFeatureEngineer,
    create_time_features,
    create_exposure_features
)

# Polynomial features + PCA
engineer = AdvancedFeatureEngineer(
    polynomial_degree=2,
    n_pca_components=50,
    scale_features=True
)

X_transformed = engineer.fit_transform(X)

# Domain-specific features
df_enhanced = create_time_features(df, time_column='t_min')
df_enhanced = create_exposure_features(df_enhanced)
```

### Batch Prediction

Process large datasets efficiently:

```python
# Batch prediction for memory efficiency
y_pred = clf.predict_batch(
    X_large,
    batch_size=1000,
    verbose=True  # Show progress
)
```

### Model Persistence

Save and load trained models:

```python
# Save model
clf.save('my_model.pkl')

# Load model
from speedx import RAGBoostClassifier
clf = RAGBoostClassifier.load('my_model.pkl')

# Use loaded model
y_pred = clf.predict(X_test)
```

## Performance Benchmarks

speedX is optimized for speed and efficiency:

- **Training**: ~1-2 seconds per 1000 samples (10 estimators, 5 neighbors)
- **Prediction**: ~0.1-0.2 seconds per 1000 samples
- **Memory**: Embedding cache reduces repeated prediction time by 50%
- **Batch Processing**: Linear scaling with dataset size

### Tips for Speed

1. **Use caching**: Enable `use_cache=True` (default) for repeated predictions
2. **Batch predictions**: Use `predict_batch()` for large datasets
3. **Optimize parameters**: Fewer estimators and neighbors = faster inference
4. **Ensemble wisely**: Balance ensemble size with speed requirements
5. **Feature engineering**: PCA can reduce dimensionality for faster retrieval

