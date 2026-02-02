# Changelog

All notable changes to speedX will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-02

### Added
- Initial release of speedX
- RABoostClassifier: Custom retrieval-augmented boosting classifier
  - Embedding generation via random projection
  - Exact kNN retrieval with cosine similarity
  - Adaptive boosting with decision stumps
- MAST client for HST observations
  - Query building with flexible filters
  - Metadata caching (parquet/CSV)
  - Product download functionality
  - Rate limiting and retry logic
- Feature extraction
  - MetadataFeaturizer for observation metadata
  - FITS sketch extraction with sampling
- CLI with 5 commands
  - query-metadata: Query HST observations
  - fetch-sample: Download sample data
  - train: Train classifier
  - predict: Make predictions
  - sketch-fits: Extract FITS statistics
- Comprehensive test suite (62 tests)
  - Unit tests for all modules
  - Integration tests
  - Validation tests
  - Mocked HTTP tests for MAST client
- Complete documentation
  - Installation guide
  - CLI reference
  - Python API documentation
  - Architecture overview
- GitHub Actions CI/CD workflow

### Quality Improvements
- Input validation for all classifier parameters
- Proper error messages with context
- Edge case handling for empty inputs
- Retry logic with exponential backoff for MAST requests
- Verbose mode for CLI commands
- Better error reporting in CLI

### Known Limitations
- No real training on HST labels (by design)
- Exact kNN only (ANN libraries can be added)
- No GPU support (CPU-only)
