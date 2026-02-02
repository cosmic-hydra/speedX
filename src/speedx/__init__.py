"""speedX: A high-throughput science-content classifier for HST data."""

__version__ = "0.1.0"

from .classifier import RABoostClassifier
from .mast_client import MASTClient

__all__ = ["RABoostClassifier", "MASTClient", "__version__"]
