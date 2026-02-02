"""speedX: A high-throughput science-content classifier for HST data."""

__version__ = "0.1.0"

from .classifier import RABoostClassifier
from .mast_client import MASTClient
from .ensemble import EnsembleClassifier, StackedClassifier
from .online_learning import OnlineRABoostClassifier, ActiveLearningClassifier
from .advanced_features import AdvancedFeatureEngineer

__all__ = [
    "RABoostClassifier",
    "MASTClient",
    "EnsembleClassifier",
    "StackedClassifier",
    "OnlineRABoostClassifier",
    "ActiveLearningClassifier",
    "AdvancedFeatureEngineer",
    "__version__"
]
