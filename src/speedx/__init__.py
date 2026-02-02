"""speedX: A high-throughput science-content classifier for HST data."""

__version__ = "0.2.0"

from .classifier import RAGBoostClassifier
from .mast_client import MASTClient
from .ensemble import EnsembleClassifier, StackedClassifier
from .online_learning import OnlineRAGBoostClassifier, ActiveLearningClassifier
from .advanced_features import AdvancedFeatureEngineer
from .self_improving import SelfImprovingClassifier, AutoMLClassifier
from .intensive_training import OnlineDataTrainer, TransferLearningTrainer

__all__ = [
    "RAGBoostClassifier",
    "MASTClient",
    "EnsembleClassifier",
    "StackedClassifier",
    "OnlineRAGBoostClassifier",
    "ActiveLearningClassifier",
    "AdvancedFeatureEngineer",
    "SelfImprovingClassifier",
    "AutoMLClassifier",
    "OnlineDataTrainer",
    "TransferLearningTrainer",
    "__version__"
]
