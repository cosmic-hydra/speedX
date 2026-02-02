"""
Online learning support for incremental model updates.
"""
import numpy as np
from typing import Optional
from .classifier import RAGBoostClassifier


class OnlineRAGBoostClassifier(RAGBoostClassifier):
    """
    Online learning variant of RAGBoostClassifier.
    
    Supports partial_fit for incremental learning on streaming data.
    
    Parameters
    ----------
    n_estimators : int, default=10
        Number of boosting rounds.
    k_neighbors : int, default=5
        Number of neighbors to retrieve.
    embedding_dim : int, default=32
        Embedding dimension.
    learning_rate : float, default=0.1
        Learning rate.
    max_train_samples : int, default=10000
        Maximum number of training samples to keep in memory.
    random_state : int, optional
        Random seed.
    """
    
    def __init__(
        self,
        n_estimators: int = 10,
        k_neighbors: int = 5,
        embedding_dim: int = 32,
        learning_rate: float = 0.1,
        max_train_samples: int = 10000,
        random_state: Optional[int] = None
    ):
        super().__init__(
            n_estimators=n_estimators,
            k_neighbors=k_neighbors,
            embedding_dim=embedding_dim,
            learning_rate=learning_rate,
            random_state=random_state
        )
        self.max_train_samples = max_train_samples
        self.n_samples_seen_ = 0
    
    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classes: Optional[np.ndarray] = None
    ) -> "OnlineRAGBoostClassifier":
        """
        Incrementally fit the model.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Training labels.
        classes : np.ndarray, optional
            All possible class labels. Required on first call.
            
        Returns
        -------
        self : OnlineRAGBoostClassifier
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # First call: initialize
        if self.X_train_ is None:
            if classes is None:
                classes = np.unique(y)
            self.classes_ = classes
            self.X_train_ = X.copy()
            self.y_train_ = y.copy()
            self.n_samples_seen_ = len(X)
            
            # Fit initial model
            return self.fit(X, y)
        
        # Subsequent calls: append new data
        self.X_train_ = np.vstack([self.X_train_, X])
        self.y_train_ = np.concatenate([self.y_train_, y])
        self.n_samples_seen_ += len(X)
        
        # Limit memory by keeping only recent samples
        if len(self.X_train_) > self.max_train_samples:
            keep_from = len(self.X_train_) - self.max_train_samples
            self.X_train_ = self.X_train_[keep_from:]
            self.y_train_ = self.y_train_[keep_from:]
        
        # Refit model on updated data
        return self.fit(self.X_train_, self.y_train_)


class ActiveLearningClassifier:
    """
    Active learning wrapper for efficient labeling.
    
    Identifies samples where the model is most uncertain
    for prioritized labeling.
    
    Parameters
    ----------
    base_classifier : RAGBoostClassifier
        Base classifier to use.
    uncertainty_threshold : float, default=0.3
        Entropy threshold for uncertain predictions.
    """
    
    def __init__(
        self,
        base_classifier: Optional[RAGBoostClassifier] = None,
        uncertainty_threshold: float = 0.3
    ):
        self.base_classifier = base_classifier or RAGBoostClassifier()
        self.uncertainty_threshold = uncertainty_threshold
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ActiveLearningClassifier":
        """
        Fit the classifier.
        
        Parameters
        ----------
        X : np.ndarray
            Training features.
        y : np.ndarray
            Training labels.
            
        Returns
        -------
        self : ActiveLearningClassifier
        """
        self.base_classifier.fit(X, y)
        return self
    
    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Compute prediction uncertainty using entropy.
        
        Parameters
        ----------
        X : np.ndarray
            Input features.
            
        Returns
        -------
        uncertainty : np.ndarray
            Uncertainty score for each sample (higher = more uncertain).
        """
        proba = self.base_classifier.predict_proba(X)
        
        # Compute entropy
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        entropy = -np.sum(proba * np.log(proba + epsilon), axis=1)
        
        # Normalize by max entropy (uniform distribution)
        n_classes = proba.shape[1]
        max_entropy = -np.log(1.0 / n_classes)
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy
    
    def get_uncertain_samples(
        self,
        X: np.ndarray,
        n_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Get indices of most uncertain samples.
        
        Parameters
        ----------
        X : np.ndarray
            Input features.
        n_samples : int, optional
            Number of uncertain samples to return.
            If None, returns all samples above uncertainty threshold.
            
        Returns
        -------
        indices : np.ndarray
            Indices of uncertain samples.
        """
        uncertainty = self.predict_uncertainty(X)
        
        if n_samples is None:
            # Return all samples above threshold
            return np.where(uncertainty > self.uncertainty_threshold)[0]
        else:
            # Return top n most uncertain
            return np.argsort(uncertainty)[-n_samples:]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.base_classifier.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.base_classifier.predict_proba(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy score."""
        return self.base_classifier.score(X, y)
