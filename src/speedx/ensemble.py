"""
Ensemble methods for improved prediction accuracy.
"""
import numpy as np
from typing import List, Optional
from .classifier import RABoostClassifier


class EnsembleClassifier:
    """
    Ensemble of multiple RABoostClassifiers with voting.
    
    Combines predictions from multiple classifiers trained with
    different random seeds or configurations for improved accuracy.
    
    Parameters
    ----------
    n_models : int, default=5
        Number of models in the ensemble.
    base_params : dict, optional
        Base parameters for each RABoostClassifier.
    voting : str, default='soft'
        Voting method: 'hard' for majority vote, 'soft' for averaged probabilities.
    """
    
    def __init__(
        self,
        n_models: int = 5,
        base_params: Optional[dict] = None,
        voting: str = 'soft'
    ):
        self.n_models = n_models
        self.base_params = base_params or {}
        self.voting = voting
        
        if voting not in ['hard', 'soft']:
            raise ValueError("voting must be 'hard' or 'soft'")
        
        self.models_ = []
        self.classes_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleClassifier":
        """
        Fit the ensemble.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Training labels.
            
        Returns
        -------
        self : EnsembleClassifier
        """
        self.models_ = []
        self.classes_ = np.unique(y)
        
        # Train multiple models with different random seeds
        for i in range(self.n_models):
            params = self.base_params.copy()
            params['random_state'] = params.get('random_state', 42) + i
            
            model = RABoostClassifier(**params)
            model.fit(X, y)
            self.models_.append(model)
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using ensemble.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features.
            
        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Averaged class probabilities.
        """
        if not self.models_:
            raise ValueError("Ensemble must be fitted before predicting")
        
        # Collect predictions from all models
        all_probas = np.array([model.predict_proba(X) for model in self.models_])
        
        # Average probabilities
        return all_probas.mean(axis=0)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using ensemble.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features.
            
        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted labels.
        """
        if self.voting == 'soft':
            # Soft voting: use averaged probabilities
            proba = self.predict_proba(X)
            return self.classes_[np.argmax(proba, axis=1)]
        else:
            # Hard voting: majority vote
            predictions = np.array([model.predict(X) for model in self.models_])
            
            # Get mode (most common prediction) for each sample
            y_pred = np.empty(X.shape[0], dtype=self.classes_.dtype)
            for i in range(X.shape[0]):
                votes = predictions[:, i]
                unique, counts = np.unique(votes, return_counts=True)
                y_pred[i] = unique[np.argmax(counts)]
            
            return y_pred
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the mean accuracy.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test samples.
        y : np.ndarray of shape (n_samples,)
            True labels.
            
        Returns
        -------
        score : float
            Mean accuracy.
        """
        y_pred = self.predict(X)
        return (y_pred == y).mean()


class StackedClassifier:
    """
    Stacked ensemble using predictions as meta-features.
    
    Uses predictions from base classifiers as input to a meta-classifier.
    
    Parameters
    ----------
    base_classifiers : list of RABoostClassifier
        Base classifiers for the first level.
    meta_classifier : RABoostClassifier, optional
        Meta-classifier for the second level. If None, creates a new one.
    """
    
    def __init__(
        self,
        base_classifiers: List[RABoostClassifier],
        meta_classifier: Optional[RABoostClassifier] = None
    ):
        self.base_classifiers = base_classifiers
        self.meta_classifier = meta_classifier or RABoostClassifier(
            n_estimators=5,
            k_neighbors=3,
            random_state=42
        )
        self.classes_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackedClassifier":
        """
        Fit the stacked classifier.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Training labels.
            
        Returns
        -------
        self : StackedClassifier
        """
        self.classes_ = np.unique(y)
        
        # Train base classifiers
        for clf in self.base_classifiers:
            clf.fit(X, y)
        
        # Generate meta-features (predictions from base classifiers)
        meta_features = self._generate_meta_features(X)
        
        # Train meta-classifier
        self.meta_classifier.fit(meta_features, y)
        
        return self
    
    def _generate_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Generate meta-features from base classifier predictions."""
        meta_features_list = []
        
        for clf in self.base_classifiers:
            proba = clf.predict_proba(X)
            meta_features_list.append(proba)
        
        # Concatenate all probability predictions
        return np.hstack(meta_features_list)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features.
            
        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        meta_features = self._generate_meta_features(X)
        return self.meta_classifier.predict_proba(meta_features)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features.
            
        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the mean accuracy.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test samples.
        y : np.ndarray of shape (n_samples,)
            True labels.
            
        Returns
        -------
        score : float
            Mean accuracy.
        """
        y_pred = self.predict(X)
        return (y_pred == y).mean()
