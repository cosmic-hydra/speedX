"""Tests for ensemble methods."""
import numpy as np
import pytest
from speedx.ensemble import EnsembleClassifier, StackedClassifier
from speedx.classifier import RAGBoostClassifier


def test_ensemble_initialization():
    """Test ensemble initialization."""
    ensemble = EnsembleClassifier(n_models=3, voting='soft')
    assert ensemble.n_models == 3
    assert ensemble.voting == 'soft'


def test_ensemble_invalid_voting():
    """Test that invalid voting method raises error."""
    with pytest.raises(ValueError, match="voting must be"):
        EnsembleClassifier(voting='invalid')


def test_ensemble_fit_predict():
    """Test ensemble fit and predict."""
    np.random.seed(42)
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, size=50)
    
    ensemble = EnsembleClassifier(n_models=3, voting='soft')
    ensemble.fit(X, y)
    
    assert len(ensemble.models_) == 3
    assert ensemble.classes_ is not None
    
    y_pred = ensemble.predict(X)
    assert y_pred.shape == (50,)


def test_ensemble_hard_voting():
    """Test ensemble with hard voting."""
    np.random.seed(42)
    X = np.random.randn(30, 5)
    y = np.random.randint(0, 2, size=30)
    
    ensemble = EnsembleClassifier(n_models=3, voting='hard')
    ensemble.fit(X, y)
    
    y_pred = ensemble.predict(X)
    assert y_pred.shape == (30,)


def test_ensemble_score():
    """Test ensemble scoring."""
    np.random.seed(42)
    X = np.random.randn(40, 5)
    y = np.random.randint(0, 2, size=40)
    
    ensemble = EnsembleClassifier(n_models=2)
    ensemble.fit(X, y)
    
    score = ensemble.score(X, y)
    assert 0 <= score <= 1


def test_stacked_classifier():
    """Test stacked classifier."""
    np.random.seed(42)
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 3, size=50)
    
    # Create base classifiers
    base_clfs = [
        RAGBoostClassifier(n_estimators=3, random_state=i)
        for i in range(2)
    ]
    
    stacked = StackedClassifier(base_classifiers=base_clfs)
    stacked.fit(X, y)
    
    y_pred = stacked.predict(X)
    assert y_pred.shape == (50,)
    
    y_proba = stacked.predict_proba(X)
    assert y_proba.shape == (50, 3)


def test_stacked_classifier_score():
    """Test stacked classifier scoring."""
    np.random.seed(42)
    X = np.random.randn(40, 5)
    y = np.random.randint(0, 2, size=40)
    
    base_clfs = [RAGBoostClassifier(n_estimators=2, random_state=i) for i in range(2)]
    stacked = StackedClassifier(base_classifiers=base_clfs)
    stacked.fit(X, y)
    
    score = stacked.score(X, y)
    assert 0 <= score <= 1
