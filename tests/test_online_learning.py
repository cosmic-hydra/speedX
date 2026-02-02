"""Tests for online learning."""
import numpy as np
import pytest
from speedx.online_learning import OnlineRABoostClassifier, ActiveLearningClassifier


def test_online_classifier_partial_fit():
    """Test online classifier with partial_fit."""
    np.random.seed(42)
    
    # Initial batch
    X1 = np.random.randn(20, 5)
    y1 = np.random.randint(0, 2, size=20)
    
    clf = OnlineRABoostClassifier(n_estimators=3, random_state=42)
    clf.partial_fit(X1, y1, classes=np.array([0, 1]))
    
    assert clf.n_samples_seen_ == 20
    assert clf.X_train_ is not None
    
    # Second batch
    X2 = np.random.randn(15, 5)
    y2 = np.random.randint(0, 2, size=15)
    
    clf.partial_fit(X2, y2)
    
    assert clf.n_samples_seen_ == 35
    
    # Predict
    y_pred = clf.predict(X2)
    assert y_pred.shape == (15,)


def test_online_classifier_memory_limit():
    """Test that online classifier respects memory limit."""
    np.random.seed(42)
    
    clf = OnlineRABoostClassifier(max_train_samples=50, random_state=42)
    
    # Add many samples
    for i in range(10):
        X = np.random.randn(10, 5)
        y = np.random.randint(0, 2, size=10)
        if i == 0:
            clf.partial_fit(X, y, classes=np.array([0, 1]))
        else:
            clf.partial_fit(X, y)
    
    # Should only keep max_train_samples
    assert len(clf.X_train_) <= 50


def test_active_learning_uncertainty():
    """Test active learning uncertainty computation."""
    np.random.seed(42)
    X_train = np.random.randn(50, 5)
    y_train = np.random.randint(0, 3, size=50)
    
    al_clf = ActiveLearningClassifier()
    al_clf.fit(X_train, y_train)
    
    X_test = np.random.randn(20, 5)
    uncertainty = al_clf.predict_uncertainty(X_test)
    
    assert uncertainty.shape == (20,)
    assert (uncertainty >= 0).all()
    assert (uncertainty <= 1).all()


def test_active_learning_get_uncertain_samples():
    """Test getting uncertain samples."""
    np.random.seed(42)
    X_train = np.random.randn(40, 5)
    y_train = np.random.randint(0, 2, size=40)
    
    al_clf = ActiveLearningClassifier(uncertainty_threshold=0.3)
    al_clf.fit(X_train, y_train)
    
    X_pool = np.random.randn(30, 5)
    
    # Get top 5 uncertain samples
    uncertain_indices = al_clf.get_uncertain_samples(X_pool, n_samples=5)
    assert len(uncertain_indices) == 5
    
    # Get all uncertain samples above threshold
    uncertain_all = al_clf.get_uncertain_samples(X_pool)
    assert len(uncertain_all) <= 30


def test_active_learning_predict():
    """Test active learning predictions."""
    np.random.seed(42)
    X_train = np.random.randn(40, 5)
    y_train = np.random.randint(0, 2, size=40)
    
    al_clf = ActiveLearningClassifier()
    al_clf.fit(X_train, y_train)
    
    X_test = np.random.randn(10, 5)
    y_pred = al_clf.predict(X_test)
    
    assert y_pred.shape == (10,)
    
    y_proba = al_clf.predict_proba(X_test)
    assert y_proba.shape == (10, 2)
