"""Tests for self-improving AI."""
import numpy as np
import pytest
from speedx.self_improving import SelfImprovingClassifier, AutoMLClassifier


def test_self_improving_initialization():
    """Test self-improving classifier initialization."""
    clf = SelfImprovingClassifier(
        optimization_rounds=5,
        auto_tune=True
    )
    assert clf.optimization_rounds == 5
    assert clf.auto_tune is True


def test_self_improving_fit():
    """Test self-improving classifier fit."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, size=100)
    
    clf = SelfImprovingClassifier(
        optimization_rounds=3,
        auto_tune=True
    )
    clf.fit(X, y)
    
    assert clf.classifier_ is not None
    assert clf.best_params_ is not None
    assert clf.best_score_ > 0


def test_self_improving_predict():
    """Test self-improving classifier prediction."""
    np.random.seed(42)
    X_train = np.random.randn(80, 10)
    y_train = np.random.randint(0, 2, size=80)
    X_test = np.random.randn(20, 10)
    
    clf = SelfImprovingClassifier(
        optimization_rounds=2,
        auto_tune=False  # Faster for testing
    )
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    assert y_pred.shape == (20,)


def test_self_improving_continuous_improvement():
    """Test continuous improvement."""
    np.random.seed(42)
    X1 = np.random.randn(50, 5)
    y1 = np.random.randint(0, 2, size=50)
    
    clf = SelfImprovingClassifier(
        optimization_rounds=2,
        auto_tune=False
    )
    clf.fit(X1, y1)
    
    initial_iterations = clf.n_iterations_
    
    # Add more data
    X2 = np.random.randn(30, 5)
    y2 = np.random.randint(0, 2, size=30)
    
    clf.improve(X2, y2)
    
    assert clf.n_iterations_ > initial_iterations
    assert len(clf.performance_history_) > 1


def test_self_improving_performance_report():
    """Test performance report generation."""
    np.random.seed(42)
    X = np.random.randn(60, 5)
    y = np.random.randint(0, 3, size=60)
    
    clf = SelfImprovingClassifier(optimization_rounds=2, auto_tune=False)
    clf.fit(X, y)
    
    report = clf.get_performance_report()
    
    assert 'current_score' in report
    assert 'best_score' in report
    assert 'n_iterations' in report
    assert 'best_params' in report


def test_automl_initialization():
    """Test AutoML classifier initialization."""
    automl = AutoMLClassifier(
        models_to_try=['single', 'ensemble'],
        optimization_budget=10
    )
    assert automl.optimization_budget == 10
    assert 'single' in automl.models_to_try


def test_automl_fit():
    """Test AutoML fit."""
    np.random.seed(42)
    X = np.random.randn(80, 8)
    y = np.random.randint(0, 2, size=80)
    
    automl = AutoMLClassifier(
        models_to_try=['single'],
        optimization_budget=5
    )
    automl.fit(X, y)
    
    assert automl.best_model_ is not None
    assert automl.best_model_type_ is not None
    assert automl.best_score_ > 0


def test_automl_predict():
    """Test AutoML prediction."""
    np.random.seed(42)
    X_train = np.random.randn(70, 6)
    y_train = np.random.randint(0, 2, size=70)
    X_test = np.random.randn(20, 6)
    
    automl = AutoMLClassifier(
        models_to_try=['single'],
        optimization_budget=3
    )
    automl.fit(X_train, y_train)
    
    y_pred = automl.predict(X_test)
    assert y_pred.shape == (20,)
    
    y_proba = automl.predict_proba(X_test)
    assert y_proba.shape[0] == 20


def test_automl_model_selection():
    """Test that AutoML evaluates multiple models."""
    np.random.seed(42)
    X = np.random.randn(60, 5)
    y = np.random.randint(0, 2, size=60)
    
    automl = AutoMLClassifier(
        models_to_try=['single', 'ensemble'],
        optimization_budget=6
    )
    automl.fit(X, y)
    
    # Should have evaluated multiple configurations
    assert len(automl.evaluation_results_) > 0
    assert automl.best_model_type_ in ['single', 'ensemble']
