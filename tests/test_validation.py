"""Tests for input validation and error handling."""
import numpy as np
import pytest
from speedx.classifier import RABoostClassifier


def test_classifier_negative_n_estimators():
    """Test that negative n_estimators raises ValueError."""
    with pytest.raises(ValueError, match="n_estimators must be positive"):
        RABoostClassifier(n_estimators=-1)


def test_classifier_zero_n_estimators():
    """Test that zero n_estimators raises ValueError."""
    with pytest.raises(ValueError, match="n_estimators must be positive"):
        RABoostClassifier(n_estimators=0)


def test_classifier_negative_k_neighbors():
    """Test that negative k_neighbors raises ValueError."""
    with pytest.raises(ValueError, match="k_neighbors must be positive"):
        RABoostClassifier(k_neighbors=-1)


def test_classifier_negative_embedding_dim():
    """Test that negative embedding_dim raises ValueError."""
    with pytest.raises(ValueError, match="embedding_dim must be positive"):
        RABoostClassifier(embedding_dim=-1)


def test_classifier_negative_learning_rate():
    """Test that negative learning_rate raises ValueError."""
    with pytest.raises(ValueError, match="learning_rate must be positive"):
        RABoostClassifier(learning_rate=-0.1)


def test_classifier_fit_empty_X():
    """Test that fitting with empty X raises ValueError."""
    clf = RABoostClassifier()
    X = np.array([])
    y = np.array([])
    
    with pytest.raises(ValueError, match="X cannot be empty"):
        clf.fit(X, y)


def test_classifier_fit_empty_y():
    """Test that fitting with empty y raises ValueError."""
    clf = RABoostClassifier()
    X = np.random.randn(10, 5)
    y = np.array([])
    
    with pytest.raises(ValueError, match="y cannot be empty"):
        clf.fit(X, y)


def test_classifier_fit_mismatched_lengths():
    """Test that mismatched X and y lengths raise ValueError."""
    clf = RABoostClassifier()
    X = np.random.randn(10, 5)
    y = np.array([0, 1, 0])
    
    with pytest.raises(ValueError, match="X and y must have the same length"):
        clf.fit(X, y)


def test_classifier_fit_wrong_X_dimension():
    """Test that 1D X raises ValueError."""
    clf = RABoostClassifier()
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([0, 1, 0, 1, 0])
    
    with pytest.raises(ValueError, match="X must be 2-dimensional"):
        clf.fit(X, y)


def test_classifier_predict_before_fit():
    """Test that predicting before fitting raises ValueError."""
    clf = RABoostClassifier()
    X = np.random.randn(5, 3)
    
    with pytest.raises(ValueError, match="Model must be fitted before predicting"):
        clf.predict(X)


def test_classifier_predict_proba_before_fit():
    """Test that predict_proba before fitting raises ValueError."""
    clf = RABoostClassifier()
    X = np.random.randn(5, 3)
    
    with pytest.raises(ValueError, match="Model must be fitted before predicting"):
        clf.predict_proba(X)


def test_classifier_predict_empty_X():
    """Test that predicting with empty X raises ValueError."""
    clf = RABoostClassifier(random_state=42)
    X_train = np.random.randn(10, 5)
    y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    clf.fit(X_train, y_train)
    
    X_test = np.array([])
    with pytest.raises(ValueError, match="X cannot be empty"):
        clf.predict(X_test)


def test_classifier_predict_wrong_dimension():
    """Test that predicting with 1D X raises ValueError."""
    clf = RABoostClassifier(random_state=42)
    X_train = np.random.randn(10, 5)
    y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    clf.fit(X_train, y_train)
    
    X_test = np.array([1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="X must be 2-dimensional"):
        clf.predict(X_test)


def test_classifier_predict_wrong_n_features():
    """Test that predicting with wrong number of features raises ValueError."""
    clf = RABoostClassifier(random_state=42)
    X_train = np.random.randn(10, 5)
    y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    clf.fit(X_train, y_train)
    
    X_test = np.random.randn(5, 3)  # Wrong number of features
    with pytest.raises(ValueError, match="X has 3 features, but model was trained with 5 features"):
        clf.predict(X_test)
