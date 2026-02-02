"""Tests for RABoostClassifier."""
import numpy as np
import pytest
from speedx.classifier import RABoostClassifier, DecisionStump


def test_classifier_initialization():
    """Test classifier initialization."""
    clf = RABoostClassifier(
        n_estimators=5,
        k_neighbors=3,
        embedding_dim=16,
        learning_rate=0.1,
        random_state=42
    )
    
    assert clf.n_estimators == 5
    assert clf.k_neighbors == 3
    assert clf.embedding_dim == 16
    assert clf.learning_rate == 0.1
    assert clf.random_state == 42


def test_classifier_fit_predict():
    """Test end-to-end fit and predict on synthetic data."""
    # Create synthetic dataset
    np.random.seed(42)
    
    n_samples = 100
    n_features = 10
    
    # Generate two clusters
    X1 = np.random.randn(n_samples // 2, n_features) + np.array([1.0] * n_features)
    X2 = np.random.randn(n_samples // 2, n_features) + np.array([-1.0] * n_features)
    X = np.vstack([X1, X2])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    # Fit classifier
    clf = RABoostClassifier(
        n_estimators=5,
        k_neighbors=3,
        embedding_dim=8,
        random_state=42
    )
    clf.fit(X, y)
    
    # Check fitted attributes
    assert clf.X_train_ is not None
    assert clf.y_train_ is not None
    assert clf.embeddings_train_ is not None
    assert clf.embedding_matrix_ is not None
    assert len(clf.estimators_) == 5
    assert len(clf.estimator_weights_) == 5
    assert len(clf.classes_) == 2
    
    # Predict
    y_pred = clf.predict(X)
    assert y_pred.shape == (n_samples,)
    assert set(y_pred).issubset(set(clf.classes_))
    
    # Check accuracy is above chance
    accuracy = (y_pred == y).mean()
    assert accuracy > 0.6, f"Accuracy {accuracy} is too low"


def test_classifier_predict_proba():
    """Test probability predictions."""
    np.random.seed(42)
    
    n_samples = 50
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 3, size=n_samples)
    
    clf = RABoostClassifier(
        n_estimators=3,
        k_neighbors=2,
        random_state=42
    )
    clf.fit(X, y)
    
    # Predict probabilities
    y_proba = clf.predict_proba(X)
    
    assert y_proba.shape == (n_samples, 3)
    assert np.allclose(y_proba.sum(axis=1), 1.0)
    assert (y_proba >= 0).all()
    assert (y_proba <= 1).all()


def test_embedding_reproducibility():
    """Test that embeddings are reproducible with fixed random state."""
    np.random.seed(42)
    
    X = np.random.randn(20, 5)
    y = np.random.randint(0, 2, size=20)
    
    # Fit two classifiers with same random state
    clf1 = RABoostClassifier(random_state=42)
    clf1.fit(X, y)
    
    clf2 = RABoostClassifier(random_state=42)
    clf2.fit(X, y)
    
    # Embeddings should be identical
    assert np.allclose(clf1.embeddings_train_, clf2.embeddings_train_)
    assert np.allclose(
        clf1.embedding_matrix_,
        clf2.embedding_matrix_
    )


def test_embedding_normalization():
    """Test that embeddings are L2 normalized."""
    np.random.seed(42)
    
    X = np.random.randn(10, 5)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    
    clf = RABoostClassifier(random_state=42)
    clf.fit(X, y)
    
    # Check L2 norm of embeddings
    norms = np.linalg.norm(clf.embeddings_train_, axis=1)
    assert np.allclose(norms, 1.0)


def test_knn_retrieval():
    """Test kNN retrieval functionality."""
    np.random.seed(42)
    
    X_train = np.random.randn(30, 5)
    y_train = np.random.randint(0, 2, size=30)
    
    clf = RABoostClassifier(k_neighbors=5, random_state=42)
    clf.fit(X_train, y_train)
    
    # Query with training data
    X_query = X_train[:3]
    embeddings_query = clf._generate_embedding(X_query)
    
    indices, distances = clf._knn_retrieve(embeddings_query, k=5)
    
    assert indices.shape == (3, 5)
    assert distances.shape == (3, 5)
    
    # First neighbor should be itself (or very close)
    assert (distances[:, 0] < 0.01).all()


def test_decision_stump():
    """Test DecisionStump weak learner."""
    np.random.seed(42)
    
    # Simple linearly separable data
    X = np.array([
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [4.0, 1.0],
        [5.0, 1.0],
        [6.0, 1.0],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    weights = np.ones(6) / 6
    
    stump = DecisionStump()
    stump.fit(X, y, weights)
    
    y_pred = stump.predict(X)
    
    # Should achieve perfect classification
    assert (y_pred == y).all()


def test_classifier_with_single_class():
    """Test classifier behavior with single class."""
    X = np.random.randn(10, 5)
    y = np.zeros(10)  # All same class
    
    clf = RABoostClassifier(random_state=42)
    clf.fit(X, y)
    
    y_pred = clf.predict(X)
    assert (y_pred == 0).all()
    
    y_proba = clf.predict_proba(X)
    assert y_proba.shape == (10, 1)


def test_classifier_multiclass():
    """Test classifier with multiple classes."""
    np.random.seed(42)
    
    n_samples = 90
    n_features = 5
    n_classes = 3
    
    X = []
    y = []
    for c in range(n_classes):
        X_c = np.random.randn(n_samples // n_classes, n_features) + c * 2
        y_c = np.full(n_samples // n_classes, c)
        X.append(X_c)
        y.append(y_c)
    
    X = np.vstack(X)
    y = np.concatenate(y)
    
    clf = RABoostClassifier(n_estimators=5, random_state=42)
    clf.fit(X, y)
    
    y_pred = clf.predict(X)
    
    # Should have reasonable accuracy
    accuracy = (y_pred == y).mean()
    assert accuracy > 0.5
