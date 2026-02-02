"""
RABoostClassifier: Retrieval-Augmented Boosting Classifier.

A lightweight custom classifier that uses embedding-based retrieval
and boosting for fast inference on large astronomy datasets.
"""
import numpy as np
from typing import Optional, List, Tuple, Any
import warnings


class RABoostClassifier:
    """
    Retrieval-Augmented Boosting Classifier.
    
    This classifier combines:
    1. Embedding generation for feature representation
    2. kNN-based retrieval for context
    3. Simple boosting ensemble for prediction
    
    Parameters
    ----------
    n_estimators : int, default=10
        Number of boosting rounds.
    k_neighbors : int, default=5
        Number of neighbors to retrieve for augmentation.
    embedding_dim : int, default=32
        Dimension of the learned embedding space.
    learning_rate : float, default=0.1
        Learning rate for boosting.
    random_state : int, optional
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        n_estimators: int = 10,
        k_neighbors: int = 5,
        embedding_dim: int = 32,
        learning_rate: float = 0.1,
        random_state: Optional[int] = None
    ):
        # Input validation
        if n_estimators <= 0:
            raise ValueError(f"n_estimators must be positive, got {n_estimators}")
        if k_neighbors <= 0:
            raise ValueError(f"k_neighbors must be positive, got {k_neighbors}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        
        self.n_estimators = n_estimators
        self.k_neighbors = k_neighbors
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # Internal state
        self.embedding_matrix_ = None
        self.estimators_ = []
        self.estimator_weights_ = []
        self.classes_ = None
        self.X_train_ = None
        self.y_train_ = None
        self.embeddings_train_ = None
        
        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)
    
    def _generate_embedding(self, X: np.ndarray) -> np.ndarray:
        """
        Generate embeddings for input features.
        
        Uses a simple random projection for efficiency.
        In practice, this could be replaced with learned embeddings.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features.
            
        Returns
        -------
        embeddings : np.ndarray of shape (n_samples, embedding_dim)
            Generated embeddings.
        """
        n_features = X.shape[1]
        
        # Initialize embedding matrix if not done
        if self.embedding_matrix_ is None:
            # Random projection matrix (Johnson-Lindenstrauss)
            self.embedding_matrix_ = np.random.randn(
                n_features, self.embedding_dim
            ) / np.sqrt(n_features)
        
        # Project to embedding space
        embeddings = X @ self.embedding_matrix_
        
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        embeddings = embeddings / norms
        
        return embeddings
    
    def _knn_retrieve(
        self,
        query_embeddings: np.ndarray,
        k: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve k nearest neighbors for query embeddings.
        
        Uses exact nearest neighbor search with cosine similarity.
        
        Parameters
        ----------
        query_embeddings : np.ndarray of shape (n_queries, embedding_dim)
            Query embeddings.
        k : int, optional
            Number of neighbors. If None, uses self.k_neighbors.
            
        Returns
        -------
        indices : np.ndarray of shape (n_queries, k)
            Indices of k nearest neighbors.
        distances : np.ndarray of shape (n_queries, k)
            Distances to k nearest neighbors (cosine distance).
        """
        if k is None:
            k = self.k_neighbors
        
        # Compute cosine similarities (dot product of normalized vectors)
        similarities = query_embeddings @ self.embeddings_train_.T
        
        # Convert to distances
        distances = 1 - similarities
        
        # Get top-k indices
        # Use argpartition for efficiency when k << n
        n_train = self.embeddings_train_.shape[0]
        k_actual = min(k, n_train)
        
        if k_actual < n_train:
            # Partial sort is faster
            partition_indices = np.argpartition(distances, k_actual-1, axis=1)
            indices = partition_indices[:, :k_actual]
            
            # Sort the k nearest
            rows = np.arange(query_embeddings.shape[0])[:, np.newaxis]
            distances_k = distances[rows, indices]
            sorted_indices = np.argsort(distances_k, axis=1)
            indices = indices[rows, sorted_indices]
            distances_k = distances_k[rows, sorted_indices]
        else:
            # Sort all
            indices = np.argsort(distances, axis=1)[:, :k_actual]
            rows = np.arange(query_embeddings.shape[0])[:, np.newaxis]
            distances_k = distances[rows, indices]
        
        return indices, distances_k
    
    def _augment_features(
        self,
        X: np.ndarray,
        embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Augment features with retrieval-based context.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Original features.
        embeddings : np.ndarray of shape (n_samples, embedding_dim)
            Query embeddings.
            
        Returns
        -------
        X_augmented : np.ndarray of shape (n_samples, n_features + k_neighbors)
            Augmented features including neighbor labels.
        """
        indices, distances = self._knn_retrieve(embeddings)
        
        # Get neighbor labels
        neighbor_labels = self.y_train_[indices]
        
        # Weight by inverse distance
        weights = 1.0 / (distances + 1e-6)
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        # Compute weighted label features
        label_features = []
        for c in self.classes_:
            label_mask = (neighbor_labels == c).astype(float)
            weighted_label = (label_mask * weights).sum(axis=1, keepdims=True)
            label_features.append(weighted_label)
        
        label_features = np.hstack(label_features)
        
        # Concatenate with original features
        X_augmented = np.hstack([X, label_features])
        
        return X_augmented
    
    def _create_weak_learner(self) -> Any:
        """Create a weak learner (decision stump)."""
        return DecisionStump()
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RABoostClassifier":
        """
        Fit the RABoostClassifier.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Training labels.
            
        Returns
        -------
        self : RABoostClassifier
            Fitted classifier.
            
        Raises
        ------
        ValueError
            If X or y are empty, or if they have incompatible shapes.
            
        Examples
        --------
        >>> import numpy as np
        >>> from speedx import RABoostClassifier
        >>> X = np.random.randn(100, 10)
        >>> y = np.random.randint(0, 3, size=100)
        >>> clf = RABoostClassifier(n_estimators=10, random_state=42)
        >>> clf.fit(X, y)
        RABoostClassifier(...)
        >>> y_pred = clf.predict(X)
        >>> accuracy = (y_pred == y).mean()
        
        See Also
        --------
        predict : Predict class labels
        predict_proba : Predict class probabilities
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Input validation
        if X.size == 0:
            raise ValueError("X cannot be empty")
        if y.size == 0:
            raise ValueError("y cannot be empty")
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same length. Got X: {len(X)}, y: {len(y)}"
            )
        if len(X.shape) != 2:
            raise ValueError(
                f"X must be 2-dimensional. Got shape: {X.shape}"
            )
        
        n_samples = X.shape[0]
        
        # Store training data
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        self.classes_ = np.unique(y)
        
        # Generate embeddings
        self.embeddings_train_ = self._generate_embedding(X)
        
        # Initialize sample weights
        sample_weights = np.ones(n_samples) / n_samples
        
        # Boosting rounds
        self.estimators_ = []
        self.estimator_weights_ = []
        
        for m in range(self.n_estimators):
            # Augment features with retrieval context
            X_aug = self._augment_features(X, self.embeddings_train_)
            
            # Train weak learner
            weak_learner = self._create_weak_learner()
            weak_learner.fit(X_aug, y, sample_weights)
            
            # Predict
            y_pred = weak_learner.predict(X_aug)
            
            # Compute error
            incorrect = y_pred != y
            error = (sample_weights * incorrect).sum()
            
            # Avoid division by zero
            error = np.clip(error, 1e-10, 1 - 1e-10)
            
            # Compute estimator weight
            estimator_weight = self.learning_rate * 0.5 * np.log((1 - error) / error)
            
            # Update sample weights
            sample_weights *= np.exp(estimator_weight * (2 * incorrect - 1))
            sample_weights /= sample_weights.sum()
            
            # Store estimator
            self.estimators_.append(weak_learner)
            self.estimator_weights_.append(estimator_weight)
        
        return self
    
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
            
        Raises
        ------
        ValueError
            If called before fitting or if X has wrong shape.
        """
        if self.X_train_ is None:
            raise ValueError("Model must be fitted before predicting. Call fit() first.")
        
        X = np.asarray(X)
        
        if X.size == 0:
            raise ValueError("X cannot be empty")
        if len(X.shape) != 2:
            raise ValueError(f"X must be 2-dimensional. Got shape: {X.shape}")
        if X.shape[1] != self.X_train_.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was trained with "
                f"{self.X_train_.shape[1]} features"
            )
        
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        # Generate embeddings
        embeddings = self._generate_embedding(X)
        
        # Augment features
        X_aug = self._augment_features(X, embeddings)
        
        # Accumulate predictions from all estimators
        scores = np.zeros((n_samples, n_classes))
        
        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            y_pred = estimator.predict(X_aug)
            for i, c in enumerate(self.classes_):
                scores[y_pred == c, i] += weight
        
        # Convert to probabilities
        proba = np.exp(scores)
        proba /= proba.sum(axis=1, keepdims=True)
        
        return proba
    
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


class DecisionStump:
    """
    Simple decision stump (one-level decision tree).
    
    Finds the best feature and threshold to split on.
    """
    
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.left_label = None
        self.right_label = None
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray
    ) -> "DecisionStump":
        """Fit the decision stump."""
        n_samples, n_features = X.shape
        classes = np.unique(y)
        
        best_error = float('inf')
        
        # Try each feature
        for feat_idx in range(n_features):
            feature_values = X[:, feat_idx]
            thresholds = np.unique(feature_values)
            
            # Try each threshold
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                # Predict based on weighted majority
                left_label = self._weighted_majority(y[left_mask], sample_weights[left_mask], classes)
                right_label = self._weighted_majority(y[right_mask], sample_weights[right_mask], classes)
                
                # Compute error
                y_pred = np.where(left_mask, left_label, right_label)
                error = (sample_weights * (y_pred != y)).sum()
                
                if error < best_error:
                    best_error = error
                    self.feature_idx = feat_idx
                    self.threshold = threshold
                    self.left_label = left_label
                    self.right_label = right_label
        
        return self
    
    def _weighted_majority(
        self,
        y: np.ndarray,
        weights: np.ndarray,
        classes: np.ndarray
    ) -> int:
        """Find the weighted majority class."""
        if len(y) == 0:
            return classes[0]
        
        class_weights = np.array([
            weights[y == c].sum() for c in classes
        ])
        return classes[np.argmax(class_weights)]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        feature_values = X[:, self.feature_idx]
        left_mask = feature_values <= self.threshold
        return np.where(left_mask, self.left_label, self.right_label)
