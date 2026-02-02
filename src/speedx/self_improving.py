"""
Self-improving AI capabilities for automatic optimization.

Features:
- Automatic hyperparameter tuning
- Adaptive learning rates
- Performance monitoring
- Self-optimization
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import time
from collections import deque
from .classifier import RAGBoostClassifier


class SelfImprovingClassifier:
    """
    Self-improving classifier that automatically optimizes itself.
    
    Features:
    - Automatic hyperparameter tuning
    - Adaptive learning rate
    - Performance tracking
    - Continuous improvement
    
    Parameters
    ----------
    base_params : dict, optional
        Initial parameters for base classifier.
    optimization_rounds : int, default=10
        Number of optimization iterations.
    performance_memory : int, default=50
        Number of recent performances to track.
    improvement_threshold : float, default=0.01
        Minimum improvement to trigger optimization.
    auto_tune : bool, default=True
        Whether to automatically tune hyperparameters.
    """
    
    def __init__(
        self,
        base_params: Optional[Dict[str, Any]] = None,
        optimization_rounds: int = 10,
        performance_memory: int = 50,
        improvement_threshold: float = 0.01,
        auto_tune: bool = True
    ):
        self.base_params = base_params or {
            'n_estimators': 10,
            'k_neighbors': 5,
            'embedding_dim': 32,
            'learning_rate': 0.1
        }
        self.optimization_rounds = optimization_rounds
        self.performance_memory = performance_memory
        self.improvement_threshold = improvement_threshold
        self.auto_tune = auto_tune
        
        # Internal state
        self.classifier_ = None
        self.best_params_ = None
        self.best_score_ = 0.0
        self.performance_history_ = deque(maxlen=performance_memory)
        self.optimization_history_ = []
        self.n_iterations_ = 0
    
    def _evaluate_params(
        self,
        params: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2
    ) -> float:
        """Evaluate a parameter configuration."""
        # Split data
        n_samples = len(X)
        n_val = int(n_samples * validation_split)
        
        # Random split
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
        
        # Train and evaluate
        try:
            clf = RAGBoostClassifier(**params)
            clf.fit(X_train, y_train)
            score = clf.score(X_val, y_val)
            return score
        except Exception:
            return 0.0
    
    def _optimize_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using random search.
        
        Returns best parameters found.
        """
        param_grid = {
            'n_estimators': [5, 10, 15, 20, 25],
            'k_neighbors': [3, 5, 7, 10, 15],
            'embedding_dim': [16, 32, 64, 128],
            'learning_rate': [0.05, 0.1, 0.15, 0.2]
        }
        
        best_score = 0.0
        best_params = self.base_params.copy()
        
        print(f"🔍 Optimizing hyperparameters over {self.optimization_rounds} rounds...")
        
        for i in range(self.optimization_rounds):
            # Random parameter selection
            test_params = {
                'n_estimators': np.random.choice(param_grid['n_estimators']),
                'k_neighbors': np.random.choice(param_grid['k_neighbors']),
                'embedding_dim': np.random.choice(param_grid['embedding_dim']),
                'learning_rate': np.random.choice(param_grid['learning_rate']),
                'random_state': 42
            }
            
            # Evaluate
            score = self._evaluate_params(test_params, X, y)
            
            if score > best_score:
                best_score = score
                best_params = test_params.copy()
                print(f"  ✓ Round {i+1}: New best score {score:.4f} with params {test_params}")
            else:
                print(f"  • Round {i+1}: Score {score:.4f}")
            
            # Store history
            self.optimization_history_.append({
                'round': i + 1,
                'params': test_params,
                'score': score
            })
        
        print(f"🎯 Best score: {best_score:.4f}")
        return best_params
    
    def _adapt_learning_rate(self) -> float:
        """
        Adapt learning rate based on recent performance.
        
        Returns adjusted learning rate.
        """
        if len(self.performance_history_) < 5:
            return self.base_params.get('learning_rate', 0.1)
        
        # Check if performance is improving
        recent_scores = list(self.performance_history_)[-5:]
        trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
        
        current_lr = self.base_params.get('learning_rate', 0.1)
        
        if trend > 0:
            # Performance improving - maintain or slightly increase
            new_lr = min(current_lr * 1.1, 0.5)
            print(f"📈 Performance improving (trend: {trend:.4f}), increasing LR: {current_lr:.4f} → {new_lr:.4f}")
        else:
            # Performance declining - reduce learning rate
            new_lr = max(current_lr * 0.9, 0.01)
            print(f"📉 Performance declining (trend: {trend:.4f}), reducing LR: {current_lr:.4f} → {new_lr:.4f}")
        
        return new_lr
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2
    ) -> "SelfImprovingClassifier":
        """
        Fit the self-improving classifier.
        
        Automatically optimizes hyperparameters if enabled.
        
        Parameters
        ----------
        X : np.ndarray
            Training features.
        y : np.ndarray
            Training labels.
        validation_split : float, default=0.2
            Fraction of data to use for validation.
            
        Returns
        -------
        self : SelfImprovingClassifier
        """
        print("🤖 Self-Improving AI Classifier - Starting training...")
        start_time = time.time()
        
        # Optimize hyperparameters if enabled
        if self.auto_tune and self.n_iterations_ == 0:
            self.best_params_ = self._optimize_hyperparameters(X, y)
        else:
            self.best_params_ = self.base_params.copy()
        
        # Adapt learning rate if we have history
        if self.n_iterations_ > 0:
            self.best_params_['learning_rate'] = self._adapt_learning_rate()
        
        # Train final model with best parameters
        print(f"🎓 Training final model with optimized parameters...")
        self.classifier_ = RAGBoostClassifier(**self.best_params_)
        self.classifier_.fit(X, y)
        
        # Evaluate performance
        score = self.classifier_.score(X, y)
        self.performance_history_.append(score)
        self.best_score_ = max(self.best_score_, score)
        self.n_iterations_ += 1
        
        elapsed = time.time() - start_time
        print(f"✅ Training complete in {elapsed:.2f}s")
        print(f"📊 Training accuracy: {score:.4f}")
        print(f"🔧 Final parameters: {self.best_params_}")
        
        return self
    
    def improve(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray
    ) -> "SelfImprovingClassifier":
        """
        Continuously improve the model with new data.
        
        Parameters
        ----------
        X_new : np.ndarray
            New training features.
        y_new : np.ndarray
            New training labels.
            
        Returns
        -------
        self : SelfImprovingClassifier
        """
        print(f"🔄 Continuous improvement iteration {self.n_iterations_ + 1}")
        
        # Combine with existing data if available
        if hasattr(self.classifier_, 'X_train_'):
            X_combined = np.vstack([self.classifier_.X_train_, X_new])
            y_combined = np.concatenate([self.classifier_.y_train_, y_new])
        else:
            X_combined = X_new
            y_combined = y_new
        
        # Retrain
        return self.fit(X_combined, y_combined)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self.classifier_ is None:
            raise ValueError("Model must be fitted before predicting")
        return self.classifier_.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.classifier_ is None:
            raise ValueError("Model must be fitted before predicting")
        return self.classifier_.predict_proba(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy score."""
        if self.classifier_ is None:
            raise ValueError("Model must be fitted before scoring")
        return self.classifier_.score(X, y)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive performance report.
        
        Returns
        -------
        report : dict
            Performance metrics and history.
        """
        if len(self.performance_history_) == 0:
            return {"status": "not_trained"}
        
        recent_scores = list(self.performance_history_)
        
        # Safely compute trend
        try:
            if len(recent_scores) > 1:
                trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
            else:
                trend = 0.0
        except (np.linalg.LinAlgError, ValueError):
            trend = 0.0
        
        return {
            "current_score": recent_scores[-1],
            "best_score": max(recent_scores),
            "worst_score": min(recent_scores),
            "average_score": np.mean(recent_scores),
            "improvement_trend": trend,
            "n_iterations": self.n_iterations_,
            "best_params": self.best_params_,
            "performance_history": recent_scores,
            "optimization_history": self.optimization_history_
        }


class AutoMLClassifier:
    """
    Automated Machine Learning classifier with model selection.
    
    Automatically selects the best model type and parameters.
    
    Parameters
    ----------
    models_to_try : list, optional
        List of model types to evaluate. Options: 'single', 'ensemble', 'stacked'.
    optimization_budget : int, default=20
        Total optimization budget (model evaluations).
    """
    
    def __init__(
        self,
        models_to_try: Optional[List[str]] = None,
        optimization_budget: int = 20
    ):
        self.models_to_try = models_to_try or ['single', 'ensemble']
        self.optimization_budget = optimization_budget
        
        self.best_model_ = None
        self.best_model_type_ = None
        self.best_score_ = 0.0
        self.evaluation_results_ = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "AutoMLClassifier":
        """
        Automatically find and train the best model.
        
        Parameters
        ----------
        X : np.ndarray
            Training features.
        y : np.ndarray
            Training labels.
            
        Returns
        -------
        self : AutoMLClassifier
        """
        from .ensemble import EnsembleClassifier, StackedClassifier
        
        print("🤖 AutoML - Automatic model selection and optimization")
        print(f"📊 Dataset: {len(X)} samples, {X.shape[1]} features")
        print(f"🎯 Models to evaluate: {self.models_to_try}")
        print()
        
        # Split data for validation
        n_val = int(len(X) * 0.2)
        indices = np.random.permutation(len(X))
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
        
        budget_per_model = self.optimization_budget // len(self.models_to_try)
        
        for model_type in self.models_to_try:
            print(f"🔍 Evaluating {model_type} model...")
            
            if model_type == 'single':
                # Evaluate single classifier with different params
                for i in range(budget_per_model):
                    params = {
                        'n_estimators': np.random.choice([5, 10, 15, 20]),
                        'k_neighbors': np.random.choice([3, 5, 7, 10]),
                        'embedding_dim': np.random.choice([16, 32, 64]),
                        'learning_rate': np.random.choice([0.05, 0.1, 0.15]),
                        'random_state': 42
                    }
                    
                    try:
                        clf = RAGBoostClassifier(**params)
                        clf.fit(X_train, y_train)
                        score = clf.score(X_val, y_val)
                        
                        self.evaluation_results_.append({
                            'model_type': model_type,
                            'params': params,
                            'score': score
                        })
                        
                        if score > self.best_score_:
                            self.best_score_ = score
                            self.best_model_ = clf
                            self.best_model_type_ = model_type
                            print(f"  ✓ New best! Score: {score:.4f}")
                    except Exception as e:
                        print(f"  ✗ Failed: {e}")
            
            elif model_type == 'ensemble':
                # Evaluate ensemble
                for i in range(budget_per_model):
                    n_models = np.random.choice([3, 5, 7])
                    voting = np.random.choice(['soft', 'hard'])
                    
                    try:
                        ensemble = EnsembleClassifier(
                            n_models=n_models,
                            voting=voting,
                            base_params={
                                'n_estimators': 10,
                                'k_neighbors': 5,
                                'random_state': 42
                            }
                        )
                        ensemble.fit(X_train, y_train)
                        score = ensemble.score(X_val, y_val)
                        
                        self.evaluation_results_.append({
                            'model_type': model_type,
                            'params': {'n_models': n_models, 'voting': voting},
                            'score': score
                        })
                        
                        if score > self.best_score_:
                            self.best_score_ = score
                            self.best_model_ = ensemble
                            self.best_model_type_ = model_type
                            print(f"  ✓ New best! Score: {score:.4f}")
                    except Exception as e:
                        print(f"  ✗ Failed: {e}")
        
        # Retrain best model on full data
        print()
        print(f"🏆 Best model: {self.best_model_type_}")
        print(f"📈 Best validation score: {self.best_score_:.4f}")
        print(f"🔧 Retraining on full dataset...")
        
        if self.best_model_type_ == 'single':
            self.best_model_.fit(X, y)
        elif self.best_model_type_ == 'ensemble':
            self.best_model_.fit(X, y)
        
        final_score = self.best_model_.score(X, y)
        print(f"✅ Final training score: {final_score:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with best model."""
        if self.best_model_ is None:
            raise ValueError("Model must be fitted before predicting")
        return self.best_model_.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities with best model."""
        if self.best_model_ is None:
            raise ValueError("Model must be fitted before predicting")
        return self.best_model_.predict_proba(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Score with best model."""
        if self.best_model_ is None:
            raise ValueError("Model must be fitted before scoring")
        return self.best_model_.score(X, y)
