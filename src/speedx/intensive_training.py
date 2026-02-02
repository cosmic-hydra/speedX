"""
Intensive online training with automatic data fetching and augmentation.

Provides tools for continuous learning from online astronomy data sources.
"""
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Callable
import time
from pathlib import Path
import requests
from .mast_client import MASTClient
from .features import get_default_featurizer
from .self_improving import SelfImprovingClassifier


class OnlineDataTrainer:
    """
    Intensive trainer that continuously learns from online data.
    
    Features:
    - Automatic data fetching from MAST
    - Data augmentation
    - Continuous training
    - Performance monitoring
    - Automatic checkpointing
    
    Parameters
    ----------
    classifier : SelfImprovingClassifier
        Classifier to train.
    mast_client : MASTClient, optional
        MAST client for data fetching.
    batch_size : int, default=100
        Number of samples per training batch.
    augmentation_factor : int, default=3
        Number of augmented versions per sample.
    checkpoint_dir : str, optional
        Directory to save checkpoints.
    verbose : bool, default=True
        Whether to print progress.
    """
    
    def __init__(
        self,
        classifier: Optional[SelfImprovingClassifier] = None,
        mast_client: Optional[MASTClient] = None,
        batch_size: int = 100,
        augmentation_factor: int = 3,
        checkpoint_dir: Optional[str] = None,
        verbose: bool = True
    ):
        self.classifier = classifier or SelfImprovingClassifier(
            optimization_rounds=5,
            auto_tune=True
        )
        self.mast_client = mast_client or MASTClient()
        self.batch_size = batch_size
        self.augmentation_factor = augmentation_factor
        self.checkpoint_dir = checkpoint_dir
        self.verbose = verbose
        
        # Training state
        self.n_batches_trained_ = 0
        self.training_history_ = []
        self.best_score_ = 0.0
        
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def _log(self, message: str):
        """Log message if verbose."""
        if self.verbose:
            print(f"[{time.strftime('%H:%M:%S')}] {message}")
    
    def _augment_data(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> tuple:
        """
        Augment training data.
        
        Applies various augmentation techniques:
        - Gaussian noise injection
        - Feature scaling variations
        - Feature dropout
        - Mixup between samples
        """
        X_aug_list = [X]
        y_aug_list = [y]
        
        for _ in range(self.augmentation_factor - 1):
            # Gaussian noise
            noise = np.random.randn(*X.shape) * 0.1
            X_noisy = X + noise
            X_aug_list.append(X_noisy)
            y_aug_list.append(y)
            
            # Feature scaling
            scale = np.random.uniform(0.9, 1.1, X.shape[1])
            X_scaled = X * scale
            X_aug_list.append(X_scaled)
            y_aug_list.append(y)
            
            # Feature dropout (random masking)
            mask = np.random.binomial(1, 0.9, X.shape)
            X_masked = X * mask
            X_aug_list.append(X_masked)
            y_aug_list.append(y)
        
        X_augmented = np.vstack(X_aug_list)
        y_augmented = np.concatenate(y_aug_list)
        
        # Shuffle
        indices = np.random.permutation(len(X_augmented))
        return X_augmented[indices], y_augmented[indices]
    
    def fetch_training_batch(
        self,
        instrument: Optional[str] = None,
        n_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch a batch of training data from MAST.
        
        Parameters
        ----------
        instrument : str, optional
            Instrument filter.
        n_samples : int, optional
            Number of samples to fetch.
            
        Returns
        -------
        df : pd.DataFrame
            Fetched observations.
        """
        self._log(f"🌐 Fetching batch from MAST...")
        
        n_samples = n_samples or self.batch_size
        
        try:
            observations = self.mast_client.query_observations(
                instrument=instrument,
                max_records=n_samples
            )
            
            if not observations.empty:
                self._log(f"✓ Fetched {len(observations)} observations")
                return observations
            else:
                self._log("⚠ No observations fetched")
                return pd.DataFrame()
                
        except Exception as e:
            self._log(f"✗ Error fetching data: {e}")
            return pd.DataFrame()
    
    def generate_synthetic_labels(
        self,
        df: pd.DataFrame
    ) -> np.ndarray:
        """
        Generate synthetic labels for unsupervised learning.
        
        Uses clustering-like approach based on exposure time and instrument.
        
        Parameters
        ----------
        df : pd.DataFrame
            Observation metadata.
            
        Returns
        -------
        labels : np.ndarray
            Synthetic labels.
        """
        self._log("🏷️  Generating synthetic labels...")
        
        labels = np.zeros(len(df), dtype=int)
        
        if 't_exptime' in df.columns:
            # Label based on exposure time ranges
            exptime = df['t_exptime'].fillna(0)
            labels[exptime < 100] = 0  # Short exposure
            labels[(exptime >= 100) & (exptime < 300)] = 1  # Medium
            labels[exptime >= 300] = 2  # Long exposure
        
        if 'instrument_name' in df.columns:
            # Refine labels with instrument info
            for i, inst in enumerate(df['instrument_name'].unique()):
                mask = df['instrument_name'] == inst
                labels[mask] = labels[mask] + (i * 3)
        
        # Ensure labels are in valid range
        labels = labels % 10  # Keep labels 0-9
        
        self._log(f"✓ Generated labels: {np.bincount(labels)}")
        return labels
    
    def intensive_train(
        self,
        n_iterations: int = 10,
        instrument: Optional[str] = None,
        use_augmentation: bool = True,
        save_checkpoints: bool = True
    ):
        """
        Intensive training loop with online data.
        
        Parameters
        ----------
        n_iterations : int, default=10
            Number of training iterations.
        instrument : str, optional
            Instrument filter for MAST queries.
        use_augmentation : bool, default=True
            Whether to use data augmentation.
        save_checkpoints : bool, default=True
            Whether to save periodic checkpoints.
        """
        self._log("=" * 80)
        self._log("🚀 INTENSIVE ONLINE TRAINING")
        self._log("=" * 80)
        self._log(f"Iterations: {n_iterations}")
        self._log(f"Batch size: {self.batch_size}")
        self._log(f"Augmentation: {'ON' if use_augmentation else 'OFF'}")
        self._log("")
        
        for iteration in range(n_iterations):
            self._log(f"\n{'='*80}")
            self._log(f"📊 ITERATION {iteration + 1}/{n_iterations}")
            self._log(f"{'='*80}")
            
            # Fetch data
            df = self.fetch_training_batch(instrument=instrument)
            
            if df.empty:
                self._log("⚠ No data fetched, skipping iteration")
                continue
            
            # Extract features
            self._log("🔧 Extracting features...")
            featurizer = get_default_featurizer()
            X = featurizer.fit_transform(df)
            
            # Generate labels
            y = self.generate_synthetic_labels(df)
            
            # Augment data if enabled
            if use_augmentation:
                self._log(f"🎨 Augmenting data (factor: {self.augmentation_factor})...")
                X, y = self._augment_data(X, y)
                self._log(f"✓ Augmented to {len(X)} samples")
            
            # Train/improve model
            if iteration == 0:
                self._log("🎓 Initial training...")
                self.classifier.fit(X, y)
            else:
                self._log("🔄 Continuous improvement...")
                self.classifier.improve(X, y)
            
            # Evaluate
            score = self.classifier.score(X, y)
            self._log(f"📈 Current score: {score:.4f}")
            
            # Track progress
            self.training_history_.append({
                'iteration': iteration + 1,
                'n_samples': len(X),
                'score': score,
                'timestamp': time.time()
            })
            
            # Update best score
            if score > self.best_score_:
                self.best_score_ = score
                self._log(f"🏆 New best score: {score:.4f}!")
                
                if save_checkpoints and self.checkpoint_dir:
                    checkpoint_path = Path(self.checkpoint_dir) / "best_model.pkl"
                    self.classifier.classifier_.save(str(checkpoint_path))
                    self._log(f"💾 Saved checkpoint: {checkpoint_path}")
            
            self.n_batches_trained_ += 1
            
            # Save periodic checkpoint
            if save_checkpoints and self.checkpoint_dir and (iteration + 1) % 5 == 0:
                checkpoint_path = Path(self.checkpoint_dir) / f"checkpoint_iter_{iteration+1}.pkl"
                self.classifier.classifier_.save(str(checkpoint_path))
                self._log(f"💾 Saved periodic checkpoint: {checkpoint_path}")
        
        self._log("\n" + "=" * 80)
        self._log("✅ INTENSIVE TRAINING COMPLETE")
        self._log("=" * 80)
        self._log(f"Total batches trained: {self.n_batches_trained_}")
        self._log(f"Best score achieved: {self.best_score_:.4f}")
        self._log(f"Final score: {self.training_history_[-1]['score']:.4f}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get summary of training progress.
        
        Returns
        -------
        summary : dict
            Training metrics and history.
        """
        if not self.training_history_:
            return {"status": "not_trained"}
        
        scores = [h['score'] for h in self.training_history_]
        
        return {
            "n_iterations": len(self.training_history_),
            "n_batches": self.n_batches_trained_,
            "best_score": self.best_score_,
            "current_score": scores[-1],
            "average_score": np.mean(scores),
            "improvement": scores[-1] - scores[0] if len(scores) > 1 else 0.0,
            "training_history": self.training_history_,
            "classifier_params": self.classifier.best_params_
        }


class TransferLearningTrainer:
    """
    Transfer learning from pre-trained models.
    
    Enables faster training by leveraging knowledge from related tasks.
    
    Parameters
    ----------
    source_classifier : SelfImprovingClassifier
        Pre-trained source classifier.
    freeze_embeddings : bool, default=False
        Whether to freeze embedding layers.
    """
    
    def __init__(
        self,
        source_classifier: Optional[SelfImprovingClassifier] = None,
        freeze_embeddings: bool = False
    ):
        self.source_classifier = source_classifier
        self.freeze_embeddings = freeze_embeddings
        self.target_classifier_ = None
    
    def transfer_and_train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fine_tune_epochs: int = 5
    ) -> SelfImprovingClassifier:
        """
        Transfer knowledge and fine-tune on target data.
        
        Parameters
        ----------
        X : np.ndarray
            Target training features.
        y : np.ndarray
            Target training labels.
        fine_tune_epochs : int, default=5
            Number of fine-tuning epochs.
            
        Returns
        -------
        target_classifier : SelfImprovingClassifier
            Fine-tuned classifier.
        """
        print("🔄 Transfer Learning")
        print(f"Source model: {'Available' if self.source_classifier else 'None'}")
        print(f"Target data: {len(X)} samples")
        
        # Initialize target classifier
        if self.source_classifier and self.source_classifier.best_params_:
            # Use source model parameters as starting point
            print("✓ Transferring hyperparameters from source model")
            self.target_classifier_ = SelfImprovingClassifier(
                base_params=self.source_classifier.best_params_.copy(),
                optimization_rounds=3,
                auto_tune=False  # Use transferred params
            )
        else:
            # Start from scratch
            print("⚠ No source model, training from scratch")
            self.target_classifier_ = SelfImprovingClassifier(
                optimization_rounds=5,
                auto_tune=True
            )
        
        # Train on target data
        print(f"🎓 Fine-tuning for {fine_tune_epochs} epochs...")
        for epoch in range(fine_tune_epochs):
            print(f"\nEpoch {epoch + 1}/{fine_tune_epochs}")
            self.target_classifier_.fit(X, y)
            score = self.target_classifier_.score(X, y)
            print(f"Score: {score:.4f}")
        
        print("\n✅ Transfer learning complete")
        return self.target_classifier_


def create_intensive_training_pipeline(
    output_dir: str = "./training_output",
    n_iterations: int = 20,
    batch_size: int = 100,
    instruments: Optional[List[str]] = None
):
    """
    Create and run an intensive training pipeline.
    
    Parameters
    ----------
    output_dir : str
        Directory for checkpoints and logs.
    n_iterations : int
        Number of training iterations.
    batch_size : int
        Samples per batch.
    instruments : list of str, optional
        Instruments to query. If None, uses common HST instruments.
    """
    print("=" * 80)
    print("🚀 INTENSIVE TRAINING PIPELINE")
    print("=" * 80)
    
    # Default instruments
    if instruments is None:
        instruments = ["ACS/WFC", "WFC3/UVIS", "WFC3/IR"]
    
    # Create trainer
    trainer = OnlineDataTrainer(
        batch_size=batch_size,
        augmentation_factor=3,
        checkpoint_dir=output_dir,
        verbose=True
    )
    
    # Train on each instrument
    for instrument in instruments:
        print(f"\n{'='*80}")
        print(f"📡 Training on {instrument}")
        print(f"{'='*80}")
        
        trainer.intensive_train(
            n_iterations=n_iterations // len(instruments),
            instrument=instrument,
            use_augmentation=True,
            save_checkpoints=True
        )
    
    # Final mixed training
    print(f"\n{'='*80}")
    print("🎯 Final mixed training (all instruments)")
    print(f"{'='*80}")
    
    trainer.intensive_train(
        n_iterations=5,
        instrument=None,  # All instruments
        use_augmentation=True,
        save_checkpoints=True
    )
    
    # Print summary
    summary = trainer.get_training_summary()
    print("\n" + "=" * 80)
    print("📊 TRAINING SUMMARY")
    print("=" * 80)
    print(f"Total iterations: {summary['n_iterations']}")
    print(f"Total batches: {summary['n_batches']}")
    print(f"Best score: {summary['best_score']:.4f}")
    print(f"Final score: {summary['current_score']:.4f}")
    print(f"Total improvement: {summary['improvement']:.4f}")
    print("=" * 80)
    
    return trainer
