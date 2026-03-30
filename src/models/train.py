"""Model training with MLflow tracking."""

import logging
import pickle
import time
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import mlflow
import mlflow.sklearn
import yaml

logger = logging.getLogger(__name__)


class SentimentModelTrainer:
    """Train sentiment analysis model with MLflow tracking."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = None
        self.vectorizer = None
        
        # Set up MLflow
        mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
        mlflow.set_experiment(config['mlflow']['experiment_name'])
        
    def create_vectorizer(self) -> TfidfVectorizer:
        """
        Create TF-IDF vectorizer.
        
        Returns:
            Configured TfidfVectorizer
        """
        feature_config = self.config['features']
        
        vectorizer = TfidfVectorizer(
            max_features=self.config['data']['max_features'],
            ngram_range=tuple(feature_config['ngram_range']),
            min_df=feature_config['min_df'],
            max_df=feature_config['max_df']
        )
        
        logger.info(f"Created TF-IDF vectorizer with max_features={self.config['data']['max_features']}")
        
        return vectorizer
    
    def create_model(self) -> LogisticRegression:
        """
        Create model based on configuration.
        
        Returns:
            Initialized model
        """
        model_config = self.config['model']
        
        if model_config['type'] == 'logistic_regression':
            model = LogisticRegression(**model_config['params'])
            logger.info("Created Logistic Regression model")
        else:
            raise ValueError(f"Unsupported model type: {model_config['type']}")
        
        return model
    
    def train(
        self,
        X_train: pd.Series,
        y_train: pd.Series,
        X_val: pd.Series = None,
        y_val: pd.Series = None
    ) -> Tuple[Any, TfidfVectorizer]:
        """
        Train the sentiment analysis model.
        
        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Tuple of (trained_model, vectorizer)
        """
        with mlflow.start_run():
            logger.info("Starting model training...")
            
            # Log parameters
            mlflow.log_params({
                "model_type": self.config['model']['type'],
                "max_features": self.config['data']['max_features'],
                "ngram_range": str(self.config['features']['ngram_range']),
                **self.config['model']['params']
            })
            
            # Create and fit vectorizer
            start_time = time.time()
            self.vectorizer = self.create_vectorizer()
            X_train_vec = self.vectorizer.fit_transform(X_train)
            vectorization_time = time.time() - start_time
            
            logger.info(f"Vectorization complete. Shape: {X_train_vec.shape}")
            mlflow.log_metric("vectorization_time_seconds", vectorization_time)
            
            # Create and train model
            start_time = time.time()
            self.model = self.create_model()
            self.model.fit(X_train_vec, y_train)
            training_time = time.time() - start_time
            
            logger.info(f"Training complete in {training_time:.2f} seconds")
            mlflow.log_metric("training_time_seconds", training_time)
            
            # Evaluate on training set
            train_predictions = self.model.predict(X_train_vec)
            train_accuracy = accuracy_score(y_train, train_predictions)
            train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
                y_train, train_predictions, average='weighted'
            )
            
            logger.info(f"Training accuracy: {train_accuracy:.4f}")
            
            # Log training metrics
            mlflow.log_metrics({
                "train_accuracy": train_accuracy,
                "train_precision": train_precision,
                "train_recall": train_recall,
                "train_f1": train_f1
            })
            
            # Evaluate on validation set if provided
            if X_val is not None and y_val is not None:
                X_val_vec = self.vectorizer.transform(X_val)
                val_predictions = self.model.predict(X_val_vec)
                val_accuracy = accuracy_score(y_val, val_predictions)
                val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
                    y_val, val_predictions, average='weighted'
                )
                
                logger.info(f"Validation accuracy: {val_accuracy:.4f}")
                
                # Log validation metrics
                mlflow.log_metrics({
                    "val_accuracy": val_accuracy,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_f1": val_f1
                })
                
                # Check thresholds
                if val_accuracy < self.config['thresholds']['min_accuracy']:
                    logger.warning(
                        f"Validation accuracy {val_accuracy:.4f} below threshold "
                        f"{self.config['thresholds']['min_accuracy']}"
                    )
            
            # Log model
            mlflow.sklearn.log_model(self.model, "model")
            
            # Save vectorizer separately
            vectorizer_path = Path(self.config['paths']['models_dir']) / "vectorizer.pkl"
            vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            mlflow.log_artifact(str(vectorizer_path))
            
            logger.info("Model training complete and logged to MLflow")
            
        return self.model, self.vectorizer
    
    def predict(self, texts: list) -> np.ndarray:
        """
        Make predictions on new texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of predictions
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_vec = self.vectorizer.transform(texts)
        predictions = self.model.predict(X_vec)
        
        return predictions
    
    def predict_proba(self, texts: list) -> np.ndarray:
        """
        Get prediction probabilities for new texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of prediction probabilities
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_vec = self.vectorizer.transform(texts)
        probabilities = self.model.predict_proba(X_vec)
        
        return probabilities
    
    def save_model(self, model_path: str, vectorizer_path: str):
        """
        Save model and vectorizer to disk.
        
        Args:
            model_path: Path to save model
            vectorizer_path: Path to save vectorizer
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create directories
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(vectorizer_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save vectorizer
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Vectorizer saved to {vectorizer_path}")
    
    @staticmethod
    def load_model(model_path: str, vectorizer_path: str):
        """
        Load model and vectorizer from disk.
        
        Args:
            model_path: Path to model file
            vectorizer_path: Path to vectorizer file
            
        Returns:
            Tuple of (model, vectorizer)
        """
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Vectorizer loaded from {vectorizer_path}")
        
        return model, vectorizer


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create sample data
    X_train = pd.Series([
        "This movie is great",
        "Terrible film",
        "Amazing performance",
        "Waste of time"
    ])
    y_train = pd.Series([1, 0, 1, 0])
    
    # Train model
    trainer = SentimentModelTrainer(config)
    model, vectorizer = trainer.train(X_train, y_train)
    
    # Make predictions
    test_texts = ["This is wonderful", "This is awful"]
    predictions = trainer.predict(test_texts)
    probas = trainer.predict_proba(test_texts)
    
    print(f"\nPredictions: {predictions}")
    print(f"Probabilities: {probas}")