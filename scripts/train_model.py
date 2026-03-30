#!/usr/bin/env python3
"""Main training script for sentiment analysis model."""

import sys
import logging
from pathlib import Path
import yaml
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.load_data import IMDBDataLoader, create_train_test_split
from src.data.preprocess import TextPreprocessor
from src.models.train import SentimentModelTrainer
from src.models.evaluate import evaluate_model


def setup_logging(config: dict):
    """Set up logging configuration."""
    log_dir = Path(config['paths']['logs_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=config['logging']['level'],
        format=config['logging']['format'],
        handlers=[
            logging.FileHandler(config['logging']['file']),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main(config_path: str = "configs/config.yaml"):
    """
    Main training pipeline.
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("Starting MLOps Sentiment Analysis Training Pipeline")
    logger.info("=" * 80)
    
    # Step 1: Load data
    logger.info("\n[Step 1/5] Loading data...")
    data_loader = IMDBDataLoader(data_dir=config['paths']['raw_data'])
    train_df, test_df = data_loader.load_data()
    
    # Get statistics
    train_stats = data_loader.get_data_statistics(train_df)
    logger.info(f"Training data statistics: {train_stats}")
    
    # Step 2: Preprocess data
    logger.info("\n[Step 2/5] Preprocessing data...")
    preprocessor = TextPreprocessor(
        lowercase=True,
        remove_stopwords=True,
        lemmatize=True,
        remove_special_chars=True
    )
    
    train_df = preprocessor.preprocess_dataframe(train_df, text_column='text')
    test_df = preprocessor.preprocess_dataframe(test_df, text_column='text')
    
    # Save preprocessed data
    processed_dir = Path(config['paths']['processed_data'])
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(processed_dir / 'train_processed.csv', index=False)
    test_df.to_csv(processed_dir / 'test_processed.csv', index=False)
    logger.info(f"Preprocessed data saved to {processed_dir}")
    
    # Step 3: Split data (validation from training)
    logger.info("\n[Step 3/5] Creating train/validation split...")
    train_data, val_data = create_train_test_split(
        train_df,
        test_size=0.2,
        random_state=config['data']['random_state']
    )
    
    X_train = train_data['text_clean']
    y_train = train_data['label']
    X_val = val_data['text_clean']
    y_val = val_data['label']
    X_test = test_df['text_clean']
    y_test = test_df['label']
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Step 4: Train model
    logger.info("\n[Step 4/5] Training model...")
    trainer = SentimentModelTrainer(config)
    model, vectorizer = trainer.train(X_train, y_train, X_val, y_val)
    
    # Step 5: Evaluate model
    logger.info("\n[Step 5/5] Evaluating model on test set...")
    
    # Transform test data
    X_test_vec = vectorizer.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_vec)
    y_pred_proba = model.predict_proba(X_test_vec)
    
    # Evaluate
    metrics = evaluate_model(y_test, y_pred, y_pred_proba)
    
    logger.info("\n" + "=" * 80)
    logger.info("Final Test Set Results:")
    logger.info("=" * 80)
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}")
    
    # Check if model meets thresholds
    if metrics['accuracy'] < config['thresholds']['min_accuracy']:
        logger.warning(
            f"⚠️  Model accuracy {metrics['accuracy']:.4f} is below threshold "
            f"{config['thresholds']['min_accuracy']}"
        )
        logger.warning("Model may not be suitable for production deployment")
    else:
        logger.info(f"✅ Model accuracy meets threshold requirements")
    
    # Save model
    model_dir = Path(config['paths']['models_dir'])
    model_path = model_dir / 'sentiment_model.pkl'
    vectorizer_path = model_dir / 'vectorizer.pkl'
    
    trainer.save_model(str(model_path), str(vectorizer_path))
    
    logger.info("\n" + "=" * 80)
    logger.info("Training pipeline completed successfully!")
    logger.info("=" * 80)
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Vectorizer saved to: {vectorizer_path}")
    logger.info(f"Check MLflow UI at: {config['mlflow']['tracking_uri']}")
    
    # Demo predictions
    logger.info("\n" + "=" * 80)
    logger.info("Demo Predictions:")
    logger.info("=" * 80)
    
    demo_texts = [
        "This movie is absolutely fantastic! I loved it!",
        "Terrible waste of time. Do not watch this.",
        "It was okay, nothing special but not terrible either."
    ]
    
    for text in demo_texts:
        pred = trainer.predict([text])[0]
        proba = trainer.predict_proba([text])[0]
        sentiment = "Positive" if pred == 1 else "Negative"
        confidence = max(proba)
        
        logger.info(f"\nText: '{text}'")
        logger.info(f"Prediction: {sentiment} (confidence: {confidence:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train sentiment analysis model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    main(config_path=args.config)