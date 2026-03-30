"""Data loading utilities for IMDB sentiment dataset."""

import os
import logging
import tarfile
import urllib.request
from pathlib import Path
from typing import Tuple, List
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class IMDBDataLoader:
    """Load and prepare IMDB movie review dataset."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing the IMDB dataset
        """
        self.data_dir = Path(data_dir)
        self.dataset_url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        
    def download_data(self) -> None:
        """
        Download IMDB dataset from Stanford if not already present.
        """
        # Check if data already exists
        train_csv = self.data_dir / 'train.csv'
        test_csv = self.data_dir / 'test.csv'
        
        if train_csv.exists() and test_csv.exists():
            logger.info(f"Data already exists at {self.data_dir}")
            return
        
        logger.info("Downloading IMDB dataset from Stanford...")
        
        # Create directory
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Download tar file
        tar_path = self.data_dir / 'aclImdb_v1.tar.gz'
        if not tar_path.exists():
            logger.info("Downloading... this may take a few minutes")
            urllib.request.urlretrieve(self.dataset_url, tar_path)
            logger.info("Download complete!")
        
        # Extract tar file
        logger.info("Extracting dataset...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(self.data_dir)
        
        # Process the extracted files
        logger.info("Processing files...")
        aclImdb_dir = self.data_dir / 'aclImdb'
        
        # Load training data
        train_data = self._load_reviews(aclImdb_dir / 'train')
        train_df = pd.DataFrame(train_data, columns=['text', 'label'])
        train_df.to_csv(train_csv, index=False)
        
        # Load test data
        test_data = self._load_reviews(aclImdb_dir / 'test')
        test_df = pd.DataFrame(test_data, columns=['text', 'label'])
        test_df.to_csv(test_csv, index=False)
        
        logger.info(f"Dataset downloaded and saved to {self.data_dir}")
        logger.info(f"Train: {len(train_df)} samples, Test: {len(test_df)} samples")
        
    def _load_reviews(self, data_path: Path) -> List[Tuple[str, int]]:
        """
        Load reviews from directory structure.
        
        Args:
            data_path: Path to train or test directory
            
        Returns:
            List of (text, label) tuples
        """
        reviews = []
        
        # Load positive reviews
        pos_dir = data_path / 'pos'
        if pos_dir.exists():
            for file_path in pos_dir.glob('*.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    reviews.append((text, 1))  # 1 = positive
        
        # Load negative reviews
        neg_dir = data_path / 'neg'
        if neg_dir.exists():
            for file_path in neg_dir.glob('*.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    reviews.append((text, 0))  # 0 = negative
        
        return reviews
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the IMDB dataset.
        
        Returns:
            Tuple of (train_df, test_df)
        """
        # Download if needed
        if not (self.data_dir / 'train.csv').exists():
            self.download_data()
            
        logger.info("Loading IMDB dataset...")
        
        train_df = pd.read_csv(self.data_dir / 'train.csv')
        test_df = pd.read_csv(self.data_dir / 'test.csv')
        
        # Map labels: 0=negative, 1=positive
        train_df['sentiment'] = train_df['label'].map({0: 'negative', 1: 'positive'})
        test_df['sentiment'] = test_df['label'].map({0: 'negative', 1: 'positive'})
        
        logger.info(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples")
        
        return train_df, test_df
    
    def load_custom_data(self, file_path: str) -> pd.DataFrame:
        """
        Load custom data from CSV file.
        
        Expected format: CSV with 'text' and 'label' columns
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame with text and labels
        """
        logger.info(f"Loading custom data from {file_path}")
        df = pd.read_csv(file_path)
        
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV must contain 'text' and 'label' columns")
            
        return df
    
    def get_data_statistics(self, df: pd.DataFrame) -> dict:
        """
        Get statistics about the dataset.
        
        Args:
            df: DataFrame with text and labels
            
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'total_samples': len(df),
            'label_distribution': df['label'].value_counts().to_dict(),
            'avg_text_length': df['text'].str.len().mean(),
            'max_text_length': df['text'].str.len().max(),
            'min_text_length': df['text'].str.len().min(),
        }
        
        logger.info(f"Dataset statistics: {stats}")
        return stats


def create_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets.
    
    Args:
        df: Input DataFrame
        test_size: Proportion of test data
        random_state: Random seed
        
    Returns:
        Tuple of (train_df, test_df)
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['label']
    )
    
    logger.info(f"Split data: {len(train_df)} train, {len(test_df)} test")
    
    return train_df, test_df


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    loader = IMDBDataLoader()
    train_df, test_df = loader.load_data()
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"\nSample data:\n{train_df.head()}")
    
    stats = loader.get_data_statistics(train_df)
    print(f"\nStatistics:\n{stats}")