"""Tests for data loading."""

import pytest
import pandas as pd
from pathlib import Path
from src.data.load_data import IMDBDataLoader, create_train_test_split


class TestIMDBDataLoader:
    """Test suite for IMDBDataLoader class."""
    
    @pytest.fixture
    def loader(self):
        """Create a data loader instance for testing."""
        return IMDBDataLoader(data_dir="data/raw")
    
    def test_loader_initialization(self, loader):
        """Test that loader initializes correctly."""
        assert loader.data_dir == Path("data/raw")
        assert loader.dataset_url is not None
    
    def test_load_data_returns_dataframes(self, loader):
        """Test that load_data returns two DataFrames."""
        train_df, test_df = loader.load_data()
        
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)
    
    def test_data_has_required_columns(self, loader):
        """Test that loaded data has required columns."""
        train_df, test_df = loader.load_data()
        
        required_columns = ['text', 'label', 'sentiment']
        
        for col in required_columns:
            assert col in train_df.columns
            assert col in test_df.columns
    
    def test_data_not_empty(self, loader):
        """Test that loaded data is not empty."""
        train_df, test_df = loader.load_data()
        
        assert len(train_df) > 0
        assert len(test_df) > 0
    
    def test_labels_are_binary(self, loader):
        """Test that labels are 0 or 1."""
        train_df, _ = loader.load_data()
        
        unique_labels = train_df['label'].unique()
        assert set(unique_labels).issubset({0, 1})
    
    def test_balanced_dataset(self, loader):
        """Test that dataset is roughly balanced."""
        train_df, _ = loader.load_data()
        
        label_counts = train_df['label'].value_counts()
        
        # Check that classes are within 10% of each other
        ratio = label_counts.min() / label_counts.max()
        assert ratio > 0.9
    
    def test_sentiment_mapping(self, loader):
        """Test that sentiment labels are correctly mapped."""
        train_df, _ = loader.load_data()
        
        # Check label 0 maps to negative
        negative_rows = train_df[train_df['label'] == 0]
        assert all(negative_rows['sentiment'] == 'negative')
        
        # Check label 1 maps to positive
        positive_rows = train_df[train_df['label'] == 1]
        assert all(positive_rows['sentiment'] == 'positive')
    
    def test_get_data_statistics(self, loader):
        """Test data statistics calculation."""
        train_df, _ = loader.load_data()
        stats = loader.get_data_statistics(train_df)
        
        assert 'total_samples' in stats
        assert 'label_distribution' in stats
        assert 'avg_text_length' in stats
        assert 'max_text_length' in stats
        assert 'min_text_length' in stats
        
        assert stats['total_samples'] == len(train_df)


class TestTrainTestSplit:
    """Test suite for train/test split functionality."""
    
    def test_split_maintains_proportions(self):
        """Test that split maintains correct proportions."""
        # Create sample data
        df = pd.DataFrame({
            'text': ['sample'] * 100,
            'label': [0] * 50 + [1] * 50
        })
        
        train_df, test_df = create_train_test_split(df, test_size=0.2)
        
        # Check sizes
        assert len(train_df) == 80
        assert len(test_df) == 20
    
    def test_split_stratification(self):
        """Test that split maintains class balance."""
        # Create sample data
        df = pd.DataFrame({
            'text': ['sample'] * 100,
            'label': [0] * 50 + [1] * 50
        })
        
        train_df, test_df = create_train_test_split(df, test_size=0.2)
        
        # Check balance in both sets
        train_ratio = train_df['label'].value_counts()[0] / train_df['label'].value_counts()[1]
        test_ratio = test_df['label'].value_counts()[0] / test_df['label'].value_counts()[1]
        
        # Ratios should be close to 1.0 (balanced)
        assert 0.9 < train_ratio < 1.1
        assert 0.8 < test_ratio < 1.25  # Test set is smaller, allow more variance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
