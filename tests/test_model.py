"""Tests for model training and evaluation."""

import pytest
import numpy as np
import pandas as pd
from src.models.evaluate import evaluate_model, check_performance_thresholds


class TestModelEvaluation:
    """Test suite for model evaluation functions."""
    
    def test_evaluate_model_basic(self):
        """Test basic model evaluation."""
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 0])
        
        metrics = evaluate_model(y_true, y_pred)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'confusion_matrix' in metrics
    
    def test_evaluate_model_with_probabilities(self):
        """Test evaluation with probability predictions."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_pred_proba = np.array([
            [0.9, 0.1],
            [0.2, 0.8],
            [0.85, 0.15],
            [0.3, 0.7]
        ])
        
        metrics = evaluate_model(y_true, y_pred, y_pred_proba)
        
        assert 'roc_auc' in metrics
        assert isinstance(metrics['roc_auc'], float)
        assert 0 <= metrics['roc_auc'] <= 1
    
    def test_perfect_predictions(self):
        """Test metrics for perfect predictions."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        
        metrics = evaluate_model(y_true, y_pred)
        
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
    
    def test_check_performance_thresholds_pass(self):
        """Test threshold checking when thresholds are met."""
        metrics = {
            'accuracy': 0.90,
            'f1_score': 0.88
        }
        
        result = check_performance_thresholds(metrics, min_accuracy=0.85, min_f1=0.83)
        assert result is True
    
    def test_check_performance_thresholds_fail_accuracy(self):
        """Test threshold checking when accuracy fails."""
        metrics = {
            'accuracy': 0.80,
            'f1_score': 0.88
        }
        
        result = check_performance_thresholds(metrics, min_accuracy=0.85, min_f1=0.83)
        assert result is False
    
    def test_check_performance_thresholds_fail_f1(self):
        """Test threshold checking when F1 fails."""
        metrics = {
            'accuracy': 0.90,
            'f1_score': 0.80
        }
        
        result = check_performance_thresholds(metrics, min_accuracy=0.85, min_f1=0.83)
        assert result is False
    
    def test_confusion_matrix_shape(self):
        """Test that confusion matrix has correct shape."""
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 0])
        
        metrics = evaluate_model(y_true, y_pred)
        
        cm = metrics['confusion_matrix']
        assert len(cm) == 2
        assert len(cm[0]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
