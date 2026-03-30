"""Model evaluation utilities."""

import logging
from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

logger = logging.getLogger(__name__)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray = None
) -> Dict[str, Any]:
    """
    Evaluate model performance with comprehensive metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities (optional)
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # ROC AUC (if probabilities provided)
    roc_auc = None
    if y_pred_proba is not None:
        try:
            # For binary classification
            if y_pred_proba.shape[1] == 2:
                roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:
                roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
        except Exception as e:
            logger.warning(f"Could not compute ROC AUC: {e}")
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist(),
        'roc_auc': roc_auc if roc_auc is not None else 'N/A'
    }
    
    logger.info(f"Evaluation metrics: Accuracy={accuracy:.4f}, F1={f1:.4f}")
    
    return metrics


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Print detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    """
    report = classification_report(y_true, y_pred, target_names=['Negative', 'Positive'])
    print("\nClassification Report:")
    print(report)


def check_performance_thresholds(
    metrics: Dict[str, Any],
    min_accuracy: float = 0.85,
    min_f1: float = 0.83
) -> bool:
    """
    Check if model meets performance thresholds.
    
    Args:
        metrics: Dictionary of evaluation metrics
        min_accuracy: Minimum required accuracy
        min_f1: Minimum required F1 score
        
    Returns:
        True if all thresholds are met
    """
    accuracy_ok = metrics['accuracy'] >= min_accuracy
    f1_ok = metrics['f1_score'] >= min_f1
    
    if not accuracy_ok:
        logger.warning(f"Accuracy {metrics['accuracy']:.4f} below threshold {min_accuracy}")
    
    if not f1_ok:
        logger.warning(f"F1 score {metrics['f1_score']:.4f} below threshold {min_f1}")
    
    return accuracy_ok and f1_ok


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Simulate predictions
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 0])
    y_pred_proba = np.array([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.85, 0.15],
        [0.6, 0.4],
        [0.3, 0.7],
        [0.95, 0.05]
    ])
    
    metrics = evaluate_model(y_true, y_pred, y_pred_proba)
    
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    print_classification_report(y_true, y_pred)
    
    thresholds_met = check_performance_thresholds(metrics)
    print(f"\nThresholds met: {thresholds_met}")