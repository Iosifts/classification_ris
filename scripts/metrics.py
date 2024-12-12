# scripts/metrics.py

from sklearn.metrics import recall_score, precision_recall_fscore_support, confusion_matrix
import torch
import pandas as pd

def compute_metrics(eval_pred, label_encoder):
    """
    Compute Unweighted Average Recall (UAR) and per-class accuracy.

    Args:
        eval_pred (tuple): Tuple containing logits and true labels.
        label_encoder (LabelEncoder): An instance of LabelEncoder.

    Returns:
        dict: Dictionary containing UAR and per-class accuracy.
    """
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=1).numpy()
    uar = recall_score(labels, preds, average="macro")
    
    # Per-class accuracy (same as recall)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=range(len(label_encoder.classes_)), zero_division=0
    )
    
    metrics = {"uar": uar}
    for idx, cls in enumerate(label_encoder.classes_):
        metrics[f"accuracy_class_{cls}"] = recall[idx]  # Accuracy per class
    
    return metrics

def generate_confusion_matrix(labels, preds, label_encoder):
    """
    Generate a confusion matrix as a pandas DataFrame.

    Args:
        labels (numpy.ndarray): True labels.
        preds (numpy.ndarray): Predicted labels.
        label_encoder (LabelEncoder): An instance of LabelEncoder.

    Returns:
        pandas.DataFrame: Confusion matrix with class names as indices and columns.
    """
    cm = confusion_matrix(labels, preds, labels=range(len(label_encoder.classes_)))
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
    return cm_df
