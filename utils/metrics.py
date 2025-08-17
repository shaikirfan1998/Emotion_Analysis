# metrics.py

"""
Metrics and evaluation utilities for emotion recognition
"""

import os
import json
import time
from typing import List, Dict, Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

import config


def parse_llm_output(output: str, labels: List[str]) -> int:
    """
    Parse the LLM output to extract the predicted emotion label

    Args:
        output: Raw output from the model
        labels: List of label names, index corresponds to label id

    Returns:
        Predicted label index or -1 if no match found
    """
    output = output.strip().lower()
    # Build mapping from lowercase label → index
    label_to_idx = {lbl.lower(): idx for idx, lbl in enumerate(labels)}

    # Direct substring match
    for lbl, idx in label_to_idx.items():
        if lbl in output:
            return idx

    # Word-boundary match
    for lbl, idx in label_to_idx.items():
        if f" {lbl} " in f" {output} " or f" {lbl}." in output or f" {lbl}," in output:
            return idx

    # No match found
    return -1


def evaluate_predictions(
    y_true: List[int],
    y_pred: List[int],
    labels: List[str]
) -> Dict[str, Any]:
    """
    Evaluate predictions and calculate metrics

    Args:
        y_true: List of true label indices
        y_pred: List of predicted label indices
        labels: List of label names, index corresponds to label id

    Returns:
        Dictionary of evaluation metrics
    """
    # Filter out invalid predictions (-1)
    valid_indices = [i for i, p in enumerate(y_pred) if p != -1]
    filtered_y_true = [y_true[i] for i in valid_indices]
    filtered_y_pred = [y_pred[i] for i in valid_indices]

    if not filtered_y_pred:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "invalid_predictions": len(y_true),
            "total_examples": len(y_true),
            "confusion_matrix": None
        }

    # Overall metrics
    accuracy = accuracy_score(filtered_y_true, filtered_y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        filtered_y_true,
        filtered_y_pred,
        average='weighted'
    )

    # Per-class metrics
    class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
        filtered_y_true,
        filtered_y_pred,
        average=None,
        labels=list(range(len(labels)))
    )

    per_class_metrics: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(labels):
        if i in set(filtered_y_true + filtered_y_pred):
            per_class_metrics[name] = {
                "precision": class_precision[i] if i < len(class_precision) else 0.0,
                "recall":    class_recall[i]   if i < len(class_recall)   else 0.0,
                "f1":        class_f1[i]       if i < len(class_f1)       else 0.0
            }

    # Confusion matrix
    cm = confusion_matrix(
        filtered_y_true,
        filtered_y_pred,
        labels=list(range(len(labels)))
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_class_metrics": per_class_metrics,
        "invalid_predictions": len(y_true) - len(filtered_y_true),
        "total_examples": len(y_true),
        "confusion_matrix": cm.tolist()
    }


def save_metrics(
    metrics: Dict[str, Any],
    model_name: str,
    experiment_type: str,
    timestamp: str = None
) -> str:
    """
    Save evaluation metrics to a JSON file
    """
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

    output_dir = os.path.join(config.RESULTS_DIR, experiment_type)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"{model_name}_{timestamp}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    return output_file


def plot_confusion_matrix(
    cm: List[List[int]],
    labels: List[str],
    model_name: str,
    experiment_type: str,
    timestamp: str = None
) -> str:
    """
    Plot confusion matrix and save to file
    """
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

    output_dir = os.path.join(config.RESULTS_DIR, experiment_type)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"{model_name}_cm_{timestamp}.png")
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix: {model_name} ({experiment_type})')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    return output_file


def generate_report(
    model_name: str,
    experiment_type: str,
    metrics: Dict[str, Any]
) -> str:
    """
    Generate a human-readable report from metrics
    """
    report = f"# Evaluation Report: {model_name} ({experiment_type})\n\n"
    report += "## Overall Metrics\n\n"
    report += f"- Accuracy: {metrics['accuracy']:.4f}\n"
    report += f"- Precision: {metrics['precision']:.4f}\n"
    report += f"- Recall: {metrics['recall']:.4f}\n"
    report += f"- F1 Score: {metrics['f1']:.4f}\n"
    report += f"- Invalid Predictions: {metrics['invalid_predictions']} / {metrics['total_examples']}\n\n"
    report += "## Per-Class Metrics\n\n"
    report += "| Emotion | Precision | Recall | F1 Score |\n"
    report += "|---------|-----------|--------|----------|\n"
    for emo, cmets in metrics.get('per_class_metrics', {}).items():
        report += f"| {emo} | {cmets['precision']:.4f} | {cmets['recall']:.4f} | {cmets['f1']:.4f} |\n"
    return report


def save_report(
    report: str,
    model_name: str,
    experiment_type: str,
    timestamp: str = None
) -> str:
    """
    Save report to file
    """
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

    output_dir = os.path.join(config.RESULTS_DIR, experiment_type)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"{model_name}_report_{timestamp}.md")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)

    return output_file
