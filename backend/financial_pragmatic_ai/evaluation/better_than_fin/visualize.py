"""Visualization helpers for better-than-FinBERT evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _heatmap(ax, matrix: np.ndarray, labels: List[str], title: str):
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(int(matrix[row, col])), ha="center", va="center")

    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def save_confusion_matrices(
    labels: List[str],
    finbert_cm: np.ndarray,
    our_cm: np.ndarray,
    output_path: Path,
):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _heatmap(axes[0], finbert_cm, labels, "FinBERT Baseline Confusion Matrix")
    _heatmap(axes[1], our_cm, labels, "Our System Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_performance_bars(
    finbert_metrics: Dict,
    our_metrics: Dict,
    output_path: Path,
):
    names = ["Accuracy", "Macro F1", "Weighted F1"]
    fin_values = [
        finbert_metrics["accuracy"],
        finbert_metrics["macro_f1"],
        finbert_metrics["weighted_f1"],
    ]
    our_values = [
        our_metrics["accuracy"],
        our_metrics["macro_f1"],
        our_metrics["weighted_f1"],
    ]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, fin_values, width, label="FinBERT")
    ax.bar(x + width / 2, our_values, width, label="Our System")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Metric Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_class_distribution(
    labels: List[str],
    y_true: List[str],
    y_finbert: List[str],
    y_ours: List[str],
    output_path: Path,
):
    counts_true = [sum(1 for y in y_true if y == label) for label in labels]
    counts_finbert = [sum(1 for y in y_finbert if y == label) for label in labels]
    counts_ours = [sum(1 for y in y_ours if y == label) for label in labels]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, counts_true, width=width, label="Ground Truth")
    ax.bar(x, counts_finbert, width=width, label="FinBERT")
    ax.bar(x + width, counts_ours, width=width, label="Our System")
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_agreement_pie(
    agreement_rate_value: float,
    output_path: Path,
):
    agreement_pct = max(0.0, min(1.0, agreement_rate_value))
    disagreement_pct = 1.0 - agreement_pct

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        [agreement_pct, disagreement_pct],
        labels=["Agreement", "Disagreement"],
        autopct="%1.1f%%",
        colors=["#4caf50", "#f44336"],
        startangle=90,
    )
    ax.set_title("FinBERT vs Our System Agreement")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
