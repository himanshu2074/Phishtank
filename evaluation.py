"""
evaluation.py – Compute metrics, measure latency, and generate all plots.
"""

import os
import time
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)

import config

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Return a dict of classification metrics."""
    metrics = {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob)
    return metrics


def measure_latency(model, X_sample, n_repeats=5, predict_fn=None):
    """Average prediction latency in ms per sample."""
    if predict_fn is None:
        predict_fn = model.predict
    times = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        predict_fn(X_sample)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed / len(X_sample))
    return float(np.mean(times))


def measure_model_size(model):
    """Approximate model size in KB using pickle serialization."""
    try:
        data = pickle.dumps(model)
        return len(data) / 1024
    except Exception:
        return float("nan")


def evaluate_model(name, model, X_test, y_test, predict_fn=None, proba_fn=None):
    """Full evaluation: metrics + latency + size."""
    if predict_fn is None:
        y_pred = model.predict(X_test)
    else:
        y_pred = predict_fn(X_test)

    y_prob = None
    if proba_fn is not None:
        y_prob = proba_fn(X_test)
    elif hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob)
    metrics["Latency_ms"] = measure_latency(model, X_test, predict_fn=predict_fn or model.predict)
    metrics["Size_KB"] = measure_model_size(model)

    return metrics, y_pred, y_prob


# ── Plotting helpers ─────────────────────────────────────────────────────────

def _save(fig, name):
    """Save plot safely and overwrite old figures."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, name)

    # Delete old file if it exists
    if os.path.exists(path):
        os.remove(path)

    # Clear matplotlib state
    plt.clf()
    plt.cla()

    fig.savefig(path, dpi=150, bbox_inches="tight")

    # Fully close all figures
    plt.close(fig)
    plt.close("all")

    print(f"[evaluation] Saved {path}")


def plot_accuracy_comparison(results: dict):
    names = list(results.keys())
    accs = [results[n]["Accuracy"] for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("viridis", len(names))
    bars = ax.bar(names, accs, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Comparison")
    ax.set_ylim(0, 1.05)

    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                f"{v:.3f}", ha="center", fontsize=9)

    plt.xticks(rotation=30, ha="right")
    _save(fig, "accuracy_comparison.png")


def plot_roc_curves(roc_data: dict):
    fig, ax = plt.subplots(figsize=(8, 7))

    for name, (y_true, y_prob) in roc_data.items():
        if y_prob is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", linewidth=1.5)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right", fontsize=8)

    _save(fig, "roc_curves.png")


def plot_confusion_matrices(cm_data: dict):
    n = len(cm_data)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for ax, (name, (y_true, y_pred)) in zip(axes, cm_data.items()):
        cm = confusion_matrix(y_true, y_pred)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Legit", "Phish"],
            yticklabels=["Legit", "Phish"],
        )

        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    for i in range(len(cm_data), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Confusion Matrices", fontsize=14, y=1.02)
    fig.tight_layout()

    _save(fig, "confusion_matrices.png")


def plot_feature_importance(importances: pd.Series):
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    importances.plot.barh(
        ax=ax,
        color=sns.color_palette("viridis", len(importances)),
        edgecolor="black",
        linewidth=0.3,
    )

    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Importance")

    fig.tight_layout()
    _save(fig, "feature_importance.png")


def plot_temporal_comparison(random_results: dict, temporal_results: dict):

    models = sorted(set(random_results) & set(temporal_results))
    metrics_list = ["Accuracy", "Precision", "Recall", "F1"]

    x = np.arange(len(models))
    width = 0.35

    fig, axes = plt.subplots(1, len(metrics_list), figsize=(5 * len(metrics_list), 5))

    for ax, metric in zip(axes, metrics_list):

        r_vals = [random_results[m].get(metric, 0) for m in models]
        t_vals = [temporal_results[m].get(metric, 0) for m in models]

        ax.bar(x - width / 2, r_vals, width, label="Random Split", color="#5b9bd5")
        ax.bar(x + width / 2, t_vals, width, label="Temporal Split", color="#ed7d31")

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(metric)
        ax.legend(fontsize=7)

    fig.suptitle("Random Split vs Temporal Split", fontsize=14)

    fig.tight_layout()
    _save(fig, "temporal_comparison.png")


def plot_accuracy_vs_latency(results: dict):

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, m in results.items():

        ax.scatter(m["Latency_ms"], m["Accuracy"], s=100, zorder=5)

        ax.annotate(
            name,
            (m["Latency_ms"], m["Accuracy"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )

    ax.set_xlabel("Latency (ms / sample)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Prediction Latency")

    fig.tight_layout()
    _save(fig, "accuracy_vs_latency.png")


def plot_drift_comparison(before: dict, after: dict):

    metrics_list = ["Accuracy", "F1", "ROC-AUC"]
    names = sorted(set(before) & set(after))

    fig, axes = plt.subplots(1, len(metrics_list), figsize=(5 * len(metrics_list), 5))

    x = np.arange(len(names))
    w = 0.35

    for ax, metric in zip(axes, metrics_list):

        b = [before[n].get(metric, 0) for n in names]
        a = [after[n].get(metric, 0) for n in names]

        ax.bar(x - w / 2, b, w, label="Before Retraining", color="#c0504d")
        ax.bar(x + w / 2, a, w, label="After Retraining", color="#4bacc6")

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(metric)
        ax.legend(fontsize=7)

    fig.suptitle("Drift Adaptation: Before vs After Retraining", fontsize=14)

    fig.tight_layout()
    _save(fig, "drift_adaptation.png")


def print_results_table(results: dict, title: str = "Evaluation Results"):

    df = pd.DataFrame(results).T
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(4)

    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(df.to_string())
    print()
