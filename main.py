"""
main.py – Phishing Website Detection: full pipeline orchestrator.

Usage:
    python main.py

Runs data generation → feature engineering → feature selection →
ML training → DL training → ensembles → temporal evaluation →
drift adaptation → evaluation & plots.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd

import config
from data_loader import generate_dataset, load_dataset
from feature_engineering import extract_features
from feature_selection import select_features
from ml_models import train_all_models, train_logistic_regression, train_random_forest, train_xgboost
from deep_learning import build_tfidf_vectorizer, train_cnn, predict_cnn
from ensemble import VotingEnsemble, CascadeEnsemble, DistilledEnsemble
from temporal import random_split, temporal_split, drift_adaptation_split
from evaluation import (
    evaluate_model, compute_metrics, measure_latency, measure_model_size,
    plot_accuracy_comparison, plot_roc_curves, plot_confusion_matrices,
    plot_feature_importance, plot_temporal_comparison,
    plot_accuracy_vs_latency, plot_drift_comparison, print_results_table,
)

# Ensure output directories exist
os.makedirs(config.DATA_DIR, exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)
os.makedirs(config.MODELS_DIR, exist_ok=True)


def main():
    print("=" * 70)
    print("  PHISHING WEBSITE DETECTION — ML / DL / Ensemble Pipeline")
    print("=" * 70)

    # ── 1. Data ──────────────────────────────────────────────────────────
    print("\n> Step 1: loading dataset …")
    df = load_dataset("phishing_dataset_total.csv")

    # ── 2. Feature engineering ───────────────────────────────────────────
    print("\n> Step 2: Extracting URL features …")
    features = extract_features(df)
    labels = df["label"]
    timestamps = df["timestamp"]

    # ── 3. Feature selection ─────────────────────────────────────────────
    print("\n> Step 3: Selecting top features …")
    selected_feats, rf_importances = select_features(features, labels)
    X = features[selected_feats]

    plot_feature_importance(rf_importances)

    # ── 4. Random split & ML model training ──────────────────────────────
    print("\n> Step 4: Random split → Training ML models …")
    X_train, X_test, y_train, y_test = random_split(X, labels)

    print("\nFULL LABEL DISTRIBUTION")
    print(labels.value_counts())

    print("\nTRAIN LABEL DISTRIBUTION")
    print(y_train.value_counts())

    print("\nTEST LABEL DISTRIBUTION")
    print(y_test.value_counts())

    ml_models = train_all_models(X_train, y_train)

    # Evaluate ML models (random split)
    random_results = {}
    roc_data = {}
    cm_data = {}

    for name, model in ml_models.items():
        metrics, y_pred, y_prob = evaluate_model(name, model, X_test, y_test)
        random_results[name] = metrics
        roc_data[name] = (y_test, y_prob)
        cm_data[name] = (y_test, y_pred)

    print_results_table(random_results, "ML Models — Random Split")

    # ── 5. Deep learning (CNN) ───────────────────────────────────────────
    print("\n> Step 5: Training CNN on TF-IDF vectors …")
    train_urls = df.iloc[X_train.index]["url"] if hasattr(X_train, "index") else df["url"].iloc[:len(X_train)]
    test_urls = df.iloc[X_test.index]["url"] if hasattr(X_test, "index") else df["url"].iloc[:len(X_test)]

    # Reconstruct URL indices from splits
    all_urls = df["url"]
    tfidf_vec = build_tfidf_vectorizer(all_urls)

    X_train_tfidf = tfidf_vec.transform(df["url"].iloc[X_train.index]).toarray()
    X_test_tfidf = tfidf_vec.transform(df["url"].iloc[X_test.index]).toarray()

    cnn_model = train_cnn(X_train_tfidf, y_train.values)
    cnn_preds, cnn_probs = predict_cnn(cnn_model, X_test_tfidf)

    cnn_metrics = compute_metrics(y_test.values, cnn_preds, cnn_probs)
    cnn_metrics["Latency_ms"] = measure_latency(
        cnn_model, X_test_tfidf[..., np.newaxis].astype("float32"),
        predict_fn=lambda x: cnn_model.predict(x, verbose=0),
    )
    cnn_metrics["Size_KB"] = float("nan")  # Keras models not easily pickle-able
    random_results["CNN"] = cnn_metrics
    roc_data["CNN"] = (y_test.values, cnn_probs)
    cm_data["CNN"] = (y_test.values, cnn_preds)

    print_results_table({"CNN": cnn_metrics}, "CNN — Random Split")

    # ── 6. Ensemble methods ──────────────────────────────────────────────
    print("\n> Step 6: Building ensemble models …")

    # 6a. Voting ensemble (LR + RF + XGB)
    voting = VotingEnsemble({
        k: ml_models[k] for k in ["LogisticRegression", "RandomForest", "XGBoost"]
    })
    voting_preds = voting.predict(X_test)
    voting_probs = voting.predict_proba(X_test)
    voting_metrics = compute_metrics(y_test, voting_preds, voting_probs)
    voting_metrics["Latency_ms"] = measure_latency(voting, X_test)
    voting_metrics["Size_KB"] = sum(
        measure_model_size(ml_models[k])
        for k in ["LogisticRegression", "RandomForest", "XGBoost"]
    )
    random_results["VotingEnsemble"] = voting_metrics
    roc_data["VotingEnsemble"] = (y_test, voting_probs)
    cm_data["VotingEnsemble"] = (y_test, voting_preds)

    # 6b. Cascade ensemble (LR → XGBoost)
    cascade = CascadeEnsemble(ml_models["LogisticRegression"], ml_models["XGBoost"])
    cascade_preds, n_confident, n_escalated = cascade.predict(X_test)
    cascade_probs = cascade.predict_proba(X_test)
    cascade_metrics = compute_metrics(y_test, cascade_preds, cascade_probs)
    cascade_metrics["Latency_ms"] = measure_latency(
        cascade, X_test, predict_fn=lambda x: cascade.predict(x)[0]
    )
    cascade_metrics["Size_KB"] = (
        measure_model_size(ml_models["LogisticRegression"]) +
        measure_model_size(ml_models["XGBoost"])
    )
    random_results["CascadeEnsemble"] = cascade_metrics
    roc_data["CascadeEnsemble"] = (y_test, cascade_probs)
    cm_data["CascadeEnsemble"] = (y_test, cascade_preds)
    print(f"  Cascade: {n_confident} confident, {n_escalated} escalated to XGBoost")

    # 6c. Distilled ensemble
    distilled = DistilledEnsemble()
    distilled.distill(voting, X_train)
    dist_preds = distilled.predict(X_test)
    dist_probs = distilled.predict_proba(X_test)
    dist_metrics = compute_metrics(y_test, dist_preds, dist_probs)
    dist_metrics["Latency_ms"] = measure_latency(distilled.student, X_test)
    dist_metrics["Size_KB"] = measure_model_size(distilled.student)
    random_results["DistilledEnsemble"] = dist_metrics
    roc_data["DistilledEnsemble"] = (y_test, dist_probs)
    cm_data["DistilledEnsemble"] = (y_test, dist_preds)

    print_results_table(random_results, "All Models — Random Split")

    # ── 7. Temporal split evaluation ─────────────────────────────────────
    print("\n> Step 7: Temporal split evaluation …")
    Xt_train, Xt_test, yt_train, yt_test = temporal_split(X, labels, timestamps)

    temporal_results = {}
    temporal_models_to_eval = ["LogisticRegression", "RandomForest", "XGBoost"]

    for name in temporal_models_to_eval:
        # Retrain on temporal training set
        from ml_models import (
            train_logistic_regression, train_random_forest, train_xgboost,
        )
        train_fn = {
            "LogisticRegression": train_logistic_regression,
            "RandomForest": train_random_forest,
            "XGBoost": train_xgboost,
        }[name]
        temp_model = train_fn(Xt_train, yt_train)
        metrics, _, _ = evaluate_model(name, temp_model, Xt_test, yt_test)
        temporal_results[name] = metrics

    print_results_table(temporal_results, "ML Models — Temporal Split")

    # ── 8. Drift adaptation ──────────────────────────────────────────────
    print("\n> Step 8: Drift adaptation experiment …")
    drift_splits = drift_adaptation_split(X, labels, timestamps)

    X_old, X_new_all, y_old, y_new_all = drift_splits["initial"]
    X_retrained, X_remain, y_retrained, y_remain = drift_splits["adapted"]

    drift_before = {}
    drift_after = {}

    for name in ["LogisticRegression", "RandomForest", "XGBoost"]:
        train_fn = {
            "LogisticRegression": train_logistic_regression,
            "RandomForest": train_random_forest,
            "XGBoost": train_xgboost,
        }[name]

        # Before: train on old, test on all new
        model_old = train_fn(X_old, y_old)
        m_before, _, _ = evaluate_model(name, model_old, X_new_all, y_new_all)
        drift_before[name] = m_before

        # After: retrain with additional new data, test on remaining
        model_new = train_fn(X_retrained, y_retrained)
        m_after, _, _ = evaluate_model(name, model_new, X_remain, y_remain)
        drift_after[name] = m_after

    print_results_table(drift_before, "Drift — BEFORE Retraining")
    print_results_table(drift_after, "Drift — AFTER Retraining")

    # ── 9. Generate all plots ────────────────────────────────────────────
    print("\n> Step 9: Generating plots …")
    plot_accuracy_comparison(random_results)
    plot_roc_curves(roc_data)
    plot_confusion_matrices(cm_data)
    plot_temporal_comparison(random_results, temporal_results)
    plot_accuracy_vs_latency(random_results)
    plot_drift_comparison(drift_before, drift_after)

    # ── Done ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ✓ Pipeline complete!  All results and plots saved to:")
    print(f"    {config.RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
