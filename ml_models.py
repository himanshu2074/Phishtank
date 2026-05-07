"""
ml_models.py – Train and return classical ML classifiers for phishing detection.
"""

import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

import config


def _train_and_time(name, model, X_train, y_train):
    """Fit model and print training time."""
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"[ml_models] {name} trained in {elapsed:.2f}s")
    return model


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        max_iter=config.LR_MAX_ITER,
        random_state=config.RANDOM_SEED,
        solver="lbfgs",
    )
    return _train_and_time("LogisticRegression", model, X_train, y_train)


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=config.RF_MAX_DEPTH,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )
    return _train_and_time("RandomForest", model, X_train, y_train)


def train_xgboost(X_train, y_train):
    model = XGBClassifier(
        n_estimators=config.XGB_N_ESTIMATORS,
        max_depth=config.XGB_MAX_DEPTH,
        learning_rate=config.XGB_LEARNING_RATE,
        random_state=config.RANDOM_SEED,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
    )
    return _train_and_time("XGBoost", model, X_train, y_train)


def train_svm(X_train, y_train):
    model = SVC(
        C=config.SVM_C,
        kernel=config.SVM_KERNEL,
        probability=True,
        random_state=config.RANDOM_SEED,
    )
    return _train_and_time("SVM", model, X_train, y_train)


def train_all_models(X_train, y_train) -> dict:
    """Train all four classifiers and return them as a dict."""
    models = {
        "LogisticRegression": train_logistic_regression(X_train, y_train),
        "RandomForest":       train_random_forest(X_train, y_train),
        "XGBoost":            train_xgboost(X_train, y_train),
        "SVM":                train_svm(X_train, y_train),
    }
    return models
