"""
deep_learning.py – Lightweight 1D-CNN for phishing URL classification.

Uses TF-IDF character-level n-grams as input, reshaped for Conv1D.
"""

import os
import time
import warnings

# Suppress TF informational logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import config

# Lazy-import TensorFlow to keep startup fast
_tf = None
_keras = None


def _import_tf():
    global _tf, _keras
    if _tf is None:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        _tf = tf
        _keras = tf.keras
    return _tf, _keras


def build_tfidf_vectorizer(train_urls, max_features=config.CNN_MAX_FEATURES):
    """Fit a character-level TF-IDF vectorizer on training URLs."""
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=max_features,
    )
    vec.fit(train_urls)
    return vec


def _build_cnn(input_dim: int):
    """Build a small 1D-CNN model."""
    _, keras = _import_tf()

    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim, 1)),
        keras.layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Conv1D(32, kernel_size=3, activation="relu", padding="same"),
        keras.layers.GlobalMaxPooling1D(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_cnn(X_train_tfidf, y_train,
              epochs=config.CNN_EPOCHS,
              batch_size=config.CNN_BATCH_SIZE):
    """Train the CNN on TF-IDF vectors (as numpy arrays)."""
    _import_tf()

    # Reshape for Conv1D: (samples, features, 1)
    X = np.asarray(X_train_tfidf, dtype="float32")
    if X.ndim == 2:
        X = X[..., np.newaxis]
    y = np.asarray(y_train, dtype="float32")

    model = _build_cnn(X.shape[1])

    print(f"[deep_learning] Training CNN  ({epochs} epochs, "
          f"batch_size={batch_size}) ...")
    start = time.time()
    model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=0,
    )
    elapsed = time.time() - start
    print(f"[deep_learning] CNN trained in {elapsed:.1f}s")
    return model


def predict_cnn(model, X_tfidf):
    """Return predicted probabilities and binary labels."""
    X = np.asarray(X_tfidf, dtype="float32")
    if X.ndim == 2:
        X = X[..., np.newaxis]
    probs = model.predict(X, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)
    return preds, probs
