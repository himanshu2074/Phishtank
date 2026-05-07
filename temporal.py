"""
temporal.py – Random and temporal train/test splits + drift adaptation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config


def random_split(X, y, test_size=config.TEST_SIZE):
    """Standard random train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=config.RANDOM_SEED, stratify=y,
    )
    print(f"[temporal] Random split → train={len(X_train)}, test={len(X_test)}")
    return X_train, X_test, y_train, y_test


def temporal_split(X, y, timestamps, test_ratio=config.TEMPORAL_TEST_RATIO):
    """
    Train on the older portion of data, test on the newest portion.
    This simulates zero-day phishing detection.
    """
    ts = pd.to_datetime(timestamps).values
    sorted_idx = np.argsort(ts)

    split_point = int(len(sorted_idx) * (1 - test_ratio))
    train_idx = sorted_idx[:split_point]
    test_idx = sorted_idx[split_point:]

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    print(f"[temporal] Temporal split → train={len(X_train)} (older), "
          f"test={len(X_test)} (newer)")
    return X_train, X_test, y_train, y_test


def drift_adaptation_split(X, y, timestamps,
                           retrain_frac=config.RETRAIN_NEW_SAMPLE_FRACTION):
    """
    Simulate drift adaptation:
      1. Initial train on old data.
      2. Test on new data (before retraining).
      3. Retrain with a portion of new data added.
      4. Test again on remaining new data.

    Returns initial and adapted splits.
    """
    ts = pd.to_datetime(timestamps).values
    sorted_idx = np.argsort(ts)

    n = len(sorted_idx)
    initial_end = int(n * 0.60)
    retrain_end = initial_end + int(n * retrain_frac)

    initial_train_idx = sorted_idx[:initial_end]
    new_data_idx = sorted_idx[initial_end:]

    # Before retraining → test on all new data
    X_old_train = X.iloc[initial_train_idx].reset_index(drop=True)
    y_old_train = y.iloc[initial_train_idx].reset_index(drop=True)

    X_new_test_all = X.iloc[new_data_idx].reset_index(drop=True)
    y_new_test_all = y.iloc[new_data_idx].reset_index(drop=True)

    # After retraining → add a chunk of new data to training
    retrain_idx = sorted_idx[:retrain_end]
    remaining_idx = sorted_idx[retrain_end:]

    X_retrain = X.iloc[retrain_idx].reset_index(drop=True)
    y_retrain = y.iloc[retrain_idx].reset_index(drop=True)

    X_remaining_test = X.iloc[remaining_idx].reset_index(drop=True)
    y_remaining_test = y.iloc[remaining_idx].reset_index(drop=True)

    print(f"[temporal] Drift adapt → initial_train={len(X_old_train)}, "
          f"new_test={len(X_new_test_all)}, retrain={len(X_retrain)}, "
          f"remaining_test={len(X_remaining_test)}")

    return {
        "initial": (X_old_train, X_new_test_all, y_old_train, y_new_test_all),
        "adapted": (X_retrain, X_remaining_test, y_retrain, y_remaining_test),
    }
