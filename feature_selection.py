"""
feature_selection.py – Select the most informative features using
Chi-square scores and Random Forest importances.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler

import config


def chi_square_selection(X: pd.DataFrame, y: pd.Series, k: int = config.TOP_K_FEATURES):
    """Return the top-k feature names ranked by Chi-square statistic."""
    # Chi-square requires non-negative values → min-max scale
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    selector = SelectKBest(chi2, k=k)
    selector.fit(X_scaled, y)
    mask = selector.get_support()
    selected = X.columns[mask].tolist()
    print(f"[feature_selection] Chi-square top-{k}: {selected}")
    return selected


def rf_importance_selection(X: pd.DataFrame, y: pd.Series, k: int = config.TOP_K_FEATURES):
    """Return top-k features ranked by Random Forest feature importance."""
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=config.RANDOM_SEED, n_jobs=-1
    )
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    selected = importances.nlargest(k).index.tolist()
    print(f"[feature_selection] RF importance top-{k}: {selected}")
    return selected, importances


def select_features(X: pd.DataFrame, y: pd.Series, k: int = config.TOP_K_FEATURES):
    """
    Run both methods and return the **union** of their top-k selections
    (guarantees at least k features are kept).
    """
    chi_feats = chi_square_selection(X, y, k)
    rf_feats, importances = rf_importance_selection(X, y, k)

    combined = list(dict.fromkeys(chi_feats + rf_feats))  # preserve order, no dups
    print(f"[feature_selection] Combined selected features ({len(combined)}): {combined}")
    return combined, importances
