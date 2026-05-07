"""
ensemble.py – Voting, Cascade, and Distillation ensemble methods.
"""

import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import config


# ── 1. Voting Ensemble ──────────────────────────────────────────────────────

class VotingEnsemble:
    """Soft-voting ensemble of pre-trained classifiers."""

    def __init__(self, models: dict):
        """models: dict {name: fitted_model} — each must support predict_proba."""
        self.models = models

    def predict_proba(self, X):
        probas = np.array([m.predict_proba(X)[:, 1] for m in self.models.values()])
        return probas.mean(axis=0)

    def predict(self, X):
        avg = self.predict_proba(X)
        return (avg >= 0.5).astype(int)


# ── 2. Cascade Ensemble ────────────────────────────────────────────────────

class CascadeEnsemble:
    """
    Two-stage cascade:
      Stage 1 – lightweight model (e.g. LR).  If confidence ≥ threshold → done.
      Stage 2 – stronger model (e.g. XGBoost) for uncertain samples.
    """

    def __init__(self, light_model, strong_model,
                 threshold: float = config.CASCADE_CONFIDENCE_THRESHOLD):
        self.light = light_model
        self.strong = strong_model
        self.threshold = threshold

    def predict(self, X):
        probs_light = self.light.predict_proba(X)[:, 1]
        preds = np.empty(len(X), dtype=int)
        confident = np.abs(probs_light - 0.5) >= (self.threshold - 0.5)

        preds[confident] = (probs_light[confident] >= 0.5).astype(int)

        uncertain_idx = ~confident
        if uncertain_idx.any():
            X_unc = X[uncertain_idx] if hasattr(X, "iloc") is False else X.iloc[uncertain_idx]
            # Handle both numpy arrays and DataFrames
            if hasattr(X, "iloc"):
                X_unc = X.iloc[np.where(uncertain_idx)[0]]
            else:
                X_unc = X[uncertain_idx]
            preds[uncertain_idx] = self.strong.predict(X_unc)

        return preds, confident.sum(), uncertain_idx.sum()

    def predict_proba(self, X):
        """Return stage-1 probabilities (for ROC-AUC compatibility)."""
        return self.light.predict_proba(X)[:, 1]


# ── 3. Model Distillation ──────────────────────────────────────────────────

class DistilledEnsemble:
    """Train a small MLP student to mimic a teacher ensemble's soft labels."""

    def __init__(self):
        self.student = MLPClassifier(
            hidden_layer_sizes=(32,),
            max_iter=300,
            random_state=config.RANDOM_SEED,
        )

    def distill(self, teacher_ensemble, X_train):
        """Generate soft labels from teacher and train student."""
        soft_labels = teacher_ensemble.predict_proba(X_train)

        # Convert to hard labels for MLP (sklearn doesn't support soft targets directly)
        # We use the teacher's predictions as ground-truth labels
        hard_labels = (soft_labels >= 0.5).astype(int)

        print("[ensemble] Distilling teacher → student MLP …")
        start = time.time()
        self.student.fit(X_train, hard_labels)
        elapsed = time.time() - start
        print(f"[ensemble] Student trained in {elapsed:.2f}s")

    def predict(self, X):
        return self.student.predict(X)

    def predict_proba(self, X):
        return self.student.predict_proba(X)[:, 1]
