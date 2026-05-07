"""
config.py – Central configuration for the Phishing Website Detection project.
"""

import os

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

DATASET_PATH = os.path.join(DATA_DIR, "phishing_dataset.csv")

# ── Reproducibility ─────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Dataset generation ───────────────────────────────────────────────────────
NUM_SAMPLES = 10_000          # total URLs to generate
PHISHING_RATIO = 0.45         # fraction that are phishing

# ── Feature selection ────────────────────────────────────────────────────────
TOP_K_FEATURES = 8            # number of features to keep after selection

# ── Model hyper-parameters ───────────────────────────────────────────────────
TEST_SIZE = 0.20
TEMPORAL_TEST_RATIO = 0.30    # newest 30 % used for temporal test set

RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 15

XGB_N_ESTIMATORS = 200
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.1

SVM_C = 1.0
SVM_KERNEL = "rbf"

LR_MAX_ITER = 1000

# ── CNN hyper-parameters ─────────────────────────────────────────────────────
CNN_MAX_FEATURES = 2000       # TF-IDF vocabulary size
CNN_EPOCHS = 5
CNN_BATCH_SIZE = 64

# ── Ensemble ─────────────────────────────────────────────────────────────────
CASCADE_CONFIDENCE_THRESHOLD = 0.70   # below this → escalate

# ── Drift adaptation ────────────────────────────────────────────────────────
RETRAIN_NEW_SAMPLE_FRACTION = 0.15    # fraction of new data used to retrain
