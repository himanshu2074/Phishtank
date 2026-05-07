# 🛡️ SentryLink – Phishing Website Detection System

An AI-powered phishing website detection system focused on:

- Zero-day phishing detection
- Concept drift & temporal generalization
- Machine Learning & Ensemble techniques
- Real-time URL-based phishing prediction

The system analyzes URLs using handcrafted feature engineering and machine learning models to classify websites as **Phishing** or **Legitimate**.

---

# 🚀 Features

- ✅ Real-time phishing URL detection
- ✅ Zero-day phishing analysis
- ✅ Temporal split evaluation
- ✅ Concept drift simulation
- ✅ Ensemble learning methods
- ✅ Deep Learning (CNN)
- ✅ Streamlit web interface
- ✅ Accuracy vs latency evaluation
- ✅ ROC curve & confusion matrix visualization

---

# 🧠 Research Objectives

This project focuses on two major cybersecurity research problems:

## 1. Concept Drift & Temporal Generalization

Phishing techniques continuously evolve over time.  
The system evaluates how model performance changes on newer unseen phishing URLs using:

- Temporal train-test splitting
- Drift adaptation experiments
- Retraining simulations

---

## 2. Ensemble Design Trade-offs

The project compares multiple ML and DL approaches based on:

- Accuracy
- False Positive Rate
- Inference Latency
- Model Size
- Deployment Cost

---

# 🏗️ Project Structure

```text
SentryLink/
│
├── app.py                     # Streamlit web application
├── main.py                    # Full research pipeline
├── train_model.py             # Final model training script
├── config.py                  # Hyperparameters & paths
│
├── data_loader.py             # Dataset loading
├── feature_engineering.py     # URL feature extraction
├── feature_selection.py       # Feature selection methods
├── temporal.py                # Temporal split & drift simulation
│
├── ml_models.py               # ML model implementations
├── deep_learning.py           # CNN implementation
├── ensemble.py                # Ensemble architectures
├── evaluation.py              # Metrics & plotting
│
├── phishing_dataset_total.csv # Final dataset
├── phishing_model.pkl         # Saved trained model
│
├── results/                   # Generated graphs & outputs
├── models/                    # Saved models
└── README.md
