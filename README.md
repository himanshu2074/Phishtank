🛡️ SentryLink – Phishing Website Detection System

An AI-powered phishing website detection system focused on:

Zero-day phishing detection
Concept drift & temporal generalization
Machine Learning & Ensemble techniques
Real-time URL-based phishing prediction

The system analyzes URLs using handcrafted feature engineering and machine learning models to classify websites as Phishing or Legitimate.

🚀 Features
✅ Real-time phishing URL detection
✅ Zero-day phishing analysis
✅ Temporal split evaluation
✅ Concept drift simulation
✅ Ensemble learning methods
✅ Deep Learning (CNN)
✅ Streamlit web interface
✅ Accuracy vs latency evaluation
✅ ROC curve & confusion matrix visualization
🧠 Research Objectives

This project focuses on two major cybersecurity research problems:

1. Concept Drift & Temporal Generalization

Phishing techniques continuously evolve over time.
The system evaluates how model performance changes on newer unseen phishing URLs using:

Temporal train-test splitting
Drift adaptation experiments
Retraining simulations
2. Ensemble Design Trade-offs

The project compares multiple ML and DL approaches based on:

Accuracy
False Positive Rate
Inference Latency
Model Size
Deployment Cost
🏗️ Project Structure
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
📊 Dataset

The project uses the PhiUSIIL Phishing URL Dataset combined with timestamp generation for temporal analysis.

Dataset Characteristics
Real phishing & legitimate URLs
Timestamp-based evaluation
Balanced classes
Supports concept drift experiments
🔍 Extracted URL Features

The system extracts 13 handcrafted URL-based features, including:

URL length
Number of dots
Number of subdomains
Presence of IP address
HTTPS usage
Suspicious keywords
Digit ratio
Entropy
Query parameter count
Special character ratio

These lightweight features enable:

Real-time prediction
Fast inference
Zero-day phishing detection
🤖 Implemented Models
Machine Learning Models
Logistic Regression
Random Forest
XGBoost
Support Vector Machine (SVM)
Deep Learning
CNN on TF-IDF URL vectors
Ensemble Models
Voting Ensemble
Cascade Ensemble
Distilled Ensemble
📈 Evaluation Metrics

The project evaluates:

Accuracy
Precision
Recall
F1-Score
ROC-AUC
Inference Latency
Model Size
False Positive Rate
⏳ Temporal & Drift Evaluation

The system simulates concept drift using timestamp-based dataset splitting.

Temporal Evaluation
Train on older URLs
Test on newer URLs
Drift Adaptation
Evaluate performance before retraining
Retrain using newer data
Compare performance recovery
🌐 Streamlit Web App

The Streamlit application provides:

Real-time URL scanning
Threat probability visualization
Safe / Phishing classification
⚙️ Installation
1. Clone Repository
git clone <repository-url>
cd SentryLink
2. Create Virtual Environment
Windows
python -m venv .venv
.venv\Scripts\activate
Linux / Mac
python3 -m venv .venv
source .venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
▶️ Running the Project
Train Model
python train_model.py
Run Full Research Pipeline
python main.py
Launch Web App
streamlit run app.py
📊 Generated Outputs

The project automatically generates:

ROC Curves
Confusion Matrices
Feature Importance Graphs
Accuracy vs Latency Plots
Temporal Performance Graphs
Drift Adaptation Graphs

Saved inside:

results/
🔬 Future Improvements
Real-time drift detection using KS-Test / PSI
Automatic retraining pipeline
HTML & JavaScript content analysis
Domain reputation integration
Online learning models
Live phishing feed integration
🧠 Technologies Used
Python
Scikit-learn
XGBoost
TensorFlow / Keras
Streamlit
Pandas
NumPy
Matplotlib
Seaborn
👨‍💻 Authors
Himanshu Sekhar Mishra
📜 License

This project is developed for academic and research purposes.
