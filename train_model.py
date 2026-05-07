import pickle
from data_loader import load_dataset
from feature_engineering import extract_features
from ml_models import train_random_forest

print("Loading dataset...")
df = load_dataset("phishing_dataset_total.csv")

print("Extracting features...")
features = extract_features(df)

X = features
y = df["label"]

print("Training model...")
model = train_random_forest(X, y)

print("Saving model...")
with open("phishing_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")