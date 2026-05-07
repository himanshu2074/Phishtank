import pandas as pd
import numpy as np

# -----------------------------
# 1. Load phishing dataset 
# -----------------------------
phish = pd.read_csv("verified_online.csv")

phish = phish[["url", "submission_time"]]

phish = phish.rename(columns={
    "submission_time": "timestamp"
})

phish["label"] = 1

print("Phishing URLs:", len(phish))


# -----------------------------
# 2. Load legitimate dataset (Tranco)
# -----------------------------
tranco = pd.read_csv("top-1m.csv", header=None)

tranco = tranco.iloc[:60000]  # take first 60k domains
tranco.columns = ["rank", "domain"]

# Realistic path templates
paths = [
    "",
    "/login",
    "/account",
    "/products",
    "/product/123",
    "/search?q=phone",
    "/watch?v=abc123",
    "/user/profile",
    "/support",
    "/help",
    "/docs/api",
    "/blog/article",
    "/settings",
    "/checkout",
    "/cart",
    "/about",
]

# Random path selection
random_paths = np.random.choice(paths, len(tranco))

# Optional realistic query parameters
queries = [
    "",
    "?id=123",
    "?page=2",
    "?ref=home",
    "?utm_source=google",
    "?session=abc123"
]

random_queries = np.random.choice(queries, len(tranco))

# Build legitimate URLs
tranco["url"] = "https://" + tranco["domain"] + random_paths + random_queries

tranco = tranco[["url"]]
tranco["label"] = 0


# Generate realistic timestamps for legitimate URLs
start = pd.Timestamp("2024-01-01")
end = pd.Timestamp("2025-01-01")

tranco["timestamp"] = pd.to_datetime(
    np.random.randint(start.value // 10**9, end.value // 10**9, len(tranco)),
    unit="s"
)

print("Legitimate URLs:", len(tranco))


# -----------------------------
# 3. Combine datasets
# -----------------------------
dataset = pd.concat([phish, tranco], ignore_index=True)

# Shuffle dataset
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

print("Total dataset size:", len(dataset))


# -----------------------------
# 4. Save dataset
# -----------------------------
dataset.to_csv("phishing_dataset_total.csv", index=False)

print("Dataset saved as phishing_dataset_total.csv")