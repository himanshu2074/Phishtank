"""
feature_engineering.py – Extract URL-based features for phishing detection.
"""

import re
import numpy as np
import pandas as pd


# Suspicious keywords commonly found in phishing URLs
_SUSPICIOUS_KEYWORDS = [
    "login", "secure", "account", "verify", "update", "bank", "confirm",
    "password", "signin", "billing", "suspend", "alert", "wallet",
    "recover", "unlock", "validate", "authenticate", "notification",
]


def _has_ip_address(url: str) -> int:
    """Check if the URL contains an IP address instead of a domain name."""
    pattern = r"https?://(\d{1,3}\.){3}\d{1,3}"
    return int(bool(re.search(pattern, url)))


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract numerical features from the 'url' column.

    Returns a DataFrame with one row per URL and the following columns:
        url_length, num_dots, num_subdomains, has_ip, has_at, has_hyphen,
        is_https, num_suspicious_keywords, path_length, digit_ratio,
        special_char_ratio, num_query_params, url_entropy
    """
    urls = df["url"].astype(str)

    features = pd.DataFrame()

    # 1. URL length
    features["url_length"] = urls.str.len()

    # 2. Number of dots
    features["num_dots"] = urls.str.count(r"\.")

    # 3. Number of subdomains (dots in the netloc minus 1, approximated)
    def _count_subdomains(u):
        try:
            netloc = u.split("//")[-1].split("/")[0].split("@")[-1]
            parts = netloc.split(".")
            return max(0, len(parts) - 2)
        except Exception:
            return 0

    features["num_subdomains"] = urls.apply(_count_subdomains)

    # 4. Has IP address
    features["has_ip"] = urls.apply(_has_ip_address)

    # 5. Has '@' symbol
    features["has_at"] = urls.str.contains("@", regex=False).astype(int)

    # 6. Has '-' (hyphen) in domain
    features["has_hyphen"] = urls.str.contains("-", regex=False).astype(int)

    # 7. Uses HTTPS
    features["is_https"] = urls.str.startswith("https").astype(int)

    # 8. Count of suspicious keywords
    def _count_keywords(u):
        u_lower = u.lower()
        return sum(1 for kw in _SUSPICIOUS_KEYWORDS if kw in u_lower)

    features["num_suspicious_keywords"] = urls.apply(_count_keywords)

    # 9. Path length (everything after the domain)
    def _path_length(u):
        try:
            after_domain = u.split("//")[-1]
            idx = after_domain.find("/")
            return len(after_domain[idx:]) if idx != -1 else 0
        except Exception:
            return 0

    features["path_length"] = urls.apply(_path_length)

    # 10. Digit ratio
    def _digit_ratio(u):
        digits = sum(c.isdigit() for c in u)
        return digits / max(len(u), 1)

    features["digit_ratio"] = urls.apply(_digit_ratio)

    # 11. Special character ratio
    def _special_ratio(u):
        specials = sum(not c.isalnum() and c not in (":", "/", ".") for c in u)
        return specials / max(len(u), 1)

    features["special_char_ratio"] = urls.apply(_special_ratio)

    # 12. Number of query parameters
    def _query_params(u):
        if "?" not in u:
            return 0
        query = u.split("?", 1)[1]
        return query.count("&") + 1

    features["num_query_params"] = urls.apply(_query_params)

    # 13. Shannon entropy of the URL string
    def _entropy(u):
        prob = [u.count(c) / len(u) for c in set(u)]
        return -sum(p * np.log2(p) for p in prob if p > 0)

    features["url_entropy"] = urls.apply(_entropy)

    print(f"[feature_engineering] Extracted {features.shape[1]} features "
          f"from {features.shape[0]} URLs")
    return features


FEATURE_NAMES = [
    "url_length", "num_dots", "num_subdomains", "has_ip", "has_at",
    "has_hyphen", "is_https", "num_suspicious_keywords", "path_length",
    "digit_ratio", "special_char_ratio", "num_query_params", "url_entropy",
]
