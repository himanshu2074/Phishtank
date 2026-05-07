import streamlit as st
import pandas as pd
import pickle

from feature_engineering import extract_features

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(page_title="SentryLink", layout="wide")

# ── Cyber Theme CSS ─────────────────────────────────────────
st.markdown("""
<style>

/* Background */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 20% 20%, #0f172a, #020617);
    color: white;
}

/* Animated Grid Effect */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    width: 100%;
    height: 100%;
    background-image:
        linear-gradient(rgba(0,255,200,0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,200,0.05) 1px, transparent 1px);
    background-size: 40px 40px;
    animation: moveGrid 20s linear infinite;
    z-index: 0;
}

@keyframes moveGrid {
    from {transform: translate(0,0);}
    to {transform: translate(40px,40px);}
}

/* Headline */
.hero {
    font-size: 40px;
    font-weight: 800;
    color: #00ffc8;
    margin-bottom: 5px;
}

/* Subtitle */
.subtext {
    color: #94a3b8;
    margin-bottom: 20px;
}

/* Input */
.stTextInput input {
    background-color: #020617 !important;
    color: #00ffc8 !important;
    border: 1px solid #00ffc8;
    border-radius: 10px;
}

/* Button */
.stButton>button {
    background: #00ffc8;
    color: black;
    border-radius: 10px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton>button:hover {
    background: #00ccaa;
    transform: scale(1.05);
}

/* Card */
.card {
    background: rgba(0,255,200,0.05);
    padding: 20px;
    border-radius: 12px;
    border: 1px solid rgba(0,255,200,0.2);
    box-shadow: 0 0 20px rgba(0,255,200,0.1);
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)

# ── Compact Header + Scanner ───────────────────────────────
st.markdown('<div class="hero">🛡️ SentryLink</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">AI-powered phishing detection system</div>', unsafe_allow_html=True)

# ── Load Model ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("phishing_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ── Scanner Card ───────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown("### 🔍 Scan a Website")

url = st.text_input("Enter URL")

if st.button("🚨 Scan Now"):

    if url.strip() == "":
        st.warning("Please enter a URL")
    else:
        user_df = pd.DataFrame({"url": [url]})
        user_features = extract_features(user_df)

        prediction = model.predict(user_features)[0]
        probability = model.predict_proba(user_features)[0][1]

        st.markdown("### 🔐 Result")

        st.progress(float(probability))

        if prediction == 1:
            st.error("⚠️ Phishing Website Detected")
            st.markdown(f"**Threat Level:** `{probability:.2f}`")
        else:
            st.success("✅ Safe Website")
            st.markdown(f"**Threat Level:** `{probability:.2f}`")

st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────
st.markdown("""
<div class="footer">
Developed by<br><br>
Himanshu Sekhar Mishra | Risabh Kumar | Sudhanshu Sekhar | Tanmay Pandey
<br><br>
© 2026 SentryLink
</div>
""", unsafe_allow_html=True)

