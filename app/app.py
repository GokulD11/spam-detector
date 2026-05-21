import streamlit as st
import joblib
import os
from preprocessing import preprocess_text  # our shared preprocessing module

# ─────────────────────────────────────────────
# PAGE CONFIG — must be the first Streamlit command
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Spam Detector",
    page_icon="📩",
    layout="centered"
)

# ─────────────────────────────────────────────
# LOAD MODEL (cached so it loads only ONCE, not every interaction)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'spam_classifier.joblib')
    return joblib.load(model_path)

model = load_model()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("📩 SMS Spam Detector")
st.write(
    "Enter a message below and the model will classify it as **Spam** or **Ham** "
    "(legitimate). Built with TF-IDF + Linear SVM."
)

# ─────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────
user_input = st.text_area(
    "Your message:",
    height=120,
    placeholder="e.g. Congratulations! You've won a free prize. Click now!"
)

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
if st.button("Check Message", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a message first.")
    else:
        # SAME preprocessing as training (avoids train/serve skew!)
        cleaned = preprocess_text(user_input)

        # Predict
        prediction = model.predict([cleaned])[0]

        # Show result
        if prediction == 1:
            st.error("🚨 This looks like **SPAM**")
        else:
            st.success("✅ This looks like **HAM** (legitimate)")

        # Show what the model actually saw (great for transparency)
        with st.expander("See how your message was processed"):
            st.write("**Original:**", user_input)
            st.write("**After preprocessing:**", cleaned if cleaned else "(empty after cleaning)")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.caption("Trained on the UCI SMS Spam Collection dataset · scikit-learn · Streamlit")