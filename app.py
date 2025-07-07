import streamlit as st
import pandas as pd
import pickle
import numpy as np
import re
import string

# ----------------------------
# 🎯 Preprocessing Function
# ----------------------------
def preprocess_email(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove URLs
    text = re.sub(r'\W', ' ', text)                      # remove special chars
    text = re.sub(r'\s+', ' ', text)                     # remove extra spaces
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

# ----------------------------
# 🚨 Priority Logic (for ham only)
# ----------------------------
def assign_priority(text):
    text = text.lower()
    if any(word in text for word in ["urgent", "immediately", "asap", "account", "payment"]):
        return "High"
    elif any(word in text for word in ["reminder", "invoice", "meeting", "please check"]):
        return "Medium"
    else:
        return "Low"

# ----------------------------
# 🔐 Load Model & Vectorizer
# ----------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# ----------------------------
# 🎯 App Title
# ----------------------------
st.title("📧 Smart Email Classifier")
st.markdown("An AI tool that detects **Spam/Ham** emails and assigns **Priority** (only for Ham).")

# ----------------------------
# 📁 Tabs for Input Modes
# ----------------------------
tab1, tab2 = st.tabs(["📨 Classify Single Email", "📂 Bulk Upload (CSV)"])

# ============================
# TAB 1: SINGLE EMAIL
# ============================
with tab1:
    st.subheader("✉️ Enter Your Email Below")
    user_input = st.text_area("Paste email content here...")

    if st.button("🚀 Classify"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter an email message.")
        else:
            processed = preprocess_email(user_input)
            features = vectorizer.transform([processed])
            label = model.predict(features)[0]
            confidence = model.predict_proba(features).max()

            # Priority assignment only for HAM
            priority = assign_priority(user_input) if label == "ham" else "N/A"

            # 🔍 Show results
            st.success("✅ Prediction Complete")
            st.write(f"🧠 **Prediction:** `{label.upper()}`")
            st.write(f"📌 **Priority (if HAM):** `{priority}`")
            st.progress(confidence)
            st.info(f"🔍 Confidence Score: `{confidence:.2f}`")

# ============================
# TAB 2: BULK UPLOAD
# ============================
with tab2:
    st.subheader("📤 Upload Email CSV")
    st.markdown("Make sure your file has a `text` column.")

    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            if 'text' not in df.columns:
                st.error("❌ The uploaded file must contain a column named `text`.")
            else:
                df['processed'] = df['text'].apply(preprocess_email)
                features = vectorizer.transform(df['processed'])
                df['prediction'] = model.predict(features)
                df['confidence'] = model.predict_proba(features).max(axis=1)

                # Add priority column for HAM
                df['priority'] = df.apply(
                    lambda row: assign_priority(row['text']) if row['prediction'] == 'ham' else 'N/A',
                    axis=1
                )

                st.success("✅ Bulk Classification Complete")
                st.dataframe(df[['text', 'prediction', 'confidence', 'priority']])

                # Download Results
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Results as CSV", csv, "classified_emails.csv", "text/csv")
        except Exception as e:
            st.error(f"⚠️ Something went wrong: {e}")
