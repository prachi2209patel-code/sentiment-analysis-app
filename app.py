import nltk

try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# =========================
# 🔹 NLTK Downloads (run once)
# =========================
nltk.download('punkt')
nltk.download('stopwords')

# =========================
# 🔹 Load Model
# =========================
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Stopwords
stop_words = set(stopwords.words('english'))

# =========================
# 🔹 Preprocessing Function
# =========================
def preprocess(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    
    return " ".join(tokens)

# =========================
# 🔹 Streamlit UI
# =========================

st.set_page_config(page_title="Spam Classifier", layout="centered")

st.title("📧 Spam Email Classifier")

st.write("Enter an email message to check whether it is Spam or Ham.")

# Input
user_input = st.text_area("✍️ Enter Email Text Here")

# =========================
# 🔹 Prediction Button
# =========================
if st.button("🔍 Predict"):
    
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text!")
    
    else:
        # Preprocess
        cleaned_text = preprocess(user_input)

        # =========================
        # 🔹 Rule-based Spam Check
        # =========================
        spam_keywords = ["urgent", "verify", "account", "click", "winner"]

        if any(word in cleaned_text for word in spam_keywords):
            st.subheader("🧹 Preprocessed Text")
            st.write(cleaned_text)

            st.subheader("📊 Prediction Result")
            st.error("🚫 Spam Message (rule-based)")

        else:
            # =========================
            # 🔹 ML Prediction
            # =========================
            vector = vectorizer.transform([cleaned_text])
            prediction = model.predict(vector)[0]

            # Fix numeric labels (if model outputs 0/1)
            if prediction == 0:
                prediction = "ham"
            elif prediction == 1:
                prediction = "spam"

            # Confidence score
            try:
                prob = model.predict_proba(vector).max()
            except:
                prob = 0.0

            # =========================
            # 🔹 Display Output
            # =========================
            st.subheader("🧹 Preprocessed Text")
            st.write(cleaned_text)

            st.subheader("📊 Prediction Result")

            if prediction == "ham":
                st.success(f"✅ Ham (Not Spam)\n\nConfidence: {prob:.2f}")
            else:
                st.error(f"🚫 Spam Message\n\nConfidence: {prob:.2f}")
