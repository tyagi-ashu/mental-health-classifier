import streamlit as st
import torch
import shap
import numpy as np
import pickle
import re
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from huggingface_hub import hf_hub_download
import nltk
from nltk.corpus import stopwords

# ── NLTK Setup ────────────────────────────────────────────────────────────────
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Mental Health Classifier", page_icon="🧠", layout="centered")
st.title("🧠 Mental Health Status Classifier")
st.markdown("Enter a statement to predict mental health status and see which words influenced the prediction.")

# ── Text Cleaning ─────────────────────────────────────────────────────────────
def clean_statement(statement):
    statement = statement.lower()
    statement = re.sub(r'[^\w\s]', '', statement)
    statement = re.sub(r'\d+', '', statement)
    words = statement.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# ── Load Model, Tokenizer, Label Encoder from Hugging Face ───────────────────
@st.cache_resource
def load_all():
    HF_REPO = "tyagi-ashu/mental-health-classifier"

    with st.spinner("Loading model from Hugging Face..."):
        model = AutoModelForSequenceClassification.from_pretrained(HF_REPO)
        tokenizer = AutoTokenizer.from_pretrained(HF_REPO)

        # Download label encoder
        label_encoder_path = hf_hub_download(
            repo_id=HF_REPO,
            filename="label_encoder.pkl"
        )
        label_encoder = pickle.load(open(label_encoder_path, "rb"))

        model.eval()
        model.to(torch.device("cpu"))

        clf = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=-1,
            return_all_scores=True
        )

    return model, tokenizer, label_encoder, clf

model, tokenizer, label_encoder, clf = load_all()

# ── Predict Function for SHAP ─────────────────────────────────────────────────
def predict_proba(texts):
    clean_texts = [
        " ".join(t) if isinstance(t, (list, tuple)) else str(t)
        for t in texts
    ]
    outputs = clf(clean_texts)
    return np.array([[d['score'] for d in o] for o in outputs])

# ── Single Text Prediction ────────────────────────────────────────────────────
def detect_class(text):
    cleaned = clean_statement(text)
    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.argmax(outputs.logits, dim=1).item()

# ── UI ────────────────────────────────────────────────────────────────────────
user_input = st.text_area(
    "Enter your statement:",
    height=150,
    placeholder="e.g. I have been feeling very hopeless lately..."
)

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        # ── Prediction ────────────────────────────────────────────────────────
        with st.spinner("Predicting..."):
            predicted_idx = detect_class(user_input)
            predicted_label = label_encoder.inverse_transform([predicted_idx])[0]
            probs = predict_proba([user_input])[0]
            confidence = probs[predicted_idx] * 100

        # ── Show Result ───────────────────────────────────────────────────────
        st.markdown("### 🔍 Prediction")
        st.success(f"**{predicted_label}** ({confidence:.1f}% confidence)")

        # ── Probability Chart ─────────────────────────────────────────────────
        st.markdown("### 📊 Class Probabilities")
        prob_dict = {
            label_encoder.classes_[i]: float(probs[i])
            for i in range(len(probs))
        }
        st.bar_chart(prob_dict)

        # ── SHAP Explanation ──────────────────────────────────────────────────
        st.markdown("### 💡 SHAP Explanation")
        st.caption("🔴 Red words push toward this prediction · 🔵 Blue words push away")

        with st.spinner("Generating SHAP explanation (this may take a minute)..."):
            try:
                masker = shap.maskers.Text(tokenizer)
                explainer = shap.Explainer(predict_proba, masker, algorithm="partition")
                shap_values = explainer([user_input])

                fig, ax = plt.subplots()
                shap.plots.text(shap_values[0, :, predicted_idx], display=False)
                st.pyplot(fig)
                plt.close()

            except Exception as e:
                st.error(f"SHAP explanation failed: {e}")