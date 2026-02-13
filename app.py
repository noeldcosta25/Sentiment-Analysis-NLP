import os
import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="Sentiment Analysis",layout="centered")

st.title("Sentiment Analysis Streamlit")
st.write("This app uses a fine-tuned BERT model")

# --------------------------------------------------
# Paths & constants
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Bert sentiment model")
MAX_LEN = 64

ID_TO_LABEL = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# --------------------------------------------------
# Load model & tokenizer (cached)
# --------------------------------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# --------------------------------------------------
# Prediction function
# --------------------------------------------------
def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LEN
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    pred_id = int(np.argmax(probs))
    confidence = float(probs[pred_id])

    return ID_TO_LABEL[pred_id], confidence, probs

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("BERT Sentiment Analysis")
st.write("Analyze sentiment using a fine-tuned **BERT** model.")

text_input = st.text_area(
    "Enter text",
    placeholder="Example: I absolutely loved this product!",
    height=120
)

if st.button("Analyze Sentiment"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        label, confidence, probs = predict_sentiment(text_input)

        st.subheader("Prediction")
        if label == "Positive":
            st.success(f"Sentiment: **{label}**")
        elif label == "Negative":
            st.error(f"Sentiment: **{label}**")
        else:
            st.info(f"Sentiment: **{label}**")

        st.write(f"**Confidence:** {confidence:.2f}")

        st.subheader("Class Probabilities")
        for i, p in enumerate(probs):
            st.write(f"{ID_TO_LABEL[i]}: {p:.2f}")
