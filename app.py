import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# =========================
# LOAD HUGGING FACE MODEL
# =========================

MODEL_NAME = "cardiffnlp/twitter-roberta-base-hate-latest"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_hate_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)

    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if device.type == "cuda" else -1,
        return_all_scores=True,
    )
    return clf

hate_pipeline = load_hate_model()


# =========================
# CLASSIFICATION FUNCTION
# =========================

def classify_text(text: str):
    """
    Classify text using the Cardiff hate speech model.
    Returns: (label, reason)
    """
    if not text.strip():
        return "Normal", "Empty input."

    out = hate_pipeline(text, truncation=True, max_length=512)[0]

    # Parse scores
    score_hate = None
    score_non_hate = None
    for item in out:
        if item["label"].endswith("_1"):
            score_hate = item["score"]
        elif item["label"].endswith("_0"):
            score_non_hate = item["score"]

    if score_hate is None:
        return "Error", "Unexpected model output."

    # Simple thresholds
    if score_hate >= 0.80:
        return "Hateful", f"High hate probability ({score_hate:.2f})"
    elif score_hate >= 0.40:
        return "Offensive", f"Medium hate probability ({score_hate:.2f})"
    else:
        return "Normal", f"Low hate probability ({score_hate:.2f})"


# =========================
# STREAMLIT UI
# =========================

st.title("🛡️ Hate Speech Classifier (Single Sentence)")

text = st.text_area("Enter a sentence:", height=100, placeholder="Type something...")

if st.button("Classify"):
    label, reason = classify_text(text)
    st.subheader("Result")
    st.write(f"**Classification:** `{label}`")
    st.write(f"**Reason:** {reason}")
