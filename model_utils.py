# model_utils.py

import time
from typing import Dict, Any, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import streamlit as st

MODEL_NAME = "cardiffnlp/twitter-roberta-base-hate-multiclass-latest"

LABEL_MAP: Dict[str, int] = {
    "sexism": 0,
    "racism": 1,
    "disability": 2,
    "sexual_orientation": 3,
    "religion": 4,
    "other": 5,
    "not_hate": 6,
}

RISK_MAP: Dict[str, Dict[str, str]] = {
    "sexism": {
        "status": "Sexism",
        "color": "#f59e0b",
        "icon": "⚠️",
    },
    "racism": {
        "status": "Racism",
        "color": "#ef4444",
        "icon": "⛔",
    },
    "disability": {
        "status": "Disability",
        "color": "#b91c1c",
        "icon": "⛔",
    },
    "sexual_orientation": {
        "status": "Sexual Orientation",
        "color": "#8b5cf6",
        "icon": "⛔",
    },
    "religion": {
        "status": "Religion",
        "color": "#0ea5e9",
        "icon": "⛔",
    },
    "other": {
        "status": "Other",
        "color": "#6b7280",
        "icon": "ℹ️",
    },
    "not_hate": {
        "status": "Safe",
        "color": "#16a34a",
        "icon": "✅",
    },
}


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        num = torch.cuda.device_count()
        idx = 4 if num > 4 else 0
        return torch.device(f"cuda:{idx}")
    return torch.device("cpu")


@st.cache_resource
def load_model() -> Tuple[Any, torch.device, float]:
    device = _get_device()

    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)

    nlp = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device.index if device.type == "cuda" else -1,
        truncation=True,
        max_length=128,
    )
    setup_time = time.time() - start
    return nlp, device, setup_time


def classify_text(nlp, text: str):
    text = text.strip()
    if not text:
        return None

    result = nlp(text)[0]
    raw_label = result["label"].lower().strip()
    score = float(result["score"])

    label = raw_label if raw_label in LABEL_MAP else "other"
    code = LABEL_MAP[label]

    risk_info = RISK_MAP.get(label, RISK_MAP["other"])
    status = risk_info["status"]
    color = risk_info["color"]
    icon = risk_info["icon"]

    reason = (
        f"{icon} This text is classified as **{label}** "
        f"(*{status}*) with **{score:.2%}** confidence."
    )

    return {
        "content": text,
        "label": label,
        "code": code,
        "score": score,
        "reason": reason,
        "status": status,
        "color": color,
    }
