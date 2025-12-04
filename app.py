import streamlit as st
from model_utils import MODEL_NAME, load_model

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(
    page_title="Hate Speech App",
    page_icon="🛡️",
    layout="wide",
)

# ============================
# SIDEBAR
# ============================
nlp, device, setup_time = load_model()

with st.sidebar:
    st.header("⚙️ Model Info")
    st.write(f"**Model:** `{MODEL_NAME}`")
    st.write(f"**Device:** `{device}`")
    st.write(f"**Setup time:** `{setup_time:.2f} s`")
    st.markdown("---")
    st.caption("Navigate using the links below:")

    st.page_link("pages/single_sentence.py", label="📝 Single Sentence Classifier")
    st.page_link("pages/reddit_analysis.py", label="📚 Reddit Post Analysis")

# ============================
# MAIN PAGE
# ============================
st.title("🛡️ Hate Speech Detection Demo")

st.markdown(
    """
<div style="border-radius: 12px; padding: 16px; background-color: #f9fafb; border: 1px solid #e5e7eb;">
Welcome to the **Hate Speech Detection** demo app.  
This tool allows you to classify text as safe, abusive, or hateful using a RoBERTa-based model.  
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("---")
st.subheader("📖 Instructions")
st.write(
    """
- Use the sidebar to access the **Single Sentence Classifier** or the **Batch Classifier**.  
- Enter text and view the classification result with confidence scores.  
- Results are color-coded for clarity (green = safe, orange = warning, red = danger, etc.).
"""
)
