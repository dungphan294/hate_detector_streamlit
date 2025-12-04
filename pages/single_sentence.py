import streamlit as st
from model_utils import load_model, MODEL_NAME, classify_text

st.set_page_config(
    page_title="Single Sentence Classifier",
    page_icon="🛡️",
    layout="wide",
)

nlp, device, setup_time = load_model()

# Sidebar
with st.sidebar:
    st.header("⚙️ Model Info")
    st.write(f"**Model:** `{MODEL_NAME}`")
    st.write(f"**Device:** `{device}`")
    st.write(f"**Setup time:** `{setup_time:.2f} s`")
    st.markdown("---")
    st.caption("Tip: Try neutral, rude, and clearly hateful sentences to see differences.")
    st.caption("Navigate using the links below:")
    
    st.page_link("pages/single_sentence.py", label="📝 Single Sentence Classifier")
    st.page_link("pages/reddit_analysis.py", label="📚 Reddit Post Analysis")

st.title("📝 Single Sentence Hate Speech Classifier")
st.markdown(
    "Classify a single sentence into one of the hate categories or as non-hate."
)
st.markdown("---")

left_col, right_col = st.columns([1.3, 1])

with left_col:
    st.subheader("✍️ Input")
    default_example = "I love all of you, you are amazing!"

    if "single_input" not in st.session_state:
        st.session_state.single_input = ""

    def use_example():
        st.session_state.single_input = default_example

    text = st.text_area(
        "Enter a sentence to analyse:",
        height=150,
        placeholder=default_example,
        key="single_input",
    )

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        run = st.button("🔍 Classify", type="primary")
    with col_btn2:
        st.button("Use example", on_click=use_example)

with right_col:
    st.subheader("📊 Result")

    if not text and not run:
        st.info("Enter a sentence on the left and press **Classify**.")
    elif run:
        result = classify_text(nlp, text)

        if result is None:
            st.warning("Please enter non-empty text.")
        else:
            content = result["content"]
            score = result["score"]
            reason = result["reason"]
            status = result["status"]
            color = result["color"]
            code = result["code"]

            st.markdown(
                f"""
                <div style="
                    border-radius: 12px;
                    padding: 16px 18px;
                    border: 1px solid #e5e7eb;
                    background-color: #f9fafb;
                ">
                    <h4 style="margin-top: 0; margin-bottom: 8px;">Prediction</h4>
                    <p style="margin: 0 0 4px 0;">
                        <strong>Status:</strong>
                        <span style="color:{color}; font-weight:600;">{status}</span>
                        <span style="color:#9ca3af; font-size:0.85rem;"> (code: {code})</span>
                    </p>
                    <p style="margin: 0 0 8px 0;">
                        <strong>Content:</strong> {content}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("")
            m_col1, m_col2 = st.columns([1, 2])
            with m_col1:
                st.metric("Confidence", f"{score:.2%}")
            with m_col2:
                st.write("Confidence level")
                st.progress(score)

            st.markdown("### 🧠 Reasoning")
            st.markdown(reason)
