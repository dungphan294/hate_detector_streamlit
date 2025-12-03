import streamlit as st
import pandas as pd
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import altair as alt

# =========================
#  HUGGING FACE HATE MODEL
# =========================

MODEL_NAME = "cardiffnlp/twitter-roberta-base-hate-latest"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_hate_model():
    """Load the CardiffNLP hate-speech model once (cached)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if device.type == "cuda" else -1,
        return_all_scores=True,  # we want scores for all labels
    )
    return clf


hate_pipeline = load_hate_model()


def classify_text_with_model(comment_text: str):
    """
    Use the CardiffNLP hate-speech model to classify a comment.
    We convert the model's hate probability into three buckets:
    - Hateful
    - Offensive
    - Normal
    """
    text = comment_text.strip()
    if not text:
        return "Normal", "Empty or whitespace-only comment."

    # Model inference
    out = hate_pipeline(text, truncation=True, max_length=512)[0]
    # out is a list of dicts like:
    # [{'label': 'LABEL_0', 'score': 0.85}, {'label': 'LABEL_1', 'score': 0.15}]

    score_hate = None
    score_non_hate = None
    for item in out:
        label = item["label"]
        if label.endswith("_0"):
            score_non_hate = item["score"]
        elif label.endswith("_1"):
            score_hate = item["score"]

    if score_hate is None or score_non_hate is None:
        return "Error", "Model output did not contain expected labels."

    # Simple, tunable thresholds:
    #  - hate >= 0.80         -> Hateful
    #  - 0.40 <= hate < 0.80  -> Offensive
    #  - hate < 0.40          -> Normal
    if score_hate >= 0.80:
        classification = "Hateful"
        reason = f"High hate probability ({score_hate:.2f})."
    elif score_hate >= 0.40:
        classification = "Offensive"
        reason = (
            f"Moderate hate probability ({score_hate:.2f}); "
            "likely insulting/offensive but not extreme."
        )
    else:
        classification = "Normal"
        reason = f"Low hate probability ({score_hate:.2f})."

    return classification, reason


# =========================
#  MOCK REDDIT PIPELINE
# =========================

@st.cache_data
def fetch_and_analyze_reddit_link(url: str):
    """
    Mock fetching a Reddit post + comments and analyze with the hate model.
    In a real app, this function would use PRAW to fetch real data.
    """

    if "reddit.com" not in url:
        return "Invalid URL", None, None, None

    # --- Step 1: MOCK REDDIT DATA FETCH ---
    st.info("Step 1: Fetching post and comments (MOCK for demonstration)...")
    time.sleep(1)  # Simulate network delay

    # Mock Post Data
    mock_title = "The Future of AI and its Societal Impact"
    mock_post_body = (
        "A discussion about the ethical implications of AI deployment, "
        "specifically in high-stakes fields like medicine and law. "
        "This is a crucial conversation we need to have globally."
    )

    # Mock Comments (Real app would use PRAW to get these)
    mock_comments = [
        "This is an interesting and necessary discussion. The ethical guidelines need to be established before mass deployment.",  # Normal
        "This whole AI ethics thing is a joke. Just let the engineers build whatever they want. Stop being soft, you losers.",       # Offensive
        "I'm worried about job displacement, but the potential for progress is immense. We just need better regulation.",           # Normal
        "Anyone who thinks AI should be regulated is clearly an idiot and a technophobe. You should all just shut up.",            # Offensive
        "I hate that kind of people. They are the root of all problems. They should be banned from the internet.",                 # Hateful (simulated)
        "Great article, very thought-provoking. Thanks for sharing!",                                                              # Normal
        "This article is trash and so is the author. Go back to school.",                                                          # Offensive
        "Why are there so many negative comments here? Can't we just have a civil discussion?",                                    # Normal
    ]

    st.info(f"Successfully fetched **{len(mock_comments)}** comments for post: **{mock_title}**")

    # --- Step 2: HATE SPEECH LABELING ---
    st.info("Step 2: Analyzing comments for hate/offense using local HF model...")

    results = []
    for comment in mock_comments:
        classification, reason = classify_text_with_model(comment)
        results.append(
            {
                "Comment": comment,
                "Classification": classification,
                "Reason": reason,
            }
        )

    # --- Step 3: REPORTING & DATA PREPARATION ---

    df = pd.DataFrame(results)
    df = df[["Classification", "Reason", "Comment"]]

    classification_counts = df["Classification"].value_counts().reset_index()
    classification_counts.columns = ["Classification", "Count"]

    return mock_title, mock_post_body, df, classification_counts


# =========================
#  STREAMLIT UI
# =========================

st.set_page_config(
    page_title="Reddit Hate Speech Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛡️ Reddit Comment Moderation Assistant")
st.markdown(
    "Enter a Reddit post URL to (mock) fetch comments and analyze them for hate/offense "
    "using a local Hugging Face hate-speech model."
)

# Input field for Reddit URL
reddit_url = st.text_input(
    "Enter Reddit Post URL",
    placeholder="e.g., https://www.reddit.com/r/.../comments/...",
    value="https://www.reddit.com/r/ai/comments/example_thread_id/a_discussion_on_ai_ethics/",
)

# Analysis Button
if st.button("Analyze Post", type="primary"):
    if not reddit_url or "reddit.com" not in reddit_url:
        st.error("Please enter a valid Reddit URL.")
    else:
        with st.spinner(
            "Analyzing... This may take a moment as the model processes the comments."
        ):
            post_title, post_body, results_df, counts_df = fetch_and_analyze_reddit_link(
                reddit_url
            )

            if post_title == "Invalid URL":
                st.error("Please enter a valid Reddit URL.")
            elif results_df is None or counts_df is None:
                st.error("Analysis failed. Check the logs for details.")
            else:
                st.success("Analysis Complete!")

                # --- Post Info ---
                st.header(f"Post Analysis: {post_title}")
                st.subheader("Post Body")
                st.markdown(f"> {post_body}")

                # --- Report Section ---
                st.header("Classification Report")

                total_comments = int(counts_df["Count"].sum()) if not counts_df.empty else 0

                hateful_count = 0
                offensive_count = 0
                if not counts_df.empty:
                    hateful_row = counts_df[counts_df["Classification"] == "Hateful"]
                    offensive_row = counts_df[counts_df["Classification"] == "Offensive"]
                    if not hateful_row.empty:
                        hateful_count = int(hateful_row["Count"].iloc[0])
                    if not offensive_row.empty:
                        offensive_count = int(offensive_row["Count"].iloc[0])

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Comments Analyzed", total_comments)
                col2.metric("Hateful Comments Detected", hateful_count)
                col3.metric("Offensive/Trolling Comments", offensive_count)

                # --- Chart ---
                st.subheader("Distribution of Comment Classifications")

                color_map = {
                    "Hateful": "#E84A5F",   # Red
                    "Offensive": "#FFC42B", # Yellow/Orange
                    "Normal": "#4CAF50",    # Green
                    "Error": "#999999",     # Grey
                }

                counts_df["color"] = counts_df["Classification"].map(
                    lambda x: color_map.get(x, "#999999")
                )

                chart = (
                    alt.Chart(counts_df)
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            "Classification",
                            sort=["Hateful", "Offensive", "Normal", "Error"],
                        ),
                        y=alt.Y("Count", title="Number of Comments"),
                        color=alt.Color(
                            "Classification",
                            scale=alt.Scale(
                                domain=list(color_map.keys()),
                                range=list(color_map.values()),
                            ),
                        ),
                        tooltip=["Classification", "Count"],
                    )
                    .properties(height=400)
                )

                st.altair_chart(chart, use_container_width=True)

                # --- Detailed Table ---
                st.header("Detailed Comment Breakdown")
                st.dataframe(results_df, hide_index=True, use_container_width=True)

# =========================
#  CUSTOM TEXT TESTER
# =========================

st.subheader("Test a Custom Text Snippet")
test_text = st.text_input(
    "Enter text to classify:", placeholder="e.g., This is a civil discussion."
)

if st.button("Classify Text"):
    if test_text:
        st.info("Classifying custom text with local HF model...")
        classification, reason = classify_text_with_model(test_text)
        st.markdown(f"**Classification:** `{classification}`")
        st.markdown(f"**Reason:** *{reason}*")
    else:
        st.warning("Please enter some text to classify.")
