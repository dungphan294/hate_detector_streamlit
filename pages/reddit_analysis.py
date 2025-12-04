import streamlit as st
import requests
from urllib.parse import urlsplit, urlunsplit
from datetime import datetime, timezone
from model_utils import load_model, MODEL_NAME, classify_text
import html
import re

# ============================ 
# CONFIGURATION
# ============================ 
st.set_page_config(
    page_title="Reddit Post Analysis",
    page_icon="📊",
    layout="wide",
)

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
# HELPER FUNCTIONS
# ============================ 

def format_timestamp(ts: float) -> str:
    """Convert Unix timestamp to readable UTC format."""
    try:
        return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return "N/A"

def clean_reddit_url(url: str) -> str:
    """Remove query params/fragments and return the base Reddit URL."""
    parts = urlsplit(url.strip())
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))

def build_json_url(url: str) -> str:
    """Convert a normal Reddit URL into its .json endpoint."""
    cleaned = clean_reddit_url(url)
    if cleaned.endswith("/"):
        cleaned = cleaned[:-1]
    return cleaned + ".json"

def is_valid_reddit_url(url: str) -> bool:
    """Check if URL is a valid Reddit post URL."""
    pattern = r'https?://(www\.|old\.)?reddit\.com/r/\w+/comments/[\w]+/'
    return bool(re.match(pattern, url))

def extract_comments(children, results, depth=0):
    """
    Recursively extract ALL comments with hierarchical structure.
    
    Each comment includes:
    - id: comment identifier
    - parent_id: raw Reddit parent (t1_xxx for comment, t3_xxx for post)
    - parent_comment_id: clean comment ID it replies to (None if top-level)
    - depth: nesting level (0=top-level, 1=reply, etc.)
    """
    for child in children:
        if child.get("kind") != "t1":
            continue
            
        data = child.get("data", {})
        body = data.get("body", "")
        
        # Skip deleted/removed comments
        if body in ("[deleted]", "[removed]"):
            continue
        
        comment_id = data.get("id")
        parent_id = data.get("parent_id")
        
        # Extract parent comment ID if parent is a comment
        parent_comment_id = None
        if parent_id and parent_id.startswith("t1_"):
            parent_comment_id = parent_id.split("_", 1)[1]
        
        # Attach metadata
        data["_depth"] = depth
        data["_comment_id"] = comment_id
        data["_parent_id_raw"] = parent_id
        data["_parent_comment_id"] = parent_comment_id
        
        results.append(data)
        
        # Recurse into replies
        replies = data.get("replies")
        if isinstance(replies, dict):
            reply_children = replies.get("data", {}).get("children", [])
            extract_comments(reply_children, results, depth=depth + 1)
    
    return results

def fetch_reddit_post(url: str):
    """
    Fetch Reddit post and comments using public JSON endpoint.
    Returns (post_data, comments, error_msg).
    """
    json_url = build_json_url(url)
    headers = {
        "User-Agent": "reddit-analysis-tool/1.0"
    }
    
    try:
        r = requests.get(json_url, headers=headers, timeout=15)
    except requests.exceptions.Timeout:
        return None, None, "Request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return None, None, f"Network error: {str(e)}"
    
    if r.status_code == 404:
        return None, None, "Post not found. Please check the URL."
    elif r.status_code == 429:
        return None, None, "Rate limited by Reddit. Please wait a moment and try again."
    elif r.status_code != 200:
        return None, None, f"Reddit returned status code {r.status_code}"
    
    try:
        data = r.json()
    except Exception:
        return None, None, "Failed to parse Reddit response."
    
    if not isinstance(data, list) or len(data) < 1:
        return None, None, "Unexpected response format from Reddit."
    
    # Extract post data
    try:
        post_data = data[0]["data"]["children"][0]["data"]
    except (KeyError, IndexError):
        return None, None, "Could not extract post data from response."
    
    # Extract comments
    comments = []
    if len(data) > 1 and "data" in data[1]:
        top_children = data[1]["data"].get("children", [])
        comments = extract_comments(top_children, [])
    
    return post_data, comments, None

def format_number(num: int) -> str:
    """Format large numbers with k/M suffix."""
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}k"
    return str(num)

# ============================ 
# UI COMPONENTS
# ============================ 

def render_post(post):
    """Render the main post information."""
    title = post.get("title", "(no title)")
    author = post.get("author", "(unknown)")
    subreddit = post.get("subreddit_name_prefixed", post.get("subreddit", ""))
    score = post.get("score", 0)
    created = format_timestamp(post.get("created_utc", 0))
    num_comments = post.get("num_comments", 0)
    permalink = post.get("permalink", "")
    
    st.markdown(
        f"""
        <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #ff4500;">
            <h2 style="margin-top: 0; color: #1a1a1b;">{html.escape(title)}</h2>
            <p style="color: #7c7c7c; margin: 8px 0;">
                <strong>Author:</strong> u/{html.escape(author)} &nbsp;|&nbsp; 
                <strong>Subreddit:</strong> {html.escape(subreddit)}
            </p>
            <p style="color: #7c7c7c; margin: 8px 0;">
                <strong>Score:</strong> {format_number(score)} &nbsp;|&nbsp; 
                <strong>Comments:</strong> {format_number(num_comments)} &nbsp;|&nbsp; 
                <strong>Created:</strong> {created}
            </p>
            <p style="color: #0079d3; margin: 8px 0 0 0;">
                <a href="https://reddit.com{permalink}" target="_blank" style="text-decoration: none;">
                    🔗 View on Reddit
                </a>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Post content
    selftext = post.get("selftext", "").strip()
    url_overridden = post.get("url_overridden_by_dest") or post.get("url")
    
    if selftext:
        st.markdown("### 📄 Post Body")
        st.markdown(selftext)
    elif url_overridden and "reddit.com" not in url_overridden:
        st.markdown("### 🔗 Link Post")
        st.markdown(f"[{url_overridden}]({url_overridden})")

def render_comment(comment, classification_result):
    """Render a single comment with classification result."""
    author = comment.get("author", "(unknown)")
    score = comment.get("score", 0)
    created = format_timestamp(comment.get("created_utc", 0))
    body = comment.get("body", "").strip()
    depth = comment.get("_depth", 0)
    comment_id = comment.get("_comment_id")
    parent_comment_id = comment.get("_parent_comment_id")
    
    # Calculate indent
    indent_px = depth * 24
    
    # Reply info
    reply_info = ""
    if parent_comment_id:
        reply_info = f' • reply to: {parent_comment_id}'
    
    # Create container with indent
    container = st.container()
    with container:
        cols = st.columns([indent_px if indent_px > 0 else 0.01, 1000])
        with cols[1]:
            with st.container(border=True):
                # Header
                st.markdown(
                    f'<div style="font-size: 12px; color: #7c7c7c;"><strong>u/{author}</strong> • {created} • ID: {comment_id}{reply_info}</div>',
                    unsafe_allow_html=True
                )
                
                # Body
                st.markdown(body)
                
                # Classification result
                if classification_result:
                    content = classification_result["content"]
                    score_val = classification_result["score"]
                    status = classification_result["status"]
                    color = classification_result["color"]
                    code = classification_result["code"]
                    
                    st.markdown(
                        f"""
                        <div style="
                            border-radius: 8px;
                            padding: 12px;
                            margin-top: 12px;
                            border: 1px solid {color};
                            background-color: {color}15;
                        ">
                            <p style="margin: 0; font-size: 0.9rem;">
                                <strong>Classification:</strong>
                                <span style="color:{color}; font-weight:600;">{status}</span>
                                <strong>Confidence:</strong> 
                                <span>{score_val:.1%}</span>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                
                # Footer
                st.caption(f"👍 {format_number(score)} upvotes")


# ============================ 
# MAIN APP
# ============================ 

st.title("📊 Reddit Post Analysis with Hate Speech Detection")

st.markdown(
    """
    Analyze any Reddit post by pasting its URL below. This tool will:
    1. Fetch the post content and all comments
    2. Automatically classify each comment for hate speech
    
    **Supported URL formats:**
    - `https://www.reddit.com/r/subreddit/comments/post_id/title/`
    - `https://old.reddit.com/r/subreddit/comments/post_id/title/`
    """
)

st.markdown("---")

# Input section
col1, col2 = st.columns([4, 1])
with col1:
    url = st.text_input(
        "Reddit Post URL",
        placeholder="https://www.reddit.com/r/AskReddit/comments/xxxxx/title/",
        label_visibility="collapsed"
    )
with col2:
    fetch_button = st.button("🔍 Analyze Post", use_container_width=True, type="primary")

if fetch_button:
    if not url.strip():
        st.warning("⚠️ Please enter a Reddit post URL.")
    elif not is_valid_reddit_url(url):
        st.error("❌ Invalid Reddit URL format. Please use a direct link to a Reddit post.")
    else:
        with st.spinner("Fetching data from Reddit..."):
            post, comments, error = fetch_reddit_post(url)
            
            if error:
                st.error(f"❌ {error}")
            elif post is not None:
                # Render post
                st.subheader("📝 Post")
                render_post(post)
                
                st.markdown("---")
                
                # Render comments with classification
                st.subheader(f"💬 Comments Analysis ({len(comments)} total)")
                
                if not comments:
                    st.info("No comments found or all comments are deleted/removed.")
                else:
                    # Progress bar for classification
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Classify all comments
                    total_comments = len(comments)
                    classified_comments = []
                    
                    for idx, comment in enumerate(comments):
                        # Update progress
                        progress = (idx + 1) / total_comments
                        progress_bar.progress(progress)
                        status_text.text(f"Classifying comments... {idx + 1}/{total_comments}")
                        
                        # Classify comment
                        body = comment.get("body", "").strip()
                        classification = classify_text(nlp, body)
                        classified_comments.append((comment, classification))
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Show summary statistics
                    hate_count = sum(1 for _, cls in classified_comments if cls and cls.get("code") in ["HS", "OFF"])
                    safe_count = total_comments - hate_count
                    
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("Total Comments", total_comments)
                    with col_stat2:
                        st.metric("Safe Comments", safe_count, delta=f"{(safe_count/total_comments)*100:.1f}%")
                    with col_stat3:
                        st.metric("Flagged Comments", hate_count, delta=f"{(hate_count/total_comments)*100:.1f}%", delta_color="inverse")
                    
                    st.markdown("---")
                    
                    # Display all comments with classification
                    for comment, classification in classified_comments:
                        render_comment(comment, classification)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #7c7c7c; font-size: 12px;">
        Built with Streamlit • Data from Reddit's public JSON API • AI-powered hate speech detection
    </div>
    """,
    unsafe_allow_html=True
)