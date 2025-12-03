import streamlit as st
import pandas as pd
import json
import time
import random
import requests # Used for the real API call to Gemini
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION (MANDATORY CANVAS VARIABLES) ---
# NOTE: In a real environment, you would use PRAW for Reddit API access, 
# and a proper API Key for Gemini. Here, we use a placeholder API key and URL.
# The __app_id, __firebase_config, and __initial_auth_token are not needed
# for this specific task (no persistence), but are kept as a template reminder.
API_KEY = "" # The Canvas environment provides this if empty
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"
MAX_RETRIES = 5

# --- HELPER FUNCTIONS FOR API CALLS ---

# Exponential Backoff implementation for robust API calls
def exponential_backoff(attempt):
    """Calculates the delay for exponential backoff."""
    return (2 ** attempt) + random.uniform(0, 1)

def classify_text_with_gemini(comment_text):
    """
    Calls the Gemini API to classify the comment text.
    Uses structured output (JSON) for reliable results.
    """
    
    # System Instruction: Define the model's role and rules
    system_prompt = (
        "You are an expert content moderator. Your task is to classify a given comment based on its potential for hate speech. "
        "Strictly categorize the text into one of three labels: 'Hateful' (explicit slurs, direct threats, promotion of violence/discrimination), "
        "'Offensive' (vulgar language, strong disagreement, trolling, mild insults), or 'Normal' (all other conversational comments). "
        "Your response MUST be a single JSON object matching the provided schema."
    )
    
    # User Query
    user_query = f"Classify the following Reddit comment: \"{comment_text}\""

    # API Payload for Structured Output
    payload = {
        "contents": [{ "parts": [{ "text": user_query }] }],
        "systemInstruction": { "parts": [{ "text": system_prompt }] },
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "classification": { 
                        "type": "STRING", 
                        "description": "The determined label: Hateful, Offensive, or Normal."
                    },
                    "reason": {
                        "type": "STRING",
                        "description": "A very brief, one-sentence explanation for the classification."
                    }
                },
                "required": ["classification", "reason"]
            }
        }
    }

    # API Call with Retries
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                f"{GEMINI_API_URL}?key={API_KEY}",
                headers={'Content-Type': 'application/json'},
                data=json.dumps(payload)
            )
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            
            result = response.json()
            
            # Extract the JSON string from the response
            if (result.get('candidates') and 
                result['candidates'][0].get('content') and 
                result['candidates'][0]['content'].get('parts')):
                
                json_string = result['candidates'][0]['content']['parts'][0]['text']
                # The model returns a string that is a JSON object, so we parse it
                classification_data = json.loads(json_string)
                
                return classification_data.get('classification', 'Error'), classification_data.get('reason', 'N/A')
            
            st.error("API Response structure was invalid.")
            return 'Error', 'Invalid API response structure.'

        except requests.exceptions.HTTPError as e:
            # Handle specific HTTP errors like rate limiting
            if response.status_code == 429 and attempt < MAX_RETRIES - 1:
                delay = exponential_backoff(attempt)
                time.sleep(delay)
                continue
            st.error(f"HTTP Error: {e} - Status Code: {response.status_code}")
            return 'Error', f'API HTTP Error: {response.status_code}'
        
        except Exception as e:
            st.error(f"An unexpected error occurred during API call: {e}")
            return 'Error', f'Unexpected error: {e}'

    return 'Error', 'Max retries exceeded'


@st.cache_data
def fetch_and_analyze_reddit_link(url):
    """
    Mocks the fetching of Reddit data and performs hate speech analysis.
    In a real app, this function would use PRAW to fetch data.
    """
    
    if "reddit.com" not in url:
        return "Invalid URL", None, None

    # --- Step 1: MOCK REDDIT DATA FETCH (Replace with PRAW) ---
    st.info("Step 1: Fetching post and comments (MOCK for demonstration)...")
    time.sleep(1) # Simulate network delay

    # Mock Post Data
    mock_title = "The Future of AI and its Societal Impact"
    mock_post_body = "A discussion about the ethical implications of AI deployment, specifically in high-stakes fields like medicine and law. This is a crucial conversation we need to have globally."
    
    # Mock Comments (Real app would use PRAW to get these)
    mock_comments = [
        "This is an interesting and necessary discussion. The ethical guidelines need to be established before mass deployment.", # Normal
        "This whole AI ethics thing is a joke. Just let the engineers build whatever they want. Stop being soft, you losers.", # Offensive
        "I'm worried about job displacement, but the potential for progress is immense. We just need better regulation.", # Normal
        "Anyone who thinks AI should be regulated is clearly an idiot and a technophobe. You should all just shut up.", # Offensive
        "I hate that kind of people. They are the root of all problems. They should be banned from the internet.", # Hateful (simulated)
        "Great article, very thought-provoking. Thanks for sharing!", # Normal
        "This article is trash and so is the author. Go back to school.", # Offensive
        "Why are there so many negative comments here? Can't we just have a civil discussion?", # Normal
    ]

    st.info(f"Successfully fetched **{len(mock_comments)}** comments for post: **{mock_title}**")
    
    # --- Step 2: HATE SPEECH LABELING (Real API Call) ---
    st.info("Step 2: Analyzing comments for hate speech using Gemini...")
    
    results = []
    
    # Use ThreadPoolExecutor for concurrent API calls (faster processing)
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(classify_text_with_gemini, comment) for comment in mock_comments]
        
        # Collect results
        for future, comment in zip(futures, mock_comments):
            classification, reason = future.result()
            results.append({
                'Comment': comment,
                'Classification': classification,
                'Reason': reason
            })
            
    # --- Step 3: REPORTING & DATA PREPARATION ---
    
    df = pd.DataFrame(results)
    
    # Tidy up the DataFrame columns
    df = df[['Classification', 'Reason', 'Comment']]

    # Generate the classification summary
    classification_counts = df['Classification'].value_counts().reset_index()
    classification_counts.columns = ['Classification', 'Count']
    
    return mock_title, mock_post_body, df, classification_counts

# --- STREAMLIT UI ---

st.set_page_config(
    page_title="Reddit Hate Speech Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛡️ Reddit Comment Moderation Assistant")
st.markdown("Enter a Reddit post URL to fetch comments and analyze for hate speech using an LLM classifier.")

# Input field for Reddit URL
reddit_url = st.text_input(
    "Enter Reddit Post URL", 
    placeholder="e.g., https://www.reddit.com/r/...",
    value="https://www.reddit.com/r/ai/comments/example_thread_id/a_discussion_on_ai_ethics/"
)

# Analysis Button
if st.button("Analyze Post", type="primary"):
    if not reddit_url or "reddit.com" not in reddit_url:
        st.error("Please enter a valid Reddit URL.")
    else:
        with st.spinner('Analyzing... This may take a moment as the model processes the comments.'):
            # Call the analysis function
            post_title, post_body, results_df, counts_df = fetch_and_analyze_reddit_link(reddit_url)
            
            if post_title == "Invalid URL":
                st.error("Please enter a valid Reddit URL.")
            elif results_df is None:
                st.error("Analysis failed. Please check the console for API errors.")
            else:
                st.success("Analysis Complete!")
                
                st.header(f"Post Analysis: {post_title}")
                st.subheader("Post Body")
                st.markdown(f"> {post_body}")

                # --- Report Section ---
                st.header("Classification Report")
                
                # Metrics
                total_comments = counts_df['Count'].sum()
                hateful_count = counts_df[counts_df['Classification'] == 'Hateful']['Count'].iloc[0] if 'Hateful' in counts_df['Classification'].values else 0
                offensive_count = counts_df[counts_df['Classification'] == 'Offensive']['Count'].iloc[0] if 'Offensive' in counts_df['Classification'].values else 0
                
                col1, col2, col3 = st.columns(3)
                
                col1.metric("Total Comments Analyzed", total_comments)
                col2.metric("Hateful Comments Detected", hateful_count)
                col3.metric("Offensive/Trolling Comments", offensive_count)

                # Chart
                st.subheader("Distribution of Comment Classifications")
                
                # Define colors for the chart
                color_map = {
                    'Hateful': '#E84A5F',    # Red
                    'Offensive': '#FFC42B',  # Yellow/Orange
                    'Normal': '#4CAF50'      # Green
                }
                
                # Add a custom column for coloring in Altair
                counts_df['color'] = counts_df['Classification'].map(color_map)

                import altair as alt
                
                # Create the bar chart using Altair (Streamlit's preferred library for rich charts)
                chart = alt.Chart(counts_df).mark_bar().encode(
                    x=alt.X('Classification', sort=['Hateful', 'Offensive', 'Normal']),
                    y=alt.Y('Count', title='Number of Comments'),
                    color=alt.Color('Classification', scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values()))),
                    tooltip=['Classification', 'Count']
                ).properties(
                    height=400
                )
                
                st.altair_chart(chart, use_container_width=True)

                # --- Detailed Table ---
                st.header("Detailed Comment Breakdown")
                st.dataframe(results_df, hide_index=True, use_container_width=True)

                # Add a simple text input for users to test the classifier directly
                st.subheader("Test a Custom Text Snippet")
                test_text = st.text_input("Enter text to classify:", placeholder="e.g., This is a civil discussion.")
                if st.button("Classify Text"):
                    if test_text:
                        st.info("Classifying custom text...")
                        classification, reason = classify_text_with_gemini(test_text)
                        st.markdown(f"**Classification:** `{classification}`")
                        st.markdown(f"**Reason:** *{reason}*")
                    else:
                        st.warning("Please enter some text to classify.")
