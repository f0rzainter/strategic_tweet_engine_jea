import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import json
import hashlib
from dotenv import load_dotenv
from openai import OpenAI
from newspaper import Article

st.set_page_config(page_title="Jamil Strategic Tweet Engine", layout="wide")

# Apply Arial font for text, Material Icons for icons
st.markdown("""
<style>
    /* Import Material Icons fonts */
    @import url('https://fonts.googleapis.com/icon?family=Material+Icons');
    @import url('https://fonts.googleapis.com/icon?family=Material+Icons+Outlined');
    
    /* Explicitly restore icon fonts with high specificity - must come first */
    i.material-icons,
    i.material-icons-outlined,
    span.material-icons,
    span.material-icons-outlined,
    [class*="material-icons"],
    [class*="MaterialIcons"] {
        font-family: 'Material Icons', 'Material Icons Outlined' !important;
        font-weight: normal !important;
        font-style: normal !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        display: inline-block !important;
        white-space: nowrap !important;
        direction: ltr !important;
        -webkit-font-smoothing: antialiased !important;
        text-rendering: optimizeLegibility !important;
        -moz-osx-font-smoothing: grayscale !important;
    }
    
    /* Apply Arial to specific text elements only (scoped, no !important on broad selectors) */
    body,
    p,
    h1, h2, h3, h4, h5, h6,
    label,
    input,
    textarea,
    button,
    .stMarkdown,
    .stTextInput,
    .stSelectbox,
    .stRadio,
    .stCheckbox,
    .stButton {
        font-family: Arial, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

st.title("Jamil Strategic Tweet Engine")

# ---------- Helpers ----------
REQUIRED_COLUMNS = ["text", "created_at", "favorite_count", "view_count"]

DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Lowercase and trim column names to reduce CSV format pain
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def validate_columns(df: pd.DataFrame) -> list[str]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return missing

def prepare_tweets_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Coerce numeric
    df["favorite_count"] = pd.to_numeric(df["favorite_count"], errors="coerce").fillna(0)
    df["view_count"] = pd.to_numeric(df["view_count"], errors="coerce").fillna(0)

    # Add standardized alias columns for consistent use across tabs
    df["favorites"] = df["favorite_count"]
    df["views"] = df["view_count"]

    # Parse datetime
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)

    # Drop rows with no text or no timestamp (keeps charts clean)
    df["text"] = df["text"].astype(str).fillna("")
    df = df[df["text"].str.strip() != ""]
    df = df[df["created_at"].notna()]

    # Derived fields
    df["engagement_rate"] = df.apply(
        lambda r: (r["favorites"] / r["views"]) if r["views"] and r["views"] > 0 else 0.0,
        axis=1,
    )
    df["hour"] = df["created_at"].dt.hour
    df["day"] = df["created_at"].dt.day_name().str[:3]  # Mon/Tue/...
    df["day"] = pd.Categorical(df["day"], categories=DAY_ORDER, ordered=True)

    return df

# ---------- OpenAI Helpers ----------
def get_openai_api_key():
    """Load OpenAI API key from .env file in the same folder as app.py."""
    # Get the directory where app.py is located
    app_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(app_dir, ".env")
    
    # Load .env file with override to ensure we get the latest values
    load_dotenv(dotenv_path=env_path, override=True)
    
    # Get and strip the key
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    return api_key

def validate_openai_api_key(api_key: str) -> tuple[bool, str]:
    """
    Validate OpenAI API key format.
    Returns (is_valid, error_message)
    """
    if not api_key:
        return False, "API key is missing or empty."
    
    if not api_key.startswith("sk-"):
        return False, f"API key does not start with 'sk-'. Found prefix: '{api_key[:3] if len(api_key) >= 3 else 'too short'}'"
    
    return True, ""

def test_openai_api_key(api_key: str, model: str = "gpt-4o-mini") -> tuple[bool, str]:
    """
    Test OpenAI API key by making a minimal API call.
    Returns (success, message)
    """
    try:
        client = OpenAI(api_key=api_key)
        # Make a minimal test call
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "ping"}
            ],
            max_tokens=5
        )
        return True, "API key is valid and working."
    except Exception as e:
        return False, str(e)

def prepare_tweet_corpus(df: pd.DataFrame, exclude_replies: bool = True, max_tweets: int = 300) -> str:
    """Prepare tweet corpus for analysis: deduplicate, filter, sample."""
    df_corpus = df.copy()
    
    # Filter replies if requested
    if exclude_replies:
        df_corpus = df_corpus[~df_corpus["text"].str.strip().str.startswith("@")]
    
    # Deduplicate identical tweet texts
    df_corpus = df_corpus.drop_duplicates(subset=["text"])
    
    # Strip whitespace
    df_corpus["text"] = df_corpus["text"].str.strip()
    
    # Remove empty texts
    df_corpus = df_corpus[df_corpus["text"] != ""]
    
    # Sample deterministically if needed
    if len(df_corpus) > max_tweets:
        df_corpus = df_corpus.sample(n=max_tweets, random_state=42).reset_index(drop=True)
    
    # Combine into corpus string with separators
    corpus = "\n---\n".join(df_corpus["text"].tolist())
    
    return corpus

def hash_tweet_corpus(corpus: str) -> str:
    """Create a hash of the tweet corpus for caching."""
    return hashlib.md5(corpus.encode()).hexdigest()

@st.cache_data
def analyze_topics_cached(
    corpus_hash: str,
    corpus: str,
    model: str,
    temperature: float,
    exclude_replies: bool,
    max_tweets: int,
    api_key: str
) -> str:
    """Cached function to analyze topics using OpenAI API."""
    # The cache key includes corpus_hash, model, temperature, exclude_replies, max_tweets, and api_key
    # Note: api_key is included to invalidate cache if key changes, but we don't want to expose it
    return analyze_topics_openai(corpus, model, temperature, api_key)

def analyze_topics_openai(corpus: str, model: str, temperature: float, api_key: str) -> str:
    """Call OpenAI API to analyze topics."""
    # API key should already be validated before calling this function
    client = OpenAI(api_key=api_key)
    
    prompt = f"""Analyze the following collection of tweets and identify the top 5 "Core Pillars" (main topics/themes) that this account focuses on.

The output must be STRICT JSON with this exact schema:
{{
  "pillars": [
    {{ "topic_name": "...", "description": "..." }},
    {{ "topic_name": "...", "description": "..." }},
    {{ "topic_name": "...", "description": "..." }},
    {{ "topic_name": "...", "description": "..." }},
    {{ "topic_name": "...", "description": "..." }}
  ]
}}

Requirements:
- Return exactly 5 pillars (no more, no less)
- Each topic_name should be 2-6 words, concise and strategic
- Each description should be 1-2 sentences maximum, explaining the strategic focus
- Be specific and strategic based on the actual content, not generic
- Output ONLY valid JSON, no markdown, no code blocks, no explanatory text

Tweet corpus:
{corpus}
"""
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a strategic content analyst. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        response_format={"type": "json_object"}
    )
    
    return response.choices[0].message.content

@st.cache_data
def analyze_brand_compatibility_cached(
    brand: str,
    corpus_hash: str,
    corpus: str,
    model: str,
    temperature: float,
    exclude_replies: bool,
    max_tweets: int,
    api_key: str
) -> str:
    """Cached function to analyze brand compatibility using OpenAI API."""
    # The cache key includes brand, corpus_hash, model, temperature, exclude_replies, max_tweets, and api_key
    return analyze_brand_compatibility_openai(brand, corpus, model, temperature, api_key)

def analyze_brand_compatibility_openai(brand: str, corpus: str, model: str, temperature: float, api_key: str) -> str:
    """Call OpenAI API to analyze brand compatibility."""
    # API key should already be validated before calling this function
    client = OpenAI(api_key=api_key)
    
    prompt = f"""Analyze the compatibility between the brand "{brand}" and the following collection of tweets from a social media account.

The output must be STRICT JSON with this exact schema:
{{
  "score": <integer 0-100>,
  "reasoning": "<single paragraph explaining the strategic compatibility>",
  "recommendations": ["<optional recommendation 1>", "<optional recommendation 2>", "<optional recommendation 3>"]
}}

Requirements:
- score: An integer between 0 and 100 representing compatibility (0 = no fit, 100 = perfect fit)
- reasoning: A single strategic paragraph (2-4 sentences) explaining why this score was assigned, grounded in specific examples from the tweet corpus
- recommendations: (OPTIONAL, only include if score < 70) An array of 2-3 actionable recommendations for how the account could better align with this brand. If score >= 70, omit this field or use an empty array.

Be specific and strategic based on the actual content, not generic. Reference actual themes, topics, or patterns from the tweets.
Output ONLY valid JSON, no markdown, no code blocks, no explanatory text.

Brand: {brand}
Tweet corpus:
{corpus}
"""
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a brand strategy analyst. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        response_format={"type": "json_object"}
    )
    
    return response.choices[0].message.content

def scrape_article(url: str) -> tuple[bool, str, str, str]:
    """
    Scrape article from URL using newspaper3k.
    Returns (success, title, text, error_message)
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        title = article.title or ""
        text = article.text or ""
        
        if not text or len(text.strip()) < 50:
            return False, "", "", "Article text is too short or empty. The article may be behind a paywall or not accessible."
        
        return True, title, text, ""
    
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "Not Found" in error_msg:
            return False, "", "", "Article not found (404). Please check the URL."
        elif "403" in error_msg or "Forbidden" in error_msg:
            return False, "", "", "Access forbidden (403). The article may be behind a paywall or require authentication."
        elif "timeout" in error_msg.lower() or "connection" in error_msg.lower():
            return False, "", "", "Connection error. Please check your internet connection and try again."
        else:
            return False, "", "", f"Failed to scrape article: {error_msg}"

def hash_article_text(text: str) -> str:
    """Create a hash of the article text for caching."""
    return hashlib.md5(text.encode()).hexdigest()

@st.cache_data
def generate_reactive_tweet_cached(
    url: str,
    article_text_hash: str,
    article_title: str,
    article_text: str,
    tweet_corpus_hash: str,
    tweet_corpus: str,
    model: str,
    temperature: float,
    exclude_replies: bool,
    max_tweets: int,
    char_limit: int,
    api_key: str
) -> str:
    """Cached function to generate reactive tweet using OpenAI API."""
    return generate_reactive_tweet_openai(
        article_title, article_text, tweet_corpus, model, temperature, char_limit, api_key
    )

def generate_reactive_tweet_openai(
    article_title: str, article_text: str, tweet_corpus: str, model: str, temperature: float, char_limit: int, api_key: str
) -> str:
    """Call OpenAI API to generate a reactive tweet."""
    # API key should already be validated before calling this function
    client = OpenAI(api_key=api_key)
    
    # Analyze hashtag usage in user's tweets
    hashtag_count = tweet_corpus.count("#")
    uses_hashtags = hashtag_count > len(tweet_corpus.split("\n---\n")) * 0.1  # If >10% of tweets have hashtags
    
    # Limit corpus and article text to avoid token overflow
    tweet_corpus_limited = tweet_corpus[:3000]
    article_text_limited = article_text[:4000]
    
    hashtag_instruction = "Include hashtags if they fit naturally, as the user frequently uses them." if uses_hashtags else "Do NOT use hashtags unless absolutely necessary, as the user rarely uses them."
    
    prompt = f"""You are a social media content creator. Generate ONE reactive tweet in response to a news article, written in the specific voice and style of the user based on their tweet history.

The output must be STRICT JSON with this exact schema:
{{
  "tweet": "<final tweet text>"
}}

Requirements:
1. Character limit: Maximum {char_limit} characters (strictly enforced)
2. Voice matching: Match the tone, style, and writing patterns from the user's tweet corpus below
3. Hashtags: {hashtag_instruction}
4. Tone: Avoid generic "I'm excited" or overly enthusiastic language. Be sharp, authentic, and aligned with the user's style
5. Content: React to the article's key points, angle, or implications. Be insightful, not just summarizing
6. Style: Match sentence length, punctuation patterns, and voice from the user's tweets

User's tweet corpus (voice reference):
{tweet_corpus_limited}

News article:
Title: {article_title}

Text:
{article_text_limited}

Output ONLY valid JSON, no markdown, no code blocks, no explanatory text.
"""
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a social media content creator. Return only valid JSON with a tweet."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        response_format={"type": "json_object"}
    )
    
    return response.choices[0].message.content

# ---------- Sidebar: Upload ----------
st.sidebar.header("Upload Tweets CSV")
uploaded = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

df_ready = None

if uploaded is None:
    st.info("Upload a tweets CSV to begin.")
else:
    try:
        raw = pd.read_csv(uploaded)
        raw = normalize_columns(raw)
        missing = validate_columns(raw)

        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
            st.write("Your CSV columns are:")
            st.write(list(raw.columns))
        else:
            df_ready = prepare_tweets_df(raw)
            st.sidebar.success(f"Loaded {len(df_ready):,} tweets.")

    except Exception as e:
        st.error("Could not read this CSV. Make sure it is a valid comma-separated CSV file.")
        st.exception(e)

# Sidebar navigation
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio(
    "Select Page",
    [
        "The Leaderboard",
        "The Activity Heatmap",
        "Topic Modeler",
        "Brand Compatibility Agent",
        "The News Reactor"
    ],
    label_visibility="collapsed"
)

# Render selected page
if page == "The Leaderboard":
    if df_ready is None:
        st.write("Upload a CSV to see the leaderboard.")
    else:
        st.subheader("The Leaderboard")
        st.caption("Tweets ranked by engagement rate (Favorites / Views)")
        
        # Controls
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            min_views = st.slider(
                "Minimum Views",
                min_value=0,
                max_value=100000,
                value=1000,
                step=100
            )
        
        with col2:
            hide_replies = st.checkbox("Hide replies (@...)", value=True)
        
        with col3:
            top_n_option = st.selectbox(
                "Show top N",
                options=[25, 50, 100, 250, "All"],
                index=1  # Default to 50
            )
        
        with col4:
            search_keyword = st.text_input("Search keyword", value="", placeholder="Filter by text...")
        
        # Filter and rank
        df_filtered = df_ready.copy()
        
        # Apply minimum views filter
        df_filtered = df_filtered[df_filtered["views"] >= min_views]
        
        # Apply hide replies filter (strip leading spaces before checking)
        if hide_replies:
            df_filtered = df_filtered[~df_filtered["text"].str.strip().str.startswith("@")]
        
        # Apply text search filter (case-insensitive)
        if search_keyword:
            search_keyword_lower = search_keyword.lower()
            df_filtered = df_filtered[df_filtered["text"].str.lower().str.contains(search_keyword_lower, na=False)]
        
        # Sort by engagement_rate descending
        df_filtered = df_filtered.sort_values("engagement_rate", ascending=False)
        
        # Apply top N limit
        if top_n_option != "All":
            df_filtered = df_filtered.head(top_n_option)
        
        # Check if we have any tweets after filtering
        if len(df_filtered) == 0:
            st.info("No tweets match the current filters. Try adjusting your criteria.")
        else:
            # Prepare display dataframe
            df_display = df_filtered.copy()
            
            # Add rank column (1-indexed)
            df_display.insert(0, "Rank", range(1, len(df_display) + 1))
            
            # Format created_at to local-friendly readable format
            df_display["created_at"] = df_display["created_at"].dt.strftime("%Y-%m-%d %H:%M")
            
            # Format favorites and views as integers with commas
            df_display["favorites"] = df_display["favorites"].astype(int).apply(lambda x: f"{x:,}")
            df_display["views"] = df_display["views"].astype(int).apply(lambda x: f"{x:,}")
            
            # Format engagement_rate as percent with 2 decimals
            df_display["engagement_rate"] = (df_display["engagement_rate"] * 100).round(2)
            
            # Select and reorder columns
            df_display = df_display[[
                "Rank",
                "created_at",
                "text",
                "favorites",
                "views",
                "engagement_rate"
            ]]
            
            # Rename columns for display
            df_display.columns = [
                "Rank",
                "Created At",
                "Text",
                "Favorites",
                "Views",
                "Engagement Rate (%)"
            ]
            
            # Display the table
            st.dataframe(
                df_display,
                width="stretch",
                hide_index=True
            )


elif page == "The Activity Heatmap":
    if df_ready is None:
        st.write("Upload a CSV to see the heatmap.")
    else:
        st.subheader("The Activity Heatmap")
        st.caption("Posting frequency by day of week and hour of day")
        
        # Controls
        col1, col2 = st.columns(2)
        
        with col1:
            hide_replies_heatmap = st.checkbox("Hide replies (@...)", value=True, key="heatmap_hide_replies")
        
        with col2:
            # Date range selector
            if df_ready["created_at"].notna().any():
                min_date = df_ready["created_at"].min().date()
                max_date = df_ready["created_at"].max().date()
                date_range = st.date_input(
                    "Date range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key="heatmap_date_range"
                )
            else:
                date_range = None
                st.info("No valid dates found in the data.")
        
        # Filter data
        df_filtered_heatmap = df_ready.copy()
        
        # Apply hide replies filter (strip leading spaces before checking)
        if hide_replies_heatmap:
            df_filtered_heatmap = df_filtered_heatmap[~df_filtered_heatmap["text"].str.strip().str.startswith("@")]
        
        # Apply date range filter
        if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            if start_date and end_date:
                # Convert to datetime for comparison (keep UTC timezone)
                start_dt = pd.Timestamp(start_date, tz='UTC')
                end_dt = pd.Timestamp(end_date, tz='UTC') + pd.Timedelta(days=1)  # Include entire end date
                df_filtered_heatmap = df_filtered_heatmap[
                    (df_filtered_heatmap["created_at"] >= start_dt) & 
                    (df_filtered_heatmap["created_at"] < end_dt)
                ]
        
        # Check if we have any tweets after filtering
        if len(df_filtered_heatmap) == 0:
            st.info("No tweets match the current filters. Try adjusting your criteria.")
        else:
            # Create pivot table: rows = day, columns = hour, values = count
            pivot_data = df_filtered_heatmap.groupby(["day", "hour"]).size().reset_index(name="count")
            
            # Create pivot table
            pivot_table = pivot_data.pivot_table(
                index="day",
                columns="hour",
                values="count",
                fill_value=0
            )
            
            # Ensure all 24 hours exist as columns (0-23) and all 7 days as rows
            all_hours = list(range(24))
            
            # Reindex to include all days and hours, filling missing with 0
            pivot_table = pivot_table.reindex(
                index=DAY_ORDER,
                columns=all_hours,
                fill_value=0
            )
            
            # Convert to integer (no decimals)
            pivot_table = pivot_table.astype(int)
            
            # Create Plotly heatmap
            fig = go.Figure(data=go.Heatmap(
                z=pivot_table.values,
                x=[f"{h:02d}:00" for h in pivot_table.columns],  # Format hours as "00:00", "01:00", etc.
                y=pivot_table.index,
                colorscale='Viridis',
                text=pivot_table.values,
                texttemplate='%{text}',
                textfont={"size": 10, "color": "#333333"},
                hovertemplate='Day: %{y}<br>Hour: %{x}<br>Count: %{z}<extra></extra>',
                colorbar=dict(title="Tweet Count")
            ))
            
            fig.update_layout(
                title="Posting Frequency Heatmap",
                xaxis=dict(
                    title=dict(text="Hour of Day", font=dict(family="Arial", size=14, color="#333333")),
                    tickmode='linear',
                    tick0=0,
                    dtick=2,  # Show every 2 hours
                    tickfont=dict(family="Arial", size=12, color="#333333"),
                    gridcolor='#e0e0e0'
                ),
                yaxis=dict(
                    title=dict(text="Day of Week", font=dict(family="Arial", size=14, color="#333333")),
                    tickfont=dict(family="Arial", size=12, color="#333333"),
                    gridcolor='#e0e0e0'
                ),
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(color='#333333', family='Arial'),
                coloraxis_colorbar=dict(
                    title=dict(text="Tweet Count", font=dict(family="Arial", size=12, color="#333333")),
                    tickfont=dict(family="Arial", size=11, color="#333333")
                ),
                height=500,
                margin=dict(l=100, r=50, t=50, b=50)
            )
            
            # Display the heatmap
            st.plotly_chart(fig, width="stretch")

elif page == "Topic Modeler":
    if df_ready is None:
        st.write("Upload a CSV to see the topic modeler.")
    else:
        st.subheader("Topic Modeler")
        st.caption("AI-powered analysis of your tweet corpus to identify core content pillars")
        
        # Load and validate API key
        api_key = get_openai_api_key()
        is_valid, validation_error = validate_openai_api_key(api_key)
        
        # Get OS-level env var for comparison
        os_env_key = os.environ.get("OPENAI_API_KEY", "").strip()
        
        # API Diagnostics section
        with st.expander("API Diagnostics", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Key Status:**")
                st.write(f"- Loaded: {api_key is not None and len(api_key) > 0}")
                st.write(f"- Valid format: {is_valid}")
                st.write(f"- Length: {len(api_key) if api_key else 0} characters")
                
                if api_key:
                    # Show first 7 and last 4 characters safely
                    prefix = api_key[:7] if len(api_key) >= 7 else api_key[:len(api_key)]
                    suffix = api_key[-4:] if len(api_key) >= 4 else "N/A"
                    st.write(f"- Prefix: `{prefix}...`")
                    st.write(f"- Suffix: `...{suffix}`")
                else:
                    st.write("- Prefix: N/A")
                    st.write("- Suffix: N/A")
            
            with col2:
                st.write("**Environment:**")
                st.write(f"- OS env var exists: {os_env_key is not None and len(os_env_key) > 0}")
                if os_env_key:
                    os_prefix = os_env_key[:7] if len(os_env_key) >= 7 else os_env_key[:len(os_env_key)]
                    os_suffix = os_env_key[-4:] if len(os_env_key) >= 4 else "N/A"
                    st.write(f"- OS prefix: `{os_prefix}...`")
                    st.write(f"- OS suffix: `...{os_suffix}`")
                
                # Test API key button
                if api_key and is_valid:
                    if st.button("Test API Key", key="test_api_key"):
                        with st.spinner("Testing API key with gpt-4o-mini..."):
                            # Use a default model for testing (lightweight)
                            success, message = test_openai_api_key(api_key, "gpt-4o-mini")
                            if success:
                                st.success(f"✅ {message}")
                            else:
                                st.error(f"❌ Test failed: {message}")
                                if st.session_state.get("topic_debug", False):
                                    st.code(message)
                else:
                    st.info("Fix API key issues above to enable testing.")
        
        # Show validation error if key is invalid
        if not is_valid:
            st.error(f"⚠️ **API Key Validation Failed:** {validation_error}")
            st.error("Please ensure your `.env` file in the same folder as `app.py` contains a valid OpenAI API key.")
            st.info("**How to fix:**\n1. Open the `.env` file in the same folder as `app.py`\n2. Add or update: `OPENAI_API_KEY=sk-proj-...`\n3. Restart the Streamlit app")
            st.stop()
        
        # Key is valid, continue with Tab 3
        # AI Settings section
        with st.expander("AI Settings", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                model_name = st.text_input(
                    "Model name",
                    value="gpt-4o-mini",
                    help="OpenAI model to use (e.g., gpt-4o-mini, gpt-4o, gpt-4)",
                    key="model_name_input"
                )
                # Store for test button
                st.session_state["model_name_test"] = model_name
                
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=0.4,
                    step=0.1,
                    help="Controls randomness (0 = deterministic, 2 = very creative)"
                )
            
            with col2:
                exclude_replies_topic = st.checkbox(
                    "Exclude replies (@...)",
                    value=True,
                    key="topic_exclude_replies"
                )
                max_tweets = st.slider(
                    "Max tweets to analyze",
                    min_value=10,
                    max_value=1000,
                    value=300,
                    step=10,
                    help="If you have more tweets, we'll sample deterministically"
                )
        
        # Debug option
        show_debug = st.checkbox("Show debug info", value=False, key="topic_debug")
        
        # Generate button
        if st.button("Generate Core Pillars", type="primary", width="stretch"):
            # Prepare corpus
            corpus = prepare_tweet_corpus(df_ready, exclude_replies_topic, max_tweets)
            
            if not corpus or len(corpus.strip()) == 0:
                st.warning("No tweets available after filtering. Try adjusting your settings.")
            else:
                # Create hash for caching
                corpus_hash = hash_tweet_corpus(corpus)
                
                # Show info about corpus
                tweet_count = len(corpus.split("\n---\n"))
                st.info(f"Analyzing {tweet_count:,} unique tweets...")
                
                try:
                    with st.spinner("Calling OpenAI API to analyze topics..."):
                        # Call cached function (api_key is validated above)
                        raw_response = analyze_topics_cached(
                            corpus_hash,
                            corpus,
                            model_name,
                            temperature,
                            exclude_replies_topic,
                            max_tweets,
                            api_key
                        )
                    
                    if show_debug:
                        st.text_area("Raw API Response", raw_response, height=200, key="debug_raw")
                    
                    # Parse JSON
                    try:
                        result = json.loads(raw_response)
                        
                        # Validate structure
                        if "pillars" not in result:
                            raise ValueError("Response missing 'pillars' key")
                        
                        pillars = result["pillars"]
                        
                        if not isinstance(pillars, list):
                            raise ValueError("'pillars' must be a list")
                        
                        if len(pillars) != 5:
                            st.warning(f"Expected 5 pillars, got {len(pillars)}. Showing what we received.")
                        
                        # Validate each pillar
                        valid_pillars = []
                        for i, pillar in enumerate(pillars):
                            if not isinstance(pillar, dict):
                                if show_debug:
                                    st.error(f"Pillar {i+1} is not a dictionary: {pillar}")
                                continue
                            
                            if "topic_name" not in pillar or "description" not in pillar:
                                if show_debug:
                                    st.error(f"Pillar {i+1} missing required fields: {pillar}")
                                continue
                            
                            valid_pillars.append({
                                "Topic Name": str(pillar["topic_name"]).strip(),
                                "Description": str(pillar["description"]).strip()
                            })
                        
                        if len(valid_pillars) == 0:
                            st.error("No valid pillars found in the response. Please try again or check debug info.")
                        else:
                            # Create and display dataframe
                            df_pillars = pd.DataFrame(valid_pillars)
                            st.dataframe(
                                df_pillars,
                                width="stretch",
                                hide_index=True
                            )
                            
                            st.success(f"✅ Identified {len(valid_pillars)} core pillar(s)")
                        
                    except json.JSONDecodeError as e:
                        st.error("❌ Failed to parse JSON response from OpenAI.")
                        if show_debug:
                            st.error(f"JSON Error: {str(e)}")
                            st.text_area("Raw Response (for debugging)", raw_response, height=200)
                        else:
                            st.info("Enable 'Show debug info' to see the raw response.")
                    
                    except ValueError as e:
                        st.error(f"❌ Invalid response format: {str(e)}")
                        if show_debug:
                            st.text_area("Raw Response (for debugging)", raw_response, height=200)
                
                except Exception as e:
                    error_msg = str(e)
                    st.error(f"❌ Error calling OpenAI API: {error_msg}")
                    if show_debug:
                        st.exception(e)
                    else:
                        st.info("Enable 'Show debug info' to see detailed error information.")

elif page == "Brand Compatibility Agent":
    if df_ready is None:
        st.write("Upload a CSV to see the brand compatibility agent.")
    else:
        st.subheader("Brand Compatibility Agent")
        st.caption("AI-powered analysis of brand compatibility with your tweet corpus")
        
        # Load and validate API key
        api_key = get_openai_api_key()
        is_valid, validation_error = validate_openai_api_key(api_key)
        
        # Show validation error if key is invalid
        if not is_valid:
            st.error(f"⚠️ **API Key Validation Failed:** {validation_error}")
            st.error("Please ensure your `.env` file in the same folder as `app.py` contains a valid OpenAI API key.")
            st.info("**How to fix:**\n1. Open the `.env` file in the same folder as `app.py`\n2. Add or update: `OPENAI_API_KEY=sk-proj-...`\n3. Restart the Streamlit app")
            st.stop()
        
        # Brand input
        brand_name = st.text_input(
            "Brand Name",
            value="",
            placeholder="Nike, Yale University, etc.",
            key="brand_name_input"
        )
        
        # AI Settings section
        with st.expander("AI Settings", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                model_name_brand = st.text_input(
                    "Model name",
                    value="gpt-4o-mini",
                    help="OpenAI model to use (e.g., gpt-4o-mini, gpt-4o, gpt-4)",
                    key="brand_model_name"
                )
                temperature_brand = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=0.4,
                    step=0.1,
                    help="Controls randomness (0 = deterministic, 2 = very creative)",
                    key="brand_temperature"
                )
            
            with col2:
                exclude_replies_brand = st.checkbox(
                    "Exclude replies (@...)",
                    value=True,
                    key="brand_exclude_replies"
                )
                max_tweets_brand = st.slider(
                    "Max tweets to analyze",
                    min_value=10,
                    max_value=1000,
                    value=300,
                    step=10,
                    help="If you have more tweets, we'll sample deterministically",
                    key="brand_max_tweets"
                )
        
        # Debug option
        show_debug_brand = st.checkbox("Show debug info", value=False, key="brand_debug")
        
        # Analyze button
        if st.button("Analyze Compatibility", type="primary", use_container_width=True):
            # Validate brand name
            if not brand_name or not brand_name.strip():
                st.warning("⚠️ Please enter a brand name to analyze.")
            else:
                brand_name_clean = brand_name.strip()
                
                # Prepare corpus
                corpus = prepare_tweet_corpus(df_ready, exclude_replies_brand, max_tweets_brand)
                
                if not corpus or len(corpus.strip()) == 0:
                    st.info("No tweets available after filtering. Try adjusting your settings.")
                else:
                    # Create hash for caching
                    corpus_hash = hash_tweet_corpus(corpus)
                    
                    # Show info about corpus
                    tweet_count = len(corpus.split("\n---\n"))
                    st.info(f"Analyzing compatibility with {tweet_count:,} unique tweets...")
                    
                    try:
                        with st.spinner("Calling OpenAI API to analyze brand compatibility..."):
                            # Call cached function (api_key is validated above)
                            raw_response = analyze_brand_compatibility_cached(
                                brand_name_clean,
                                corpus_hash,
                                corpus,
                                model_name_brand,
                                temperature_brand,
                                exclude_replies_brand,
                                max_tweets_brand,
                                api_key
                            )
                        
                        if show_debug_brand:
                            st.text_area("Raw API Response", raw_response, height=200, key="debug_raw_brand")
                        
                        # Parse JSON
                        try:
                            result = json.loads(raw_response)
                            
                            # Validate structure
                            if "score" not in result:
                                raise ValueError("Response missing 'score' key")
                            if "reasoning" not in result:
                                raise ValueError("Response missing 'reasoning' key")
                            
                            # Extract and validate score
                            score = int(result["score"])
                            score = max(0, min(100, score))  # Clamp to [0, 100]
                            
                            reasoning = str(result["reasoning"]).strip()
                            
                            # Get recommendations if present and score < 70
                            recommendations = result.get("recommendations", [])
                            if not isinstance(recommendations, list):
                                recommendations = []
                            
                            # Display results
                            st.markdown("---")
                            
                            # Score as metric
                            st.metric(
                                label="Compatibility Score",
                                value=f"{score}%",
                                help="Score from 0 (no fit) to 100 (perfect fit)"
                            )
                            
                            st.markdown("---")
                            
                            # Reasoning
                            st.write("**Strategic Reasoning:**")
                            st.write(reasoning)
                            
                            # Recommendations (only if score < 70)
                            if score < 70 and recommendations and len(recommendations) > 0:
                                st.markdown("---")
                                st.write("**Recommendations to Improve Fit:**")
                                for i, rec in enumerate(recommendations, 1):
                                    if rec and str(rec).strip():
                                        st.write(f"• {str(rec).strip()}")
                            
                            st.success(f"✅ Analysis complete for {brand_name_clean}")
                            
                        except json.JSONDecodeError as e:
                            st.error("❌ Failed to parse JSON response from OpenAI.")
                            if show_debug_brand:
                                st.error(f"JSON Error: {str(e)}")
                                st.text_area("Raw Response (for debugging)", raw_response, height=200, key="debug_raw_brand_error")
                            else:
                                st.info("Enable 'Show debug info' to see the raw response.")
                        
                        except ValueError as e:
                            st.error(f"❌ Invalid response format: {str(e)}")
                            if show_debug_brand:
                                st.text_area("Raw Response (for debugging)", raw_response, height=200, key="debug_raw_brand_error2")
                    
                    except Exception as e:
                        error_msg = str(e)
                        st.error(f"❌ Error calling OpenAI API: {error_msg}")
                        if show_debug_brand:
                            st.exception(e)
                        else:
                            st.info("Enable 'Show debug info' to see detailed error information.")

elif page == "The News Reactor":
    if df_ready is None:
        st.write("Upload a CSV to see the news reactor.")
    else:
        st.subheader("The News Reactor")
        st.caption("Generate reactive tweets in your voice based on news articles")
        
        # Load and validate API key
        api_key = get_openai_api_key()
        is_valid, validation_error = validate_openai_api_key(api_key)
        
        # Show validation error if key is invalid
        if not is_valid:
            st.error(f"⚠️ **API Key Validation Failed:** {validation_error}")
            st.error("Please ensure your `.env` file in the same folder as `app.py` contains a valid OpenAI API key.")
            st.info("**How to fix:**\n1. Open the `.env` file in the same folder as `app.py`\n2. Add or update: `OPENAI_API_KEY=sk-proj-...`\n3. Restart the Streamlit app")
            st.stop()
        
        # URL input
        article_url = st.text_input(
            "News Article URL",
            value="",
            placeholder="https://example.com/article",
            key="article_url_input"
        )
        
        # AI Settings section
        with st.expander("AI Settings", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                model_name_news = st.text_input(
                    "Model name",
                    value="gpt-4o-mini",
                    help="OpenAI model to use (e.g., gpt-4o-mini, gpt-4o, gpt-4)",
                    key="news_model_name"
                )
                temperature_news = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=0.7,
                    step=0.1,
                    help="Controls randomness (0 = deterministic, 2 = very creative)",
                    key="news_temperature"
                )
            
            with col2:
                exclude_replies_news = st.checkbox(
                    "Exclude replies (@...)",
                    value=True,
                    key="news_exclude_replies"
                )
                max_tweets_news = st.slider(
                    "Max tweets for voice",
                    min_value=10,
                    max_value=1000,
                    value=300,
                    step=10,
                    help="If you have more tweets, we'll sample deterministically",
                    key="news_max_tweets"
                )
                char_limit = st.slider(
                    "Character limit",
                    min_value=100,
                    max_value=500,
                    value=280,
                    step=10,
                    help="Maximum characters for the generated tweet",
                    key="news_char_limit"
                )
        
        # Debug option
        show_debug_news = st.checkbox("Show debug info", value=False, key="news_debug")
        
        # Generate button
        if st.button("Generate Reactive Tweet", type="primary", use_container_width=True):
            # Validate URL
            if not article_url or not article_url.strip():
                st.warning("⚠️ Please enter a news article URL.")
            else:
                article_url_clean = article_url.strip()
                
                # Scrape article
                with st.spinner("Scraping article..."):
                    success, title, text, error_msg = scrape_article(article_url_clean)
                
                if not success:
                    st.error(f"❌ Failed to scrape article: {error_msg}")
                    st.info("**Tips:**\n- Make sure the URL is accessible\n- Some articles may be behind paywalls\n- Try a different article URL")
                elif not text or len(text.strip()) < 50:
                    st.error("❌ Article text is too short or empty. Please try a different article.")
                else:
                    # Show article preview
                    with st.expander("Article Preview", expanded=False):
                        st.write(f"**Title:** {title}")
                        preview_text = text[:500] + "..." if len(text) > 500 else text
                        st.write(f"**Preview:** {preview_text}")
                    
                    # Prepare tweet corpus for voice reference
                    tweet_corpus = prepare_tweet_corpus(df_ready, exclude_replies_news, max_tweets_news)
                    
                    if not tweet_corpus or len(tweet_corpus.strip()) == 0:
                        st.info("No tweets available after filtering. Try adjusting your settings.")
                    else:
                        # Create hashes for caching
                        article_text_hash = hash_article_text(text)
                        tweet_corpus_hash = hash_tweet_corpus(tweet_corpus)
                        
                        # Show info
                        tweet_count = len(tweet_corpus.split("\n---\n"))
                        st.info(f"Generating tweet based on {tweet_count:,} unique tweets and article content...")
                        
                        try:
                            with st.spinner("Generating reactive tweet in your voice..."):
                                # Call cached function
                                raw_response = generate_reactive_tweet_cached(
                                    article_url_clean,
                                    article_text_hash,
                                    title,
                                    text,
                                    tweet_corpus_hash,
                                    tweet_corpus,
                                    model_name_news,
                                    temperature_news,
                                    exclude_replies_news,
                                    max_tweets_news,
                                    char_limit,
                                    api_key
                                )
                            
                            if show_debug_news:
                                st.text_area("Raw API Response", raw_response, height=200, key="debug_raw_news")
                            
                            # Parse JSON
                            try:
                                result = json.loads(raw_response)
                                
                                # Validate structure
                                if "tweet" not in result:
                                    raise ValueError("Response missing 'tweet' key")
                                
                                tweet_text = str(result["tweet"]).strip()
                                
                                # Validate character limit
                                if len(tweet_text) > char_limit:
                                    st.warning(f"⚠️ Generated tweet ({len(tweet_text)} chars) exceeds limit ({char_limit} chars). Truncating...")
                                    tweet_text = tweet_text[:char_limit]
                                
                                # Display tweet card
                                st.markdown("---")
                                st.markdown("### Generated Tweet")
                                
                                # CSS for tweet card
                                st.markdown("""
                                <style>
                                .tweet-card {
                                    border: 1px solid #e1e8ed;
                                    border-radius: 12px;
                                    padding: 16px;
                                    background-color: #ffffff;
                                    margin: 16px 0;
                                    box-shadow: 0 1px 3px rgba(0,0,0,0.12);
                                }
                                .tweet-header {
                                    display: flex;
                                    align-items: center;
                                    margin-bottom: 12px;
                                }
                                .tweet-avatar {
                                    width: 48px;
                                    height: 48px;
                                    border-radius: 50%;
                                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                    display: flex;
                                    align-items: center;
                                    justify-content: center;
                                    color: white;
                                    font-weight: bold;
                                    font-size: 18px;
                                    margin-right: 12px;
                                }
                                .tweet-name {
                                    font-weight: bold;
                                    font-size: 15px;
                                    color: #14171a;
                                }
                                .tweet-text {
                                    font-size: 15px;
                                    line-height: 1.5;
                                    color: #14171a;
                                    white-space: pre-wrap;
                                    word-wrap: break-word;
                                }
                                </style>
                                """, unsafe_allow_html=True)
                                
                                # Escape HTML in tweet text for safety
                                import html
                                tweet_text_escaped = html.escape(tweet_text)
                                
                                # Tweet card HTML
                                tweet_card_html = f"""
                                <div class="tweet-card">
                                    <div class="tweet-header">
                                        <div class="tweet-avatar">YN</div>
                                        <div class="tweet-name">Your Name</div>
                                    </div>
                                    <div class="tweet-text">{tweet_text_escaped}</div>
                                </div>
                                """
                                st.markdown(tweet_card_html, unsafe_allow_html=True)
                                
                                # Character count
                                st.caption(f"Character count: {len(tweet_text)}/{char_limit}")
                                
                                st.success(f"✅ Tweet generated successfully!")
                                
                            except json.JSONDecodeError as e:
                                st.error("❌ Failed to parse JSON response from OpenAI.")
                                if show_debug_news:
                                    st.error(f"JSON Error: {str(e)}")
                                    st.text_area("Raw Response (for debugging)", raw_response, height=200, key="debug_raw_news_error")
                                else:
                                    st.info("Enable 'Show debug info' to see the raw response.")
                            
                            except ValueError as e:
                                st.error(f"❌ Invalid response format: {str(e)}")
                                if show_debug_news:
                                    st.text_area("Raw Response (for debugging)", raw_response, height=200, key="debug_raw_news_error2")
                        
                        except Exception as e:
                            error_msg = str(e)
                            st.error(f"❌ Error calling OpenAI API: {error_msg}")
                            if show_debug_news:
                                st.exception(e)
                            else:
                                st.info("Enable 'Show debug info' to see detailed error information.")
