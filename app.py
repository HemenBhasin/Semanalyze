import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json
from semantic_analyzer import analyzer
from config import SENTENCE_TRANSFORMER_MODEL, SENTIMENT_MODEL
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Semantic Product Review Analyzer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern dark theme
st.markdown("""
    <style>
    * {
        margin: 0;
        padding: 0;
    }
    
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1428 100%);
        color: #e0e0e0;
    }
    
    [data-testid="stHeader"] {
        background: transparent;
        border-bottom: 1px solid rgba(100, 200, 255, 0.1);
    }
    
    .main {
        max-width: 1400px;
        padding: 2rem;
    }
    
    /* Input styling */
    .stTextArea > div > div > textarea {
        min-height: 200px;
        background-color: rgba(20, 30, 60, 0.8) !important;
        border: 1px solid rgba(100, 200, 255, 0.3) !important;
        color: #e0e0e0 !important;
        border-radius: 12px !important;
        font-size: 15px !important;
        transition: all 0.3s ease;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: rgba(100, 200, 255, 0.8) !important;
        box-shadow: 0 0 20px rgba(100, 200, 255, 0.2) !important;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 16px;
        background: linear-gradient(135deg, #00d4ff 0%, #0099ff 100%);
        color: #000;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0, 212, 255, 0.5);
    }
    
    /* Sentiment colors */
    .sentiment-positive {
        color: #00ff88;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
    }
    
    .sentiment-negative {
        color: #ff3366;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(255, 51, 102, 0.5);
    }
    
    .sentiment-neutral {
        color: #ffaa00;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(255, 170, 0, 0.5);
    }
    
    /* Aspect card styling */
    .aspect-card {
        background: linear-gradient(135deg, rgba(20, 50, 100, 0.6) 0%, rgba(30, 60, 120, 0.4) 100%);
        border-left: 4px solid #00d4ff;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(100, 200, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .aspect-card:hover {
        border-color: rgba(100, 200, 255, 0.5);
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.15);
        transform: translateX(4px);
    }
    
    .aspect-card h4 {
        color: #00d4ff;
        margin-bottom: 0.5rem;
        font-size: 18px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .aspect-card p {
        color: #b0b0b0;
        margin: 0.5rem 0;
        font-size: 14px;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    /* Main title - Improved visibility */
    h1 {
        font-size: 2.8rem;
        font-weight: 800;
        color: #ffffff !important;
        text-shadow: 
            0 0 8px rgba(0, 180, 255, 0.9),
            0 0 16px rgba(0, 120, 255, 0.6);
        margin: 0.5rem 0;
        letter-spacing: -0.5px;
        line-height: 1.3;
        position: relative;
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 12px;
        background: rgba(10, 14, 39, 0.4);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(100, 200, 255, 0.15);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Add a subtle glow effect on hover */
    h1:hover {
        text-shadow: 
            0 0 12px rgba(0, 200, 255, 1),
            0 0 24px rgba(0, 150, 255, 0.8);
        box-shadow: 0 6px 25px rgba(0, 150, 255, 0.3);
    }
    
    h2 {
        font-size: 1.8rem;
        color: #00d4ff;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        font-size: 1.3rem;
        color: #00ff88;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(20, 50, 100, 0.4) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(100, 200, 255, 0.2) !important;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: rgba(20, 50, 100, 0.6) !important;
        border-color: rgba(100, 200, 255, 0.4) !important;
    }
    
    /* Metric styling */
    .metric-box {
        background: linear-gradient(135deg, rgba(20, 50, 100, 0.6) 0%, rgba(30, 60, 120, 0.4) 100%);
        border: 1px solid rgba(100, 200, 255, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4ff 0%, #00ff88 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        color: #b0b0b0;
        font-size: 14px;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(100, 200, 255, 0.3), transparent);
        margin: 2rem 0;
    }
    
    /* Warning/Info boxes */
    .stWarning, .stInfo {
        background-color: rgba(255, 170, 0, 0.1) !important;
        border-left: 4px solid #ffaa00 !important;
        border-radius: 8px !important;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(20, 30, 60, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(0, 212, 255, 0.5);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(0, 212, 255, 0.8);
    }
    </style>
""", unsafe_allow_html=True)

def display_sentiment_emoji(sentiment):
    """Return emoji based on sentiment."""
    if sentiment == "positive":
        return "✨"
    elif sentiment == "negative":
        return "⚠️"
    return "◆"

def get_sentiment_color(sentiment):
    """Return color based on sentiment."""
    if sentiment == "positive":
        return "#00ff88"  # Vibrant green
    elif sentiment == "negative":
        return "#ff3366"  # Vibrant red
    return "#ffaa00"  # Vibrant orange

def create_wordcloud(text):
    """Generate a word cloud from text."""
    if not text.strip():
        return None
        
    # Set the background color to transparent
    plt.rcParams['savefig.facecolor'] = 'none'
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color=None,  # Set to None for transparent background
        mode='RGBA',  # Use RGBA mode for transparency
        max_words=100,
        colormap='cool',  # Using cool colormap for better visibility
        prefer_horizontal=0.7,
        max_font_size=150,
        min_font_size=10,
        contour_width=0
    ).generate(text)
    
    # Create figure with transparent background
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_alpha(0)
    
    # Display the generated word cloud
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    # Adjust layout to remove any extra whitespace
    plt.tight_layout(pad=0)
    
    return fig

def create_sentiment_gauge(score, sentiment):
    """Create an enhanced sentiment gauge chart."""
    if sentiment == 'negative':
        score = 1 - score
    
    fig = go.Figure(data=[
        go.Indicator(
            mode="gauge+number+delta",
            value=score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Sentiment Score", 'font': {'size': 20, 'color': '#00d4ff'}},
            number={'font': {'size': 32, 'color': get_sentiment_color(sentiment)}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': 'rgba(100, 200, 255, 0.3)'},
                'bar': {'color': get_sentiment_color(sentiment)},
                'bgcolor': 'rgba(20, 50, 100, 0.3)',
                'borderwidth': 2,
                'bordercolor': 'rgba(100, 200, 255, 0.5)',
                'steps': [
                    {'range': [0, 33], 'color': 'rgba(255, 51, 102, 0.2)'},
                    {'range': [33, 66], 'color': 'rgba(255, 170, 0, 0.2)'},
                    {'range': [66, 100], 'color': 'rgba(0, 255, 136, 0.2)'}
                ],
                'threshold': {
                    'line': {'color': '#00d4ff', 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        )
    ])
    
    fig.update_layout(
        paper_bgcolor='rgba(10, 14, 39, 0)',
        plot_bgcolor='rgba(10, 14, 39, 0)',
        font={'color': '#e0e0e0', 'family': 'Arial, sans-serif'},
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def display_aspect_analysis(aspects):
    """Display aspect-based analysis results."""
    if not aspects:
        st.warning("No specific aspects detected in the review.")
        return
    
    st.subheader("🔍 Aspect-Based Analysis")
    
    # Create columns for layout
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### Detected Aspects")
        for idx, aspect in enumerate(aspects, 1):
            sentiment = aspect['sentiment']['label']
            score = aspect['sentiment']['score']
            
            st.markdown(f"""
            <div class="aspect-card">
                <h4>#{idx} {aspect['aspect'].title()}</h4>
                <p><strong>Mentions:</strong> {', '.join([f'<code>{x}</code>' for x in aspect['mentions']])}</p>
                <p><strong>Sentiment:</strong> <span class="sentiment-{sentiment}">
                    {sentiment.upper()} ({score:.2%}) {display_sentiment_emoji(sentiment)}
                </span></p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Sentiment Distribution")
        # Sentiment distribution pie chart
        if aspects:
            sentiment_counts = {}
            for aspect in aspects:
                sentiment = aspect['sentiment']['label']
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            
            if sentiment_counts:
                df_sentiment = pd.DataFrame({
                    'Sentiment': list(sentiment_counts.keys()),
                    'Count': list(sentiment_counts.values())
                })
                
                fig = px.pie(
                    df_sentiment,
                    values='Count',
                    names='Sentiment',
                    color='Sentiment',
                    color_discrete_map={
                        'positive': '#00ff88',
                        'neutral': '#ffaa00',
                        'negative': '#ff3366'
                    }
                )
                
                fig.update_layout(
                    paper_bgcolor='rgba(10, 14, 39, 0)',
                    plot_bgcolor='rgba(10, 14, 39, 0)',
                    font={'color': '#e0e0e0'},
                    showlegend=True,
                    height=350,
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                
                fig.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    textfont={'size': 12, 'color': '#000'}
                )
                
                st.plotly_chart(fig, use_container_width=True)

def main():
    """Main Streamlit app function."""
    # Header section
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("✨ Semantic Product Review Analyzer")
        st.markdown("*Advanced NLP-powered sentiment analysis with aspect extraction*")
    
    st.markdown("---")
    
    # Model info in expander
    with st.expander("ℹ️ About This Tool", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 🤖 Technology Stack
            This tool leverages state-of-the-art NLP models to provide deep insights into product reviews.
            
            **Models Used:**
            - **Embeddings:** `all-MiniLM-L6-v2` 
            - **Sentiment:** `distilbert-base-uncased-finetuned-sst-2-english` 
            - **NLP:** `spaCy` with custom aspect extraction
            """)
        with col2:
            st.markdown("""
            ### 🔄 Analysis Pipeline
            1. **Text Preprocessing** - Cleaning & normalization
            2. **Aspect Extraction** - Identifying key product aspects
            3. **Sentiment Analysis** - Fine-tuned BERT classification
            4. **Visualization** - Interactive charts & insights
            """)
    
    st.markdown("---")
    
    # Input section
    st.header("📝 Enter a Product Review")
    review_text = st.text_area(
        "Paste your product review here...",
        height=150,
        placeholder="Example: The camera quality is amazing but the battery drains quickly. The design feels premium though.",
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        analyze_btn = st.button("🚀 Analyze Review", use_container_width=True)
    with col2:
        clear_btn = st.button("🔄 Clear", use_container_width=True)
    
    if clear_btn:
        st.rerun()
    
    if analyze_btn and review_text.strip():
        with st.spinner("🔬 Analyzing your review..."):
            # Analyze the review
            analysis_result = analyzer.analyze_review(review_text)
            
            # Display results
            st.markdown("---")
            st.header("📊 Analysis Results")
            
            # Overall sentiment section
            overall_sentiment = analysis_result['overall_sentiment']
            sentiment_emoji = display_sentiment_emoji(overall_sentiment['label'])
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(f"Overall Sentiment: {sentiment_emoji} {overall_sentiment['label'].upper()}")
                st.markdown(f"**Confidence:** {overall_sentiment['score']:.2%}")
            
            with col2:
                # Display metrics
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{overall_sentiment['score']:.0%}</div>
                    <div class="metric-label">Confidence Score</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Sentiment gauge
            st.markdown("### Sentiment Gauge")
            fig_gauge = create_sentiment_gauge(overall_sentiment['score'], overall_sentiment['label'])
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.markdown("---")
            
            # Display aspect analysis
            display_aspect_analysis(analysis_result.get('aspects', []))
            
            st.markdown("---")
            
            # Word cloud section
            st.subheader("📊 Word Cloud Visualization")
            wordcloud_fig = create_wordcloud(review_text)
            if wordcloud_fig:
                st.pyplot(wordcloud_fig, use_container_width=True)
            
            st.markdown("---")
            
            # Raw JSON output (collapsible)
            with st.expander("🔧 View Raw Analysis Data", expanded=False):
                st.json(analysis_result)
    
    elif analyze_btn and not review_text.strip():
        st.warning("⚠️ Please enter a review to analyze.")

if __name__ == "__main__":
    main()
