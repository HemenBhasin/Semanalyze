import streamlit as st
import pandas as pd
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
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        max-width: 1200px;
        padding: 2rem;
    }
    .stTextArea > div > div > textarea {
        min-height: 200px;
    }
    .stButton>button {
        width: 100%;
        padding: 0.5rem;
        font-weight: bold;
    }
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #ffc107;
        font-weight: bold;
    }
    .aspect-card {
        border-left: 4px solid #4a90e2;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f8f9fa;
        border-radius: 0.25rem;
    }
    </style>
""", unsafe_allow_html=True)

def display_sentiment_emoji(sentiment):
    """Return emoji based on sentiment."""
    if sentiment == "positive":
        return "😊"
    elif sentiment == "negative":
        return "😞"
    return "😐"

def get_sentiment_color(sentiment):
    """Return color based on sentiment."""
    if sentiment == "positive":
        return "#28a745"  # Green
    elif sentiment == "negative":
        return "#dc3545"  # Red
    return "#ffc107"  # Yellow

def create_wordcloud(text):
    """Generate a word cloud from text."""
    if not text.strip():
        return None
        
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=100,
        colormap='viridis'
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def display_aspect_analysis(aspects):
    """Display aspect-based analysis results."""
    if not aspects:
        st.warning("No specific aspects detected in the review.")
        return
    
    st.subheader("🔍 Aspect-based Analysis")
    
    # Create columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        for aspect in aspects:
            sentiment = aspect['sentiment']['label']
            score = aspect['sentiment']['score']
            
            st.markdown(f"""
            <div class="aspect-card">
                <h4>{aspect['aspect'].title()}</h4>
                <p>Mentions: {', '.join([f'"{x}"' for x in aspect['mentions']])}</p>
                <p>Sentiment: <span class="sentiment-{sentiment}">
                    {sentiment.title()} ({score:.2f}) {display_sentiment_emoji(sentiment)}
                </span></p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
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
                    title='Aspect Sentiment Distribution',
                    color='Sentiment',
                    color_discrete_map={
                        'positive': '#28a745',
                        'neutral': '#ffc107',
                        'negative': '#dc3545'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)

def main():
    """Main Streamlit app function."""
    st.title("🔍 Semantic Product Review Analyzer")
    st.markdown("---")
    
    # Model info
    with st.expander("ℹ️ About this tool"):
        st.markdown("""
        This tool analyzes product reviews to extract key aspects and determine sentiment using state-of-the-art NLP models.
        
        **Models Used:**
        - Sentence Embeddings: `all-MiniLM-L6-v2`
        - Sentiment Analysis: `distilbert-base-uncased-finetuned-sst-2-english`
        
        **How it works:**
        1. The system extracts key aspects (like camera, battery, etc.) from your review
        2. It analyzes the sentiment for each aspect
        3. Results are displayed with visualizations
        """)
    
    # Input section
    st.header("📝 Enter a Product Review")
    review_text = st.text_area(
        "Paste your product review here...",
        height=150,
        placeholder="The camera quality is amazing but the battery drains quickly..."
    )
    
    analyze_btn = st.button("Analyze Review")
    
    if analyze_btn and review_text.strip():
        with st.spinner("Analyzing your review..."):
            # Analyze the review
            analysis_result = analyzer.analyze_review(review_text)
            
            # Display results
            st.markdown("---")
            st.header("📊 Analysis Results")
            
            # Overall sentiment
            overall_sentiment = analysis_result['overall_sentiment']
            sentiment_emoji = display_sentiment_emoji(overall_sentiment['label'])
            
            st.subheader(f"Overall Sentiment: {sentiment_emoji} {overall_sentiment['label'].title()}")
            
            # Sentiment score gauge
            st.markdown("### Sentiment Score")
            sentiment_score = overall_sentiment['score']
            if overall_sentiment['label'] == 'negative':
                sentiment_score = 1 - sentiment_score  # Invert for negative sentiment
            
            # Create a gauge chart
            fig = px.bar(
                x=[sentiment_score * 100],
                orientation='h',
                range_x=[0, 100],
                text_auto='.1f',
                color_discrete_sequence=[get_sentiment_color(overall_sentiment['label'])]
            )
            
            fig.update_layout(
                xaxis_visible=False,
                yaxis_visible=False,
                showlegend=False,
                height=100,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(range=[0, 100])
            )
            
            fig.add_annotation(
                x=sentiment_score * 100,
                y=0,
                text=f"{sentiment_score*100:.1f}% {overall_sentiment['label']}",
                showarrow=True,
                arrowhead=4,
                ax=0,
                ay=-40,
                font=dict(size=14, color=get_sentiment_color(overall_sentiment['label']))
            )
            
            st.plotly_chart(fig, use_container_width=True, use_container_height=True)
            
            # Display aspect analysis
            display_aspect_analysis(analysis_result.get('aspects', []))
            
            # Word cloud
            st.subheader("📊 Word Cloud")
            wordcloud_fig = create_wordcloud(review_text)
            if wordcloud_fig:
                st.pyplot(wordcloud_fig)
            
            # Raw JSON output (collapsible)
            with st.expander("View Raw Analysis"):
                st.json(analysis_result)
    
    elif analyze_btn and not review_text.strip():
        st.warning("Please enter a review to analyze.")

if __name__ == "__main__":
    main()
