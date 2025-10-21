import os
from pathlib import Path

# Project Directories
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# Model Configuration
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# Aspect Keywords (can be expanded)
ASPECT_KEYWORDS = {
    'camera': ['camera', 'photo', 'picture', 'lens', 'selfie', 'zoom'],
    'battery': ['battery', 'charge', 'charging', 'power', 'battery life'],
    'display': ['screen', 'display', 'screen size', 'resolution', 'touch'],
    'performance': ['speed', 'fast', 'slow', 'lag', 'performance', 'processor'],
    'price': ['price', 'cost', 'expensive', 'cheap', 'affordable', 'value'],
    'design': ['design', 'look', 'feel', 'build', 'weight', 'size', 'color'],
    'sound': ['sound', 'speaker', 'audio', 'volume', 'headphone', 'music']
}

# Sentiment Labels
SENTIMENT_LABELS = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

# Create necessary directories
for directory in [DATA_DIR, MODEL_DIR]:
    directory.mkdir(exist_ok=True)
