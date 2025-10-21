import re
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import spacy
from transformers import pipeline
import torch
from collections import defaultdict
import logging

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class SemanticReviewAnalyzer:
    def __init__(self):
        """Initialize the analyzer with pre-trained models."""
        self.logger = self._setup_logging()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self.logger.info("Loading Sentence Transformer model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        
        self.logger.info("Loading sentiment analysis model...")
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if self.device == 'cuda' else -1
        )
        
        # Initialize NLP components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Load spaCy model with multiple fallback methods
        self.nlp = self._load_spacy_model()
    
    def _load_spacy_model(self):
        """Load spaCy model with multiple fallback methods."""
        import spacy
        import subprocess
        import sys
        import os
        
        model_name = 'en_core_web_sm'
        self.logger.info(f"Attempting to load spaCy model: {model_name}")
        
        # Set user-writable directory for spaCy models
        user_home = os.path.expanduser('~')
        spacy_data_dir = os.path.join(user_home, 'spacy_models')
        os.makedirs(spacy_data_dir, exist_ok=True)
        os.environ['SPACY_DATA_DIR'] = spacy_data_dir
        
        # Method 1: Try direct load first
        try:
            nlp = spacy.load(model_name)
            self.logger.info(f"Successfully loaded {model_name} via direct load")
            return nlp
        except OSError as e:
            self.logger.warning(f"Direct load failed: {str(e)}")
        
        # Method 2: Try downloading with --user flag
        try:
            self.logger.info("Attempting to download model with --user flag...")
            subprocess.check_call([
                sys.executable, 
                "-m", 
                "spacy", 
                "download", 
                "--user",
                model_name
            ])
            nlp = spacy.load(model_name)
            self.logger.info(f"Successfully loaded {model_name} after user install")
            return nlp
        except Exception as e:
            self.logger.warning(f"User install failed: {str(e)}")
        
        # Method 3: Try installing to user directory
        try:
            self.logger.info("Attempting to install model to user directory...")
            model_url = f"https://github.com/explosion/spacy-models/releases/download/{model_name}-3.6.0/{model_name}-3.6.0.tar.gz"
            
            # Install with --user flag
            subprocess.check_call([
                sys.executable, 
                "-m", 
                "pip", 
                "install",
                "--user",
                model_url
            ])
            
            # Try loading again
            nlp = spacy.load(model_name)
            self.logger.info(f"Successfully loaded {model_name} after user pip install")
            return nlp
            
        except Exception as e:
            error_msg = f"All attempts to load {model_name} failed. Last error: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error("Tried: direct load, spacy download --user, and pip install --user")
            raise RuntimeError("Failed to load spaCy model. Please check the logs for details.")
        
    def _setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess the input text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs, special characters, and numbers
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenization and lemmatization
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Remove stopwords and short tokens
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def extract_aspects(self, text: str) -> List[str]:
        """Extract product aspects from the review text."""
        doc = self.nlp(text)
        
        # Extract noun chunks as potential aspects
        aspects = []
        for chunk in doc.noun_chunks:
            # Filter out non-relevant chunks
            if len(chunk.text.split()) <= 3:  # Limit to 3-word phrases
                aspects.append(chunk.text.lower())
        
        # Remove duplicates while preserving order
        seen = set()
        return [x for x in aspects if not (x in seen or seen.add(x))]
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of the given text."""
        try:
            result = self.sentiment_analyzer(text)[0]
            return {
                'label': result['label'].lower(),
                'score': result['score']
            }
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {str(e)}")
            return {'label': 'neutral', 'score': 0.5}
    
    def map_aspects_to_categories(self, aspects: List[str]) -> Dict[str, List[str]]:
        """Map extracted aspects to predefined categories."""
        from config import ASPECT_KEYWORDS
        
        aspect_categories = defaultdict(list)
        
        for aspect in aspects:
            aspect_lower = aspect.lower()
            matched = False
            
            for category, keywords in ASPECT_KEYWORDS.items():
                if any(keyword in aspect_lower for keyword in keywords):
                    aspect_categories[category].append(aspect)
                    matched = True
                    break
            
            if not matched:
                # If no category matches, create a new one with the aspect name
                aspect_categories[aspect] = [aspect]
        
        return dict(aspect_categories)
    
    def analyze_review(self, review_text: str) -> Dict:
        """Analyze a single product review."""
        if not review_text.strip():
            return {"error": "Empty review text provided"}
        
        # Preprocess the text
        preprocessed_text = self.preprocess_text(review_text)
        
        # Extract aspects
        aspects = self.extract_aspects(preprocessed_text)
        
        # If no aspects found, use the whole review for sentiment
        if not aspects:
            sentiment = self.analyze_sentiment(review_text)
            return {
                "review": review_text,
                "overall_sentiment": {
                    "label": sentiment['label'],
                    "score": float(sentiment['score'])
                },
                "aspects": []
            }
        
        # Get aspect categories
        aspect_categories = self.map_aspects_to_categories(aspects)
        
        # Analyze sentiment for each aspect
        aspect_results = []
        for category, category_aspects in aspect_categories.items():
            # Create a sentence with the aspect and context
            aspect_text = f"The {category} is {review_text}"
            sentiment = self.analyze_sentiment(aspect_text)
            
            aspect_results.append({
                "aspect": category,
                "mentions": category_aspects,
                "sentiment": {
                    "label": sentiment['label'],
                    "score": float(sentiment['score'])
                }
            })
        
        # Get overall sentiment
        overall_sentiment = self.analyze_sentiment(review_text)
        
        return {
            "review": review_text,
            "overall_sentiment": {
                "label": overall_sentiment['label'],
                "score": float(overall_sentiment['score'])
            },
            "aspects": aspect_results
        }

# Singleton instance
analyzer = SemanticReviewAnalyzer()
