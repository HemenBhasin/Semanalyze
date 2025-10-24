import re
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from spacy.matcher import Matcher
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
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """Initialize the semantic analyzer with a pre-trained sentiment model."""
        try:
            # Try to load the spaCy model
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # If model not found, download it
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize the sentiment analyzer pipeline with batching
        device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis", 
                model=model_name,
                device=device,
                truncation=True,
                batch_size=8  # Process multiple clauses in parallel
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize sentiment analyzer: {str(e)}")
            raise
        
        # Initialize the Matcher for aspect extraction
        self.matcher = Matcher(self.nlp.vocab)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Add aspect patterns to matcher
        self._initialize_aspect_patterns()
    
    def _initialize_aspect_patterns(self):
        """Initialize patterns for aspect extraction."""
        # Adjective + Noun patterns
        adj_noun = [
            {"POS": "ADJ", "OP": "?"},
            {"POS": "NOUN"}
        ]
        
        # Noun + Preposition + Noun (e.g., "quality of the screen")
        noun_prep_noun = [
            {"POS": "NOUN"},
            {"POS": "ADP"},  # Preposition
            {"POS": "DET", "OP": "?"},
            {"POS": "NOUN"}
        ]
        
        # Add patterns to matcher
        self.matcher.add("ADJ_NOUN", [adj_noun])
        self.matcher.add("NOUN_PREP_NOUN", [noun_prep_noun])
        
        # Add common aspect patterns
        self.aspect_patterns = [
            (r'\b(screen|display|monitor)\b', 'display'),
            (r'\b(battery|battery life|battery time)\b', 'battery'),
            (r'\b(performance|speed|processing|processor|cpu|gpu|ram|memory|storage|ssd|hard drive)\b', 'performance'),
            (r'\b(keyboard|trackpad|touchpad|mouse|input device)\b', 'input devices'),
            (r'\b(speaker|audio|sound|microphone|mic)\b', 'audio'),
            (r'\b(design|build|looks|appearance|aesthetics)\b', 'design'),
            (r'\b(port|usb|hdmi|thunderbolt|connector|jack|slot|sd card|headphone)\b', 'ports'),
            (r'\b(weight|lightweight|heavy|portability|size|dimension)\b', 'portability'),
            (r'\b(price|cost|value|worth|expensive|cheap|affordable)\b', 'price'),
            (r'\b(camera|webcam|video call|selfie)\b', 'camera'),
            (r'\b(software|os|operating system|windows|macos|linux|driver|firmware)\b', 'software'),
            (r'\b(heat|temperature|cooling|fan|noise|ventilation)\b', 'thermal performance'),
            (r'\b(keyboard|key|typing|backlit|backlight|illuminated)\b', 'keyboard'),
            (r'\b(trackpad|touchpad|gesture|pointing device)\b', 'trackpad'),
            (r'\b(bluetooth|wifi|wireless|connectivity|nfc|gps|ethernet|lan)\b', 'connectivity'),
            (r'\b(upgrade|ram upgrade|storage upgrade|expandability|user replaceable)\b', 'upgradability'),
            (r'\b(keyboard|touchpad|trackpad|mouse|stylus|pen|touch screen|touchscreen)\b', 'input devices'),
            (r'\b(hinge|build quality|durability|sturdiness|material|aluminum|plastic|metal|magnesium)\b', 'build quality'),
            (r'\b(service|support|warranty|return policy|customer service|repair|replacement)\b', 'customer support'),
            (r'\b(bloatware|pre-installed|trial|adware|unnecessary software|bloat)\b', 'bloatware')
        ]
        
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
        tokens = [self.nlp(token)[0].lemma_ for token in tokens]
        
        # Remove stopwords and short tokens
        tokens = [token for token in tokens if token not in stopwords.words('english') and len(token) > 2]
        
        return ' '.join(tokens)
    
    def extract_aspects(self, text: str) -> List[str]:
        """
        Enhanced aspect extraction using spaCy's POS tagging, dependency parsing, and pattern matching.
        
        Args:
            text: The input text to extract aspects from
            
        Returns:
            List of extracted and grouped aspects
        """
        if not text.strip():
            return []
            
        doc = self.nlp(text.lower().strip())
        
        # Define more comprehensive aspect patterns
        patterns = [
            # Adjective + Noun (e.g., "fast performance", "good quality")
            [{"POS": "ADJ"}, {"POS": "NOUN"}],
            # Adverb + Adjective + Noun (e.g., "very good performance")
            [{"POS": "ADV"}, {"POS": "ADJ"}, {"POS": "NOUN"}],
            # Noun + Preposition + Noun (e.g., "quality of service")
            [{"POS": "NOUN"}, {"POS": "ADP"}, {"POS": "NOUN"}],
            # Compound nouns (e.g., "battery life", "screen resolution")
            [{"POS": "NOUN"}, {"POS": "NOUN"}],
            # Adjective + Adjective + Noun (e.g., "long battery life")
            [{"POS": "ADJ"}, {"POS": "ADJ"}, {"POS": "NOUN"}]
        ]
        
        # Initialize matcher with patterns
        matcher = Matcher(self.nlp.vocab)
        matcher.add("ASPECT_PATTERNS", patterns)
        
        # Extract aspects using multiple strategies
        aspects = set()
        
        # 1. Extract using pattern matching
        matches = matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            if 1 <= len(span) <= 4:  # Limit to 1-4 word spans
                aspect_text = span.text.lower()
                # Skip if it's just a determiner or very common word
                if len(aspect_text) > 2 and not any(t.is_stop for t in span):
                    aspects.add(aspect_text)
        
        # 2. Extract noun chunks with better filtering
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower()
            # Skip pronouns, single letters, and very common words
            if (chunk.root.pos_ in ["NOUN", "PROPN"] and 
                len(chunk_text) > 2 and
                not any(t.is_stop or t.is_punct for t in chunk) and
                chunk_text not in ["it", "this", "that", "they", "them"] and
                not chunk_text.replace("'", "").isnumeric()):
                aspects.add(chunk_text)
        
        # 3. Extract compound nouns and named entities
        for token in doc:
            # Handle compound nouns (e.g., "battery_life")
            if token.dep_ in ["compound", "amod", "nmod"] and token.head.pos_ == "NOUN":
                compound = f"{token.text} {token.head.text}".lower()
                if len(compound) > 4:  # Minimum length check
                    aspects.add(compound)
            
            # Add individual nouns that might have been missed
            elif (token.pos_ in ["NOUN", "PROPN"] and 
                  len(token.text) > 2 and 
                  not token.is_stop and 
                  not any(token.text in a for a in aspects)):
                aspects.add(token.text.lower())
        
        # Filter and clean aspects
        filtered_aspects = set()
        common_terms = {
            "product", "item", "thing", "something", "anything", "one", "way", 
            "time", "day", "people", "lot", "bit", "part", "kind", "sort",
            "use", "make", "take", "give", "get", "like", "look", "see"
        }
        
        for aspect in aspects:
            # Skip if aspect is too short or in common terms
            if len(aspect) < 3:
                continue
                
            # Skip if any word is a stopword or common term
            words = aspect.split()
            if any(word in common_terms for word in words):
                continue
                
            # Skip if it's just a number
            if all(w.replace("'", "").replace("-", "").isnumeric() for w in words):
                continue
                
            filtered_aspects.add(aspect)
        
        # Group similar aspects using word embeddings for better clustering
        aspect_list = list(filtered_aspects)
        if not aspect_list:
            return []
            
        # Group by head noun
        aspect_groups = {}
        for aspect in aspect_list:
            doc = self.nlp(aspect)
            head = next((t for t in reversed(doc) if t.dep_ == "ROOT" or t.head == t), doc[-1] if doc else None)
            if head:
                head_lemma = head.lemma_
                if head_lemma not in aspect_groups:
                    aspect_groups[head_lemma] = []
                aspect_groups[head_lemma].append(aspect)
        
        # Select the most descriptive aspect from each group
        final_aspects = []
        for group in aspect_groups.values():
            if len(group) == 1:
                final_aspects.append(group[0])
            else:
                # Prefer longer, more specific phrases
                best_aspect = max(group, key=lambda x: (
                    len(x.split()),  # Prefer multi-word phrases
                    sum(1 for t in self.nlp(x) if t.pos_ in ["ADJ", "NOUN"]),  # Prefer more content words
                    -len(x) / 10  # Slight preference for shorter phrases among equals
                ))
                final_aspects.append(best_aspect)
        
        # Sort by length (longer phrases first) then alphabetically
        final_aspects.sort(key=lambda x: (-len(x.split()), x))
        
        return final_aspects
    
    def _split_at_contrast_markers(self, text: str) -> list[tuple[str, bool]]:
        """Split text at contrast markers and return chunks with their sentiment influence."""
        # Define contrast markers and their impact on following text
        contrast_markers = {
            'but': True, 'however': True, 'although': False, 'though': False,
            'yet': True, 'except': True, 'despite': False, 'whereas': True,
            'while': True, 'nevertheless': True, 'nonetheless': True,
            'on the other hand': True, 'in contrast': True, 'conversely': True,
            'that said': True, 'having said that': True, 'then again': True
        }
        
        # Convert to lowercase for case-insensitive matching
        lower_text = text.lower()
        chunks = []
        last_pos = 0
        
        # Find all contrast markers and their positions
        markers = []
        for marker in sorted(contrast_markers.keys(), key=len, reverse=True):
            pos = 0
            while True:
                pos = lower_text.find(marker, pos)
                if pos == -1:
                    break
                # Check if it's a whole word
                if (pos == 0 or not lower_text[pos-1].isalnum()) and \
                   (pos + len(marker) >= len(lower_text) or not lower_text[pos + len(marker)].isalnum()):
                    markers.append((pos, marker, contrast_markers[marker]))
                pos += len(marker)
        
        # Sort markers by position
        markers.sort()
        
        # Split text at markers
        for pos, marker, is_strong in markers:
            if pos > last_pos:
                chunk = text[last_pos:pos].strip()
                if chunk:
                    chunks.append((chunk, False))  # Text before marker
            
            # Add the marker itself
            chunks.append((text[pos:pos+len(marker)], is_strong))
            last_pos = pos + len(marker)
        
        # Add remaining text
        if last_pos < len(text):
            chunk = text[last_pos:].strip()
            if chunk:
                chunks.append((chunk, False))
        
        # If no chunks were created, return the whole text
        if not chunks:
            return [(text, False)]
            
        return chunks

    def _analyze_sentence_sentiment(self, text: str) -> tuple[float, str]:
        """Analyze sentiment of a single sentence and return score and label."""
        try:
            # Skip very short texts that might be just punctuation or stopwords
            if len(text.split()) <= 2 and not any(c.isalpha() for c in text):
                return 0.5, 'neutral'
                
            result = self.sentiment_analyzer(text)[0]
            score = result['score'] if result['label'] == 'POSITIVE' else 1 - result['score']
            return score, result['label'].lower()
        except Exception as e:
            self.logger.warning(f"Error analyzing sentence '{text}': {str(e)}")
            return 0.5, 'neutral'

    def _detect_sentiment_shift(self, doc) -> tuple[list, float, float]:
        """
        Detect sentiment shift within a document.
        
        Returns:
            tuple: (sentiment_changes, avg_shift, shift_ratio)
                - sentiment_changes: List of (sentence, score, label, shift_from_previous)
                - avg_shift: Average magnitude of sentiment shifts
                - shift_ratio: Ratio of sentences with significant shifts
        """
        if len(doc) < 2:
            return [("", 0.5, 'neutral', 0.0)], 0.0, 0.0
            
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]
        if not sentences:
            return [("", 0.5, 'neutral', 0.0)], 0.0, 0.0
            
        # Analyze each sentence
        sentiment_data = []
        prev_score = None
        shift_magnitude = 0.0
        shift_count = 0
        
        for i, sent in enumerate(sentences):
            score, label = self._analyze_sentence_sentiment(sent)
            shift = abs(score - prev_score) if prev_score is not None else 0.0
            
            # Check for significant shift
            if prev_score is not None and shift > 0.3:  # Threshold for significant shift
                shift_magnitude += shift
                shift_count += 1
                
            sentiment_data.append((sent, score, label, shift))
            prev_score = score
        
        # Calculate metrics
        avg_shift = (shift_magnitude / shift_count) if shift_count > 0 else 0.0
        shift_ratio = shift_count / (len(sentences) - 1) if len(sentences) > 1 else 0.0
        
        return sentiment_data, avg_shift, shift_ratio

    def _summarize_sentiment_chunks(self, chunks: list[tuple[str, bool]]) -> tuple[float, list]:
        """
        Analyze and summarize sentiment across text chunks.
        
        Args:
            chunks: List of (text, is_after_contrast) tuples
            
        Returns:
            tuple: (final_score, analysis_results)
                - final_score: Combined sentiment score (0-1)
                - analysis_results: List of (text, score, label, weight)
        """
        if not chunks:
            return 0.5, []
            
        results = []
        total_weight = 0
        weighted_sum = 0
        
        # First pass: analyze all chunks
        for i, (chunk, is_contrast) in enumerate(chunks):
            # Skip very short chunks that are just contrast markers
            if len(chunk.split()) <= 2 and not any(c.isalpha() for c in chunk):
                continue
                
            # Analyze chunk sentiment
            score, label = self._analyze_sentence_sentiment(chunk)
            
            # Weight chunks after contrast markers more heavily
            weight = 2.0 if is_contrast else 1.0
            
            # Slightly increase weight of negative sentiments (negativity bias)
            if score < 0.4:
                weight *= 1.2
                
            results.append({
                'text': chunk,
                'score': score,
                'label': label,
                'weight': weight,
                'is_contrast': is_contrast
            })
            
            weighted_sum += score * weight
            total_weight += weight
        
        if not results:
            return 0.5, []
            
        # Calculate weighted average score
        final_score = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        return final_score, results

    def _analyze_sentiment_context(self, text: str) -> tuple[float, float, list]:
        """
        Analyze sentiment with context awareness and contrast handling.
        
        Returns:
            tuple: (base_score, shift_ratio, sentiment_changes)
                - base_score: Overall sentiment score (0-1)
                - shift_ratio: Ratio of sentences with significant shifts
                - sentiment_changes: List of (sentence, score, label, shift, weight)
        """
        # Split text into chunks at contrast markers
        chunks = self._split_at_contrast_markers(text)
        
        # Analyze each chunk with proper weighting
        base_score, chunk_analysis = self._summarize_sentiment_chunks(chunks)
        
        # Convert to sentiment changes format
        sentiment_changes = []
        shift_count = 0
        prev_score = None
        
        for i, chunk in enumerate(chunk_analysis):
            score = chunk['score']
            shift = abs(score - prev_score) if prev_score is not None else 0.0
            
            # Check for significant shift
            if prev_score is not None and shift > 0.3:
                shift_count += 1
                
            sentiment_changes.append((
                chunk['text'],
                score,
                chunk['label'],
                shift,
                chunk['weight']
            ))
            prev_score = score
        
        # Calculate shift ratio
        shift_ratio = shift_count / len(chunk_analysis) if chunk_analysis else 0.0
        
        # If there are significant shifts, adjust final score
        if shift_ratio > 0.3:
            # Give more weight to negative chunks
            negative_weight = sum(
                (1 - chunk['score']) * chunk['weight'] 
                for chunk in chunk_analysis 
                if chunk['score'] < 0.4
            )
            positive_weight = sum(
                chunk['score'] * chunk['weight'] 
                for chunk in chunk_analysis 
                if chunk['score'] > 0.6
            )
            
            if negative_weight > positive_weight * 1.5:
                base_score = min(base_score, 0.4)  # Cap at slightly negative
            
        return base_score, shift_ratio, sentiment_changes

    def _analyze_clause_sentiment(self, clause: str, aspect: str = None) -> Dict:
        """
        Analyze sentiment of a single clause with enhanced context awareness and contrast handling.
        
        Args:
            clause: The text clause to analyze
            aspect: Optional aspect to focus the analysis on
            
        Returns:
            Dictionary with sentiment analysis results including:
            - score: Sentiment score (0-1)
            - label: Sentiment label (positive/negative/neutral/mixed)
            - confidence: Confidence in the prediction (0-1)
            - aspects: If aspect is provided, aspect-specific sentiment
            - sentiment_changes: Detailed sentiment analysis of chunks
        """
        try:
            if not clause.strip():
                return {'score': 0.5, 'label': 'neutral', 'confidence': 0.0, 'text': ''}
            
            # Preprocess the clause
            clause = clause.strip()
            
            # If analyzing a specific aspect, add context
            if aspect:
                # Add aspect context to help the model focus
                clause = f"When considering {aspect}, {clause}"
            
            # Sentiment analysis with context and contrast handling
            base_score, shift_ratio, sentiment_changes = self._analyze_sentiment_context(clause)
            
            # Calculate confidence based on sentiment strength and consistency
            confidence = 0.8  # Base confidence
            
            # Adjust confidence based on sentiment shifts
            if shift_ratio > 0.3:  # High shift ratio indicates mixed sentiment
                confidence *= (1.0 - (shift_ratio * 0.7))  # Reduce confidence for mixed sentiment
                
            # Adjust confidence based on sentiment strength
            sentiment_strength = abs(base_score - 0.5) * 2  # 0-1, higher is stronger
            confidence = max(0.1, min(0.95, confidence * (0.7 + (sentiment_strength * 0.3))))
            
            # Determine final label
            if shift_ratio > 0.4 and len(sentiment_changes) > 1:
                # If significant shifts, label as mixed sentiment
                label = 'mixed'
                # For mixed sentiment, move score towards neutral based on shift ratio
                base_score = 0.5 + ((base_score - 0.5) * (1.0 - (shift_ratio * 0.7)))
            elif base_score > 0.6:
                label = 'positive'
            elif base_score < 0.4:
                label = 'negative'
            else:
                label = 'neutral'
            
            # Prepare result
            result = {
                'score': float(base_score),
                'label': label,
                'confidence': float(confidence),
                'text': clause,
                'sentiment_changes': [
                    {
                        'text': chunk[0],
                        'score': float(chunk[1]),
                        'label': chunk[2],
                        'shift': float(chunk[3]),
                        'weight': float(chunk[4]) if len(chunk) > 4 else 1.0
                    }
                    for chunk in sentiment_changes
                ]
            }
            
            # Add aspect-specific information if needed
            if aspect:
                result['aspect'] = aspect
                result['aspect_score'] = base_score
                
            return result
            negation_terms = {'no', 'not', 'none', 'never', 'nothing', 'nowhere', 'neither', 'nor', 
                             'barely', 'hardly', 'scarcely', 'rarely', 'seldom', 'without'}
            
            contrast_terms = {'but', 'however', 'although', 'though', 'yet', 'except', 'despite', 
                            'whereas', 'while', 'nevertheless', 'nonetheless', 'on the other hand',
                            'even though', 'in contrast', 'conversely'}
            
            intensifiers = {
                'very': 0.2, 'really': 0.2, 'extremely': 0.25, 'absolutely': 0.25,
                'completely': 0.3, 'totally': 0.25, 'utterly': 0.3, 'highly': 0.2,
                'remarkably': 0.25, 'exceptionally': 0.3, 'incredibly': 0.25,
                'especially': 0.2, 'particularly': 0.2, 'seriously': 0.15
            }
            
            diminishers = {
                'slightly': 0.15, 'somewhat': 0.15, 'a bit': 0.1, 'a little': 0.1,
                'marginally': 0.15, 'moderately': 0.15, 'partially': 0.1,
                'fairly': 0.1, 'quite': 0.1, 'rather': 0.1
            }
            
            # Sarcasm indicators
            sarcasm_indicators = {
                'as if', 'yeah right', 'sure', 'of course', 'obviously', 'clearly',
                'wow', 'oh great', 'just what i needed', 'perfect', 'wonderful'
            }
            
            # Initialize sentiment modifiers
            negation_boost = 1.0
            contrast_boost = 1.0
            intensity_modifier = 1.0
            sarcasm_detected = False
            
            # Check for negation patterns in the clause
            for token in doc:
                # Handle negation
                if token.text in negation_terms or any(dep == 'neg' for dep in [child.dep_ for child in token.children]):
                    negation_boost *= -1
                
                # Handle intensifiers and diminishers
                if token.text in intensifiers:
                    intensity_modifier += intensifiers[token.text]
                elif token.text in diminishers:
                    intensity_modifier -= diminishers[token.text]
            
            # Check for contrastive conjunctions
            for sent in doc.sents:
                sent_text = sent.text.lower()
                if any(term in sent_text for term in contrast_terms):
                    contrast_boost = 0.7  # Reduce confidence when contrast is present
            
            # Check for sarcasm
            clause_lower = clause.lower()
            sarcasm_detected = any(indicator in clause_lower for indicator in sarcasm_indicators)
            
            # Adjust score based on modifiers
            if sarcasm_detected:
                base_score = 1.0 - base_score  # Invert sentiment for sarcasm
            
            # Apply intensity modifier (cap between 0.1 and 2.0)
            intensity_modifier = max(0.1, min(2.0, intensity_modifier))
            
            # Apply negation and contrast
            adjusted_score = base_score * negation_boost * contrast_boost * intensity_modifier
            
            # Ensure score is within 0-1 range
            adjusted_score = max(0.0, min(1.0, adjusted_score))
            
            # Calculate confidence based on modifiers and shift ratio
            confidence = 0.8  # Base confidence
            if abs(adjusted_score - 0.5) < 0.1:  # If close to neutral
                confidence *= 0.8
            
            # Reduce confidence for mixed sentiment
            confidence *= (1.0 - (shift_ratio * 0.5))
            
            # Determine sentiment label
            if sarcasm_detected:
                label = 'sarcastic'
            elif adjusted_score > 0.6:
                label = 'positive'
            elif adjusted_score < 0.4:
                label = 'negative'
            else:
                label = 'neutral'
            
            # If analyzing a specific aspect
            if aspect:
                # Check if aspect is mentioned in this clause
                aspect_terms = aspect.lower().split()
                aspect_found = any(any(token.text == term for term in aspect_terms) for token in doc)
                if not aspect_found:
                    return {'score': 0.5, 'label': 'neutral', 'confidence': 0.0, 'text': clause.strip()}
                
                # Analyze with aspect focus
                result = self.sentiment_analyzer(f"{clause} [ASPECT: {aspect}]")[0]
            else:
                result = self.sentiment_analyzer(clause)[0]
            
            # Get base score and adjust for negation/intensifiers
            score = float(result['score'])
            original_label = result['label']
            
            # Check for negation patterns and negative sentiment words
            has_negation = any(token.text in negation_terms for token in doc)
            has_contrast = any(token.text in contrast_terms for token in doc)
            
            # List of strong negative words that should always indicate negative sentiment
            strong_negative_words = {
                'terrible', 'awful', 'horrible', 'disappointing', 'poor', 'bad', 
                'worst', 'garbage', 'trash', 'useless', 'waste', 'rubbish', 'pathetic',
                'disappointed', 'frustrated', 'annoyed', 'angry', 'hate', 'dislike'
            }
            
            # Check for strong negative words in the text
            strong_negative_context = any(
                token.lemma_.lower() in strong_negative_words 
                for token in doc
            )
            
            # Adjust score based on context
            if strong_negative_context:
                # If we have strong negative words, ensure the score reflects negative sentiment
                score = min(0.3, score)  # Cap at 0.3 to ensure negative classification
            
            # Handle negation
            if has_negation:
                # For negative contexts, we need to be more aggressive with sentiment inversion
                if original_label == 'POSITIVE':
                    # If the model thought it was positive but we have negation, make it strongly negative
                    score = 0.2  # Force negative sentiment
                else:
                    # For already negative sentiment with negation, it might be a double negative
                    # or reinforcement of negativity
                    score = max(0.0, 1.0 - score - 0.3)  # Invert and shift more negative
            
            # Adjust for intensifiers and diminishers
            for token in doc:
                if token.text in intensifiers:
                    if score > 0.5:  # Positive sentiment
                        score = min(1.0, score + 0.1)
                    else:  # Negative sentiment
                        score = max(0.0, score - 0.1)
                elif token.text in diminishers:
                    # Move score towards neutral
                    score = 0.5 + (score - 0.5) * 0.7
            
            # Calculate confidence
            base_confidence = abs(score - 0.5) * 2  # 0-1 range
            
            # Strong sentiment words
            strong_negative = any(
                token.lemma_.lower() in {
                    'terrible', 'awful', 'horrible', 'disappointing', 'poor', 'bad', 
                    'worst', 'garbage', 'trash', 'useless', 'waste', 'rubbish', 'pathetic'
                } 
                for token in doc
            )
            
            strong_positive = any(
                token.lemma_.lower() in {
                    'excellent', 'amazing', 'outstanding', 'perfect', 'great',
                    'awesome', 'fantastic', 'superb', 'wonderful', 'brilliant'
                } 
                for token in doc
            )
            
            # Boost confidence for strong sentiment words
            if strong_negative:
                base_confidence = min(1.0, base_confidence * 1.3)
                # Ensure strong negative words push the score lower
                if score > 0.5:
                    score = 0.5 - (score - 0.5)  # Invert the positive score
            
            if strong_positive:
                base_confidence = min(1.0, base_confidence * 1.2)
            
            # Adjust confidence based on clause length and complexity
            clause_length = len([token for token in doc if not token.is_punct])
            if clause_length > 15:  # Longer clauses might have mixed sentiment
                base_confidence *= 0.9
            
            # Determine final label with adjusted thresholds
            if strong_negative_context and not strong_positive:
                # If we have strong negative context and no strong positive, it's negative
                label = 'negative'
                score = min(score, 0.3)  # Ensure score reflects negative sentiment
            elif strong_positive and not strong_negative_context:
                # If we have strong positive context and no strong negative, it's positive
                label = 'positive'
                score = max(score, 0.7)  # Ensure score reflects positive sentiment
            elif score > 0.6:
                label = 'positive'
            elif score < 0.4 or (has_negation and score < 0.6):
                # Be more aggressive in marking negative with negation
                label = 'negative'
            else:
                # For neutral range, check if we have mixed signals
                if has_negation and has_contrast:
                    label = 'mixed'
                else:
                    label = 'neutral'
            
            return {
                'score': min(max(score, 0.0), 1.0),  # Ensure score is between 0 and 1
                'label': label,
                'confidence': min(max(base_confidence, 0.0), 1.0),  # Ensure confidence is between 0 and 1
                'text': clause.strip()
            }
            
        except Exception as e:
            self.logger.warning(f"Error analyzing clause: {str(e)}")
            return {'score': 0.5, 'label': 'neutral', 'confidence': 0.0, 'text': clause.strip()}

    def analyze_sentiment(self, text: str, aspect: str = None) -> Dict:
        """
        Analyze sentiment of the given text with clause-based analysis.
        
        Args:
            text: The text to analyze
            aspect: Optional aspect to focus the analysis on
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            doc = self.nlp(text)
            
            # If aspect is provided, find all its occurrences
            if aspect:
                aspect_tokens = [t.lower() for t in aspect.split()]
                aspect_occurrences = []
                
                # Find all occurrences of the aspect in the text
                for i in range(len(doc) - len(aspect_tokens) + 1):
                    if all(doc[i + j].text.lower() == aspect_tokens[j] for j in range(len(aspect_tokens))):
                        aspect_occurrences.append(i)
                
                if not aspect_occurrences:
                    return {
                        'label': 'neutral',
                        'score': 0.5,
                        'confidence': 0.0,
                        'aspect': aspect,
                        'clauses_analyzed': 0
                    }
                
                # Analyze each occurrence with context
                aspect_scores = []
                
                for pos in aspect_occurrences:
                    # Get context window and analyze
                    context_span, contrast_data, _ = self._get_context_window(doc, pos)
                    context_text = ' '.join([token.text for token in context_span])
                    
                    try:
                        # Get base sentiment of the context
                        result = self.sentiment_analyzer(context_text)[0]
                        base_score = result['score']
                        
                        # Adjust for contrast and context
                        adjusted_score = self._adjust_for_context(doc, base_score, pos, aspect)
                        
                        # Calculate confidence based on score distance from neutral and context length
                        confidence = abs(adjusted_score - 0.5) * 2
                        context_length = len(context_text.split())
                        
                        # Adjust confidence based on context length
                        if context_length < 5:
                            confidence *= 0.7
                        
                        aspect_scores.append({
                            'score': adjusted_score,
                            'confidence': confidence,
                            'context': context_text,
                            'position': pos,
                            'base_score': base_score
                        })
                        
                    except Exception as e:
                        self.logger.warning(f"Error analyzing aspect '{aspect}' at position {pos}: {str(e)}")
                        continue
                
                if not aspect_scores:
                    return {
                        'label': 'neutral',
                        'score': 0.5,
                        'confidence': 0.0,
                        'aspect': aspect,
                        'clauses_analyzed': 0
                    }
                
                # Calculate weighted average score
                total_weight = sum(s['confidence'] ** 2 for s in aspect_scores)
                if total_weight > 0:
                    avg_score = sum(s['score'] * (s['confidence'] ** 2) for s in aspect_scores) / total_weight
                else:
                    avg_score = sum(s['score'] for s in aspect_scores) / len(aspect_scores)
                
                # Calculate average confidence
                avg_confidence = sum(s['confidence'] for s in aspect_scores) / len(aspect_scores)
                
                # Determine label
                if avg_score > 0.6:
                    label = 'positive'
                elif avg_score < 0.4:
                    label = 'negative'
                else:
                    label = 'neutral'
                
                self.logger.info(
                    f"Aspect Analysis | '{aspect}' | "
                    f"Score: {avg_score:.3f} ({label}) | "
                    f"Confidence: {avg_confidence:.2f} | "
                    f"Occurrences: {len(aspect_scores)}"
                )
                
                return {
                    'label': label,
                    'score': avg_score,
                    'confidence': avg_confidence,
                    'aspect': aspect,
                    'clauses_analyzed': len(aspect_scores)
                }
            
            # For general sentiment (no specific aspect)
            # Enhanced clause splitting that preserves contrastive relationships
            clauses = []
            current_clause = []
            contrast_indicators = ['but', 'however', 'although', 'though', 'yet', 'except', 'despite', 'whereas', 'while', 'nevertheless', 'nonetheless']
            
            # First, split into sentences
            for sent in doc.sents:
                sent_text = sent.text
                
                # Check for contrastive conjunctions
                has_contrast = any(conj in sent_text.lower() for conj in contrast_indicators)
                
                if has_contrast:
                    # For contrastive sentences, split into segments
                    segments = []
                    current_segment = []
                    
                    for token in sent:
                        if token.text.lower() in contrast_indicators:
                            if current_segment:
                                segments.append(' '.join(current_segment).strip())
                                current_segment = []
                            segments.append(token.text)  # Keep the contrast word as a separate segment
                        else:
                            current_segment.append(token.text)
                    
                    if current_segment:  # Add the last segment
                        segments.append(' '.join(current_segment).strip())
                    
                    # Process segments, giving more weight to segments after contrast words
                    for i, segment in enumerate(segments):
                        if segment.lower() in contrast_indicators:
                            continue  # Skip the contrast word itself
                            
                        # Segments after contrast words get higher weight
                        weight = 1.5 if i > 0 and segments[i-1].lower() in contrast_indicators else 1.0
                        clauses.append({'text': segment, 'weight': weight, 'is_after_contrast': weight > 1.0})
                else:
                    # For non-contrastive sentences, add as is with normal weight
                    clauses.append({'text': sent_text, 'weight': 1.0, 'is_after_contrast': False})
            
            # Analyze each clause with context awareness
            clause_results = []
            for i, clause_info in enumerate(clauses):
                clause = clause_info['text']
                if not clause.strip():
                    continue
                
                # Analyze the clause
                result = self._analyze_clause_sentiment(clause)
                
                # Adjust confidence based on position (clauses after contrast get higher confidence)
                if clause_info['is_after_contrast']:
                    result['confidence'] = min(1.0, result['confidence'] * 1.3)  # 30% boost
                
                # Only include clauses with sufficient confidence
                if result['confidence'] > 0.1:
                    result['weight'] = clause_info['weight']
                    clause_results.append(result)
            
            if not clause_results:
                # Fallback to full text analysis if no clauses were confident
                result = self.sentiment_analyzer(text)[0]
                overall_score = float(result['score'])
                label = 'positive' if overall_score > 0.6 else 'negative' if overall_score < 0.4 else 'neutral'
                return {
                    'label': label,
                    'score': overall_score,
                    'confidence': abs(overall_score - 0.5) * 2,
                    'aspect': 'overall',
                    'clauses_analyzed': 1
                }
            
            # Calculate sentiment with contrast awareness
            positive_scores = []
            negative_scores = []
            
            for result in clause_results:
                score = result['score']
                weight = result['weight'] * result['confidence']
                
                if score > 0.5:  # Positive sentiment
                    positive_scores.append((score, weight))
                elif score < 0.5:  # Negative sentiment
                    negative_scores.append((1 - score, weight))  # Invert to get positive magnitude
            
            # Calculate average positive and negative strengths
            pos_strength = sum(s * w for s, w in positive_scores) / sum(w for _, w in positive_scores) if positive_scores else 0
            neg_strength = sum(s * w for s, w in negative_scores) / sum(w for _, w in negative_scores) if negative_scores else 0
            
            # Calculate overall score considering both positive and negative aspects
            if pos_strength > 0 and neg_strength > 0:
                # When both positive and negative aspects are present, balance them
                if pos_strength > neg_strength * 1.5:  # If positive is significantly stronger
                    overall_score = 0.5 + (pos_strength * 0.5)  # 0.5 to 1.0 range
                elif neg_strength > pos_strength * 1.5:  # If negative is significantly stronger
                    overall_score = 0.5 - (neg_strength * 0.5)  # 0.0 to 0.5 range
                else:
                    # If they're relatively balanced, move towards neutral with a slight bias
                    overall_score = 0.5 + ((pos_strength - neg_strength) * 0.25)
            elif pos_strength > 0:
                overall_score = 0.5 + (pos_strength * 0.5)  # 0.5 to 1.0 range
            elif neg_strength > 0:
                overall_score = 0.5 - (neg_strength * 0.5)  # 0.0 to 0.5 range
            else:
                overall_score = 0.5  # Neutral if no clear sentiment
            
            # Calculate average confidence
            avg_confidence = sum(r['confidence'] * r['weight'] for r in clause_results) / sum(r['weight'] for r in clause_results)
            
            # Determine overall label with more nuanced thresholds
            if overall_score > 0.7:
                label = 'positive'
            elif overall_score < 0.3:
                label = 'negative'
            else:
                if pos_strength > 0 and neg_strength > 0:
                    if abs(pos_strength - neg_strength) < 0.2:  # If strengths are close
                        label = 'mixed'
                    else:
                        label = 'slightly positive' if overall_score > 0.5 else 'slightly negative'
                else:
                    label = 'neutral'
                
            return {
                'label': label,
                'score': float(overall_score),
                'confidence': float(avg_confidence),
                'aspect': 'overall',
                'clauses_analyzed': len(clause_results)
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
        """
        Analyze a product review with enhanced context-aware sentiment analysis.
        
        Args:
            review_text: The review text to analyze
            
        Returns:
            Dictionary containing analysis results with sentiment, aspects, and key phrases
            {
                'review': str,
                'overall_sentiment': {'label': str, 'score': float, 'confidence': float},
                'aspects': [{
                    'aspect': str, 
                    'sentiment': str, 
                    'score': float,
                    'weight': float,
                    'confidence': float,
                    'occurrences': int
                }],
                'key_phrases': List[str],
                'summary': str,
                'contextual_sentences': List[dict]
            }
        """
        if not review_text.strip():
            return {
                'review': '',
                'overall_sentiment': {'label': 'neutral', 'score': 0.5, 'confidence': 0.0},
                'aspects': [],
                'key_phrases': [],
                'summary': 'No review text provided.',
                'contextual_sentences': []
            }
            
        try:
            # Preprocess the review text
            preprocessed_text = self.preprocess_text(review_text)
            doc = self.nlp(review_text)
            
            # Extract aspects with context
            aspects = self.extract_aspects(preprocessed_text)
            
            # If no aspects found, use the whole review for sentiment
            if not aspects:
                sentiment = self.analyze_sentiment(review_text)
                return {
                    'review': review_text,
                    'overall_sentiment': {
                        'label': sentiment['label'],
                        'score': float(sentiment['score']),
                        'confidence': float(sentiment.get('confidence', 0.0))
                    },
                    'aspects': [],
                    'key_phrases': [],
                    'summary': 'No specific aspects found in the review.',
                    'contextual_sentences': []
                }
            
            # Analyze sentiment for each aspect with context
            aspect_sentiments = {}
            contextual_sentences = []
            
            for aspect in aspects:
                # Get aspect weight based on importance
                aspect_weight = self._get_aspect_weight(aspect)
                
                # Analyze sentiment with the enhanced analyzer
                sentiment_result = self.analyze_sentiment(review_text, aspect)
                
                # Store the result with additional metadata
                aspect_sentiments[aspect] = {
                    'label': sentiment_result['label'],
                    'score': float(sentiment_result['score']),
                    'weight': aspect_weight,
                    'confidence': float(sentiment_result.get('confidence', 0.0)),
                    'occurrences': review_text.lower().count(aspect.lower())
                }
                
                # Store context for debugging/insights
                aspect_tokens = aspect.lower().split()
                for i in range(len(doc) - len(aspect_tokens) + 1):
                    if all(doc[i + j].text.lower() == aspect_tokens[j] for j in range(len(aspect_tokens))):
                        context_span, is_contrastive, _ = self._get_context_window(doc, i, window_size=7)
                        contextual_sentences.append({
                            'aspect': aspect,
                            'sentence': ' '.join([token.text for token in context_span]),
                            'sentiment': sentiment_result,
                            'position': i,
                            'is_contrastive': is_contrastive
                        })
            
            # Generate key phrases and summary
            key_phrases = self.extract_key_phrases(review_text)
            summary = self.generate_summary(review_text, aspect_sentiments)
            
            # Calculate weighted overall sentiment
            weighted_scores = []
            total_weight = 0
            confidences = []
            
            for aspect, data in aspect_sentiments.items():
                # Normalize score to [-0.5, 0.5] range for weighted average
                normalized_score = (data['score'] - 0.5) * data['weight']
                weighted_scores.append(normalized_score)
                total_weight += data['weight']
                confidences.append(data['confidence'])
            
            # Calculate final scores
            if total_weight > 0:
                final_score = (sum(weighted_scores) / total_weight) + 0.5  # Back to [0, 1] range
                final_score = max(0.0, min(1.0, final_score))  # Clamp to [0, 1]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                
                if final_score > 0.6:
                    label = 'positive'
                elif final_score < 0.4:
                    label = 'negative'
                else:
                    label = 'neutral'
                    
                overall_sentiment = {
                    'label': label,
                    'score': float(final_score),
                    'confidence': float(avg_confidence)
                }
            else:
                # Fallback to simple analysis if weighting fails
                overall_sentiment = self.analyze_sentiment(review_text)
            
            # Format aspect results
            aspect_results = []
            for aspect, data in aspect_sentiments.items():
                aspect_results.append({
                    'aspect': aspect,
                    'sentiment': data['label'],
                    'score': data['score'],
                    'weight': data['weight'],
                    'confidence': data['confidence'],
                    'occurrences': data['occurrences']
                })
            
            # Sort aspects by importance (weight * score magnitude)
            aspect_results.sort(
                key=lambda x: abs(x['score'] - 0.5) * x['weight'], 
                reverse=True
            )
            
            return {
                'review': review_text,
                'overall_sentiment': overall_sentiment,
                'aspects': aspect_results,
                'key_phrases': key_phrases,
                'summary': summary,
                'contextual_sentences': contextual_sentences[:5]  # For debugging
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing review: {str(e)}")
            # Fallback to simple analysis if something goes wrong
            sentiment = self.analyze_sentiment(review_text)
            return {
                'review': review_text,
                'overall_sentiment': {
                    'label': sentiment['label'],
                    'score': float(sentiment['score']),
                    'confidence': float(sentiment.get('confidence', 0.0))
                },
                'aspects': [],
                'key_phrases': [],
                'summary': 'Analysis incomplete due to an error.',
                'contextual_sentences': []
            }

# Singleton instance
analyzer = SemanticReviewAnalyzer()
