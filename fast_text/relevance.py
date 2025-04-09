"""
FastText-based relevance checker for tax/IVA related queries.

This module provides functionality to determine if text is relevant to tax/IVA topics
using a combination of FastText classifier and keyword matching.
"""

import os
import re
import logging
from typing import List, Tuple, Optional, Set, Dict
import streamlit as st

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FastTextRelevanceChecker:
    """Checks if text is relevant to tax/IVA topics using FastText classifier."""
    
    def __init__(self, model_path: str = "fast_text/models/tax_classifier.bin"):
        """Initialize the relevance checker with a FastText model."""
        self.model_path = model_path
        self.model = None
        self.relevant_labels = {"IVA"}
        self.load_model()
        self._conversation_history = []  # Store conversation history for testing
        
        # Keywords and their weights
        self.tax_keywords = {
            'iva': 1.0,
            'fiscale': 0.8,
            'tasse': 0.9,
            'imposte': 0.9,
            'dichiarazione': 0.7,
            'fatture': 0.7,
            'fattura': 0.7,
            'detrarre': 0.8,
            'detrazioni': 0.8,
            'spese': 0.7,
            'pagare': 0.4,
            'forfettario': 0.8,
            'fiscozen': 1.0,
            'partita': 0.6,
            'redditi': 0.8,
            'tributario': 0.8,
            'tributi': 0.8,
            'contributi': 0.6,
            'contribuente': 0.7,
            'agenzia': 0.5,
            'entrate': 0.5,
            'commercialista': 0.7,
            'contabile': 0.7,
            'contabilità': 0.7,
        }
        
        # Phrases and their weights
        self.tax_phrases = {
            "partita iva": 1.0,
            "agenzia delle entrate": 0.9,
            "dichiarazione dei redditi": 0.9,
            "regime forfettario": 0.8,
            "codice fiscale": 0.7,
            "fattura elettronica": 0.8,
            "spese detraibili": 0.8,
            "spese deducibili": 0.8,
            "imposta sul reddito": 0.9,
            "imposta di registro": 0.7,
            "imposta di bollo": 0.7,
            "imposta di successione": 0.7,
            "imposta ipotecaria": 0.7,
            "imposta catastale": 0.7,
            "imposta di donazione": 0.7,
            "imposta di trascrizione": 0.7,
            "imposta di registro": 0.7,
            "imposta di bollo": 0.7,
            "imposta di successione": 0.7,
            "imposta ipotecaria": 0.7,
            "imposta catastale": 0.7,
            "imposta di donazione": 0.7,
            "imposta di trascrizione": 0.7,
        }
    
    def load_model(self):
        """Load the FastText model."""
        try:
            import fasttext
            logger.info(f"Checking for model at: {self.model_path}")
            logger.info(f"Current working directory: {os.getcwd()}")
            
            # Try to find the model file
            if not os.path.exists(self.model_path):
                # Try relative path
                rel_path = os.path.join(os.getcwd(), self.model_path)
                if os.path.exists(rel_path):
                    self.model_path = rel_path
                else:
                    # Try absolute path
                    abs_path = os.path.join("/app", self.model_path)
                    if os.path.exists(abs_path):
                        self.model_path = abs_path
                    else:
                        logger.error(f"❌ Model file not found at {self.model_path}")
                        logger.error(f"Current working directory: {os.getcwd()}")
                        return
            
            logger.info(f"Loading model from: {self.model_path}")
            self.model = fasttext.load_model(self.model_path)
            logger.info("✅ Model loaded successfully")
            
            # Test the model
            test_text = "Come funziona l'IVA?"
            prediction = self.model.predict(test_text)
            logger.info(f"Model test prediction for '{test_text}': {prediction}")
            
        except Exception as e:
            logger.error(f"❌ Error loading FastText model: {e}")
            self.model = None
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for relevance checking."""
        if text is None:
            raise ValueError("Input text cannot be None")
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation except apostrophes
        text = re.sub(r'[^\w\s\']', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _calculate_keyword_score(self, text: str, context_score: float = 0.0) -> Tuple[float, Set[str], Set[str]]:
        """Calculate relevance score based on keywords and phrases."""
        if not text:
            return 0.0, set(), set()
            
        # Initialize variables
        max_score = 0.0
        found_keywords = set()
        found_phrases = set()
        
        # Check for phrases first (they take precedence)
        for phrase, weight in self.tax_phrases.items():
            if phrase in text:
                max_score = max(max_score, weight)
                found_phrases.add(f"{phrase} ({weight:.1f})")
        
        # Check for individual keywords
        words = set(text.split())
        for word in words:
            # Check for exact matches
            if word in self.tax_keywords:
                weight = self.tax_keywords[word]
                max_score = max(max_score, weight)
                found_keywords.add(f"{word} ({weight:.1f})")
            # Check for partial matches (e.g., 'iva' in 'liva')
            elif any(keyword in word for keyword in self.tax_keywords):
                # Use a lower weight for partial matches
                max_score = max(max_score, 0.6)
                found_keywords.add(f"{word} (0.6)")
        
        # Boost score for follow-up questions based on context
        if context_score > 0:
            # If we have a strong context (previous tax-related question)
            # and this is a short follow-up question, boost the score
            if len(words) <= 4 and context_score >= 0.7:
                max_score = max(max_score, context_score * 0.8)
                found_keywords.add("context_boost")
        
        return max_score, found_keywords, found_phrases
    
    def is_relevant(self, text: str, threshold: float = 0.6) -> Tuple[bool, dict]:
        """
        Check if the text is relevant to tax/IVA topics.
        
        Args:
            text: The text to check
            threshold: Confidence threshold for relevance (0.0 to 1.0)
        
        Returns:
            Tuple of (is_relevant, details)
        """
        if text is None:
            raise ValueError("Input text cannot be None")
            
        if not isinstance(text, str):
            raise TypeError(f"Input text must be a string, got {type(text)}")
            
        if not text.strip():
            return False, {'preprocessed_text': '', 'keyword_score': 0.0, 'keywords_found': set(), 'phrases_found': set()}
        
        # Preprocess the text
        processed_text = self._preprocess_text(text)
        
        # Calculate context score from previous messages
        context_score = 0.0
        if len(self._conversation_history) > 1:
            # Get the last few messages for context
            context_window = 3  # Look at last 3 exchanges
            start_idx = max(0, len(self._conversation_history) - context_window)
            recent_context = self._conversation_history[start_idx:-1]  # Exclude current query
            
            # Calculate context relevance from recent messages
            context_scores = []
            for msg in recent_context:
                score, _, _ = self._calculate_keyword_score(self._preprocess_text(msg))
                context_scores.append(score)
            
            # Get the highest context score
            context_score = max(context_scores) if context_scores else 0
        
        # Calculate keyword-based score with context
        keyword_score, found_keywords, found_phrases = self._calculate_keyword_score(processed_text, context_score)
        
        # Initialize result details
        details = {
            'preprocessed_text': processed_text,
            'keyword_score': keyword_score,
            'keywords_found': found_keywords,
            'phrases_found': found_phrases,
            'context_relevance': context_score > 0
        }
        
        # Store the query in conversation history
        self._conversation_history.append(text)
        
        # If we have a strong keyword match, consider it relevant
        if keyword_score >= 0.7 or 'iva' in processed_text or \
           any(phrase in processed_text for phrase in ['partita iva', 'aliquote iva']) or \
           any(kw in processed_text for kw in ['detrarre', 'spese', 'freelancer', 'aliquot']):
            details['combined_score'] = max(keyword_score, 0.8)
            return True, details
        
        # Try model prediction if available
        if self.model:
            # Get model predictions
            predictions = self.model.predict(processed_text, k=-1)
            labels = [label.replace('__label__', '') for label in predictions[0]]
            probs = predictions[1]
            
            # Create a dictionary of label probabilities
            label_probs = {label: prob for label, prob in zip(labels, probs)}
            details['model_predictions'] = label_probs
            
            # Calculate combined score with adjusted weights
            iva_prob = label_probs.get('IVA', 0)
            other_prob = label_probs.get('Other', 0)
            
            # If "Other" probability is very high and no strong keywords, it's not relevant
            if other_prob >= 0.8 and keyword_score < 0.6:
                details['combined_score'] = 0.0
                return False, details
            
            # If IVA probability is decent and we have keywords, it's relevant
            if iva_prob >= 0.4 and keyword_score >= 0.4:
                details['combined_score'] = (iva_prob + keyword_score) / 2
                return True, details
        
        # Default to keyword score if no model available
        details['combined_score'] = keyword_score
        return keyword_score >= threshold, details 