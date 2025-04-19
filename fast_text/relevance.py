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

# Global model instance
_FASTTEXT_MODEL = None
_FASTTEXT_MODEL_PATH = None

# Global flag to indicate if FastText is available
_FASTTEXT_AVAILABLE = False

# Try to import FastText, but don't fail if not available
try:
    import fasttext
    _FASTTEXT_AVAILABLE = True
except ImportError:
    print("FastText not available. Using keyword-based relevance checking only.")
    _FASTTEXT_AVAILABLE = False

class FastTextRelevanceChecker:
    """Checks if text is relevant to tax/IVA topics using FastText classifier."""
    
    def __init__(self, model_path: str = None):
        """Initialize the relevance checker with a FastText model."""
        global _FASTTEXT_MODEL, _FASTTEXT_MODEL_PATH, _FASTTEXT_AVAILABLE
        
        self.model_path = model_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'tax_classifier.bin')
        self.relevant_labels = {"IVA"}
        self._conversation_history = []  # Store conversation history for testing
        self.model = None
        
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
            'esenzione': 0.8,
            'evasione': 0.8,
            'rimborso': 0.6,
            'attività': 0.5,
            'impresa': 0.6,
            'azienda': 0.6,
            'freelancer': 0.7,
            'professionista': 0.7,
            'aliquote': 0.8,
            'aliquota': 0.8
        }
        
        # Tax-related phrases and their weights
        self.tax_phrases = {
            'partita iva': 1.0,
            'regime forfettario': 1.0,
            'dichiarazione redditi': 1.0,
            'agenzia delle entrate': 0.9,
            'agenzia entrate': 0.9,
            'aliquote iva': 1.0,
            'carico fiscale': 0.9,
            'imposta sul valore': 1.0,
            'valore aggiunto': 0.9,
            'codice fiscale': 0.8,
            'sistema tributario': 0.9,
            'evasione fiscale': 0.9,
            'consulenza fiscale': 0.9,
            'gestione fiscale': 0.9,
            'contabilità aziendale': 0.8,
            'rimborso iva': 0.9,
            'credito iva': 0.9,
            'debito iva': 0.9,
            'aprire attività': 0.8,
            'gestione attività': 0.8,
            'libero professionista': 0.8,
            'spese detraibili': 0.9,
            'spese deducibili': 0.9
        }
        
        # Try to load the model, but don't fail if it's not available
        if _FASTTEXT_AVAILABLE:
            try:
                # Check if model is already loaded globally
                if _FASTTEXT_MODEL is not None and _FASTTEXT_MODEL_PATH == self.model_path:
                    self.model = _FASTTEXT_MODEL
                    print("Using already loaded FastText model")
                else:
                    self._load_model()
            except Exception as e:
                print(f"Warning: Could not load FastText model: {str(e)}")
                print("Continuing with keyword-based relevance checking only")
        else:
            print("FastText not available. Using keyword-based relevance checking only.")
    
    def _load_model(self):
        """Load the FastText model with improved error handling and logging."""
        global _FASTTEXT_MODEL, _FASTTEXT_MODEL_PATH, _FASTTEXT_AVAILABLE
        
        if not _FASTTEXT_AVAILABLE:
            print("FastText module not available. Cannot load model.")
            return
            
        try:
            print(f"Attempting to load model from: {self.model_path}")
            if not os.path.exists(self.model_path):
                print(f"Model file not found at: {self.model_path}")
                print(f"Current working directory: {os.getcwd()}")
                model_dir = os.path.dirname(self.model_path)
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir, exist_ok=True)
                    print(f"Created directory: {model_dir}")
                print("Model not found. Will use keyword-based relevance checking.")
                return
            
            print("Model file found, loading...")
            self.model = fasttext.load_model(self.model_path)
            
            # Store model in global variable
            _FASTTEXT_MODEL = self.model
            _FASTTEXT_MODEL_PATH = self.model_path
            
            print("Model loaded successfully")
            
            # Test the model
            test_text = "Come funziona l'IVA?"
            prediction = self.model.predict(test_text)
            print(f"Test prediction for '{test_text}': {prediction}")
            
        except Exception as e:
            print(f"Error loading FastText model: {str(e)}")
            print(f"Model path: {self.model_path}")
            print(f"Current working directory: {os.getcwd()}")
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
    
    def _calculate_keyword_score(self, text: str) -> Tuple[float, Set[str], Set[str]]:
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
        
        return max_score, found_keywords, found_phrases
    
    def _fasttext_predict(self, text: str) -> Tuple[bool, float]:
        """Make a prediction using the FastText model."""
        if not self.model or not _FASTTEXT_AVAILABLE:
            return False, 0.0
            
        try:
            # Get prediction from FastText model
            labels, probabilities = self.model.predict(text)
            
            # Extract label and probability
            label = labels[0].replace("__label__", "")
            prob = probabilities[0]
            
            # Check if label is in relevant_labels
            is_relevant = label in self.relevant_labels
            
            return is_relevant, float(prob)
        except Exception as e:
            print(f"FastText prediction error: {e}")
            return False, 0.0
    
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
        
        # Calculate keyword-based score
        keyword_score, found_keywords, found_phrases = self._calculate_keyword_score(processed_text)
        
        # Initialize result details
        details = {
            'preprocessed_text': processed_text,
            'keyword_score': keyword_score,
            'keywords_found': found_keywords,
            'phrases_found': found_phrases,
            'model_available': self.model is not None and _FASTTEXT_AVAILABLE,
            'context_relevance': False
        }
        
        # Store the query in conversation history for testing
        self._conversation_history.append(text)
        
        # Make FastText prediction if model is available
        if self.model and _FASTTEXT_AVAILABLE:
            fasttext_relevant, fasttext_confidence = self._fasttext_predict(processed_text)
            details['fasttext_relevant'] = fasttext_relevant
            details['fasttext_confidence'] = fasttext_confidence
            
            # Combine FastText and keyword relevance
            if fasttext_relevant and fasttext_confidence > 0.7:
                return True, details
        
        # Check conversation context
        if len(self._conversation_history) > 1:
            # Get the last few messages for context
            context_window = 3  # Look at last 3 exchanges
            start_idx = max(0, len(self._conversation_history) - context_window)
            recent_context = self._conversation_history[start_idx:-1]  # Exclude current query
            
            # Calculate context relevance from recent messages
            context_scores = []
            context_keywords = set()
            for msg in recent_context:
                score, keywords, phrases = self._calculate_keyword_score(self._preprocess_text(msg))
                if score >= 0.6:  # If any recent message was tax-related
                    context_scores.append(score)
                    context_keywords.update(keywords)
            
            # Get the highest context score
            max_context_score = max(context_scores) if context_scores else 0
            
            # Check if this is a follow-up question
            is_followup = False
            
            # 1. Check for short queries with question words
            question_words = {'come', 'quanto', 'quali', 'dove', 'quando', 'perché', 'chi', 'cosa', 'e', 'ma', 'per'}
            words = set(processed_text.split())
            if words.intersection(question_words) and len(words) <= 6:
                is_followup = True
                
            # 2. Check for pronouns and articles 
            pronoun_words = {'lo', 'la', 'li', 'le', 'quello', 'questa', 'questo', 'suo', 'loro', 'mio', 'tuo', 'lui', 'lei', 'essa', 'esso', 'egli', 'essi'}
            if words.intersection(pronoun_words):
                is_followup = True
                
            # If recent messages were tax-related and this looks like a follow-up, consider it relevant
            if max_context_score >= 0.7 and is_followup:
                details['context_relevance'] = True
                return True, details
        
        # Fall back to keyword-based classification
        is_relevant = keyword_score >= threshold
        return is_relevant, details 