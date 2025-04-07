"""
FastText-based relevance checker for tax/IVA related queries.

This module provides functionality to determine if text is relevant to tax/IVA topics
using a combination of FastText classifier and keyword matching.
"""

import os
import re
import logging
from typing import List, Tuple, Optional, Set, Dict

class FastTextRelevanceChecker:
    """Checks if text is relevant to tax/IVA topics using FastText classifier."""
    
    def __init__(self, model_path: str = "fast_text/models/tax_classifier.bin"):
        """Initialize the relevance checker with a FastText model."""
        self.model_path = model_path
        self.model = None
        self.relevant_labels = {"IVA"}
        self.load_model()
        
        # Keywords and their weights
        self.tax_keywords = {
            'iva': 1.0,
            'fiscale': 0.8,
            'tasse': 0.9,
            'imposte': 0.9,
            'dichiarazione': 0.7,
            'fatture': 0.7,
            'fattura': 0.7,
            'detrarre': 0.6,
            'detrazioni': 0.6,
            'pagare': 0.4,
            'forfettario': 0.8,
            'fiscozen': 1.0,
            'partita': 0.6,  # Usually part of "partita IVA"
            'redditi': 0.8,
            'tributario': 0.8,
            'tributi': 0.8,
            'contributi': 0.6,
            'contribuente': 0.7,
            'agenzia': 0.5,  # Usually part of "Agenzia delle Entrate"
            'entrate': 0.5,  # Usually part of "Agenzia delle Entrate"
            'commercialista': 0.7,
            'contabile': 0.7,
            'contabilità': 0.7,
            'esenzione': 0.8,
            'evasione': 0.8,
            'rimborso': 0.6
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
            'debito iva': 0.9
        }
    
    def load_model(self) -> None:
        """Load the FastText model from the specified path."""
        try:
            import fasttext
            if os.path.exists(self.model_path):
                self.model = fasttext.load_model(self.model_path)
                logging.info(f"Successfully loaded FastText model from {self.model_path}")
            else:
                logging.warning(f"Model file not found at {self.model_path}")
                self.model = None
        except Exception as e:
            logging.error(f"Error loading FastText model: {e}")
            self.model = None
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for relevance checking."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation except apostrophes
        text = re.sub(r'[^\w\s\']', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _calculate_keyword_score(self, text: str) -> Tuple[float, Set[str], Set[str]]:
        """Calculate relevance score based on keywords and phrases."""
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
            if word in self.tax_keywords:
                weight = self.tax_keywords[word]
                max_score = max(max_score, weight)
                found_keywords.add(f"{word} ({weight:.1f})")
        
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
        # Preprocess the text
        processed_text = self._preprocess_text(text)
        
        # Calculate keyword-based score
        keyword_score, found_keywords, found_phrases = self._calculate_keyword_score(processed_text)
        
        # Initialize result details
        details = {
            'preprocessed_text': processed_text,
            'keyword_score': keyword_score,
            'keywords_found': found_keywords,
            'phrases_found': found_phrases
        }
        
        # If we have a strong keyword match, consider it relevant
        if keyword_score >= 0.9:
            details['combined_score'] = keyword_score
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
            
            # Calculate combined score
            iva_prob = label_probs.get('IVA', 0)
            other_prob = label_probs.get('Other', 0)
            
            # If keyword score is significant (>= 0.5) and model is somewhat confident
            if keyword_score >= 0.5 and iva_prob >= 0.3:
                combined_score = (keyword_score + iva_prob) / 2
                details['combined_score'] = combined_score
                return combined_score >= threshold, details
            
            # If model is very confident about "Other"
            if other_prob >= 0.7:
                details['combined_score'] = 1 - other_prob
                return False, details
            
            # If model is very confident about "IVA"
            if iva_prob >= 0.9:
                details['combined_score'] = iva_prob
                return True, details
            
            # Default case: use weighted combination
            combined_score = (keyword_score * 0.6 + iva_prob * 0.4)
            details['combined_score'] = combined_score
            return combined_score >= threshold, details
        
        # If no model available, fall back to keyword matching
        details['combined_score'] = keyword_score
        return keyword_score >= threshold, details 