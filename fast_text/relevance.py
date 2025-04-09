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

class FastTextRelevanceChecker:
    """Checks if text is relevant to tax/IVA topics using FastText classifier."""
    
    def __init__(self, model_path: str = None):
        """Initialize the relevance checker with a FastText model."""
        # Set default model path if none provided
        if model_path is None:
            # Try to find the model in different possible locations
            possible_paths = [
                "fast_text/models/tax_classifier.bin",
                "/app/fast_text/models/tax_classifier.bin",
                os.path.join(os.path.dirname(__file__), "models/tax_classifier.bin")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    self.model_path = path
                    break
            else:
                self.model_path = "fast_text/models/tax_classifier.bin"
        else:
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
            'detrarre': 0.8,  # Increased weight
            'detrazioni': 0.8,  # Increased weight
            'spese': 0.7,     # Added for expense-related queries
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
            'rimborso': 0.6,
            'attività': 0.5,  # Added for business-related queries
            'impresa': 0.6,   # Added for business-related queries
            'azienda': 0.6,   # Added for business-related queries
            'freelancer': 0.7, # Added for freelancer queries
            'professionista': 0.7, # Added for professional queries
            'aliquote': 0.8,  # Added for tax rate queries
            'aliquota': 0.8   # Added for tax rate queries
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
            'aprire attività': 0.8,    # Added for business queries
            'gestione attività': 0.8,  # Added for business queries
            'libero professionista': 0.8,  # Added for professional queries
            'spese detraibili': 0.9,   # Added for expense queries
            'spese deducibili': 0.9    # Added for expense queries
        }
    
    def load_model(self) -> None:
        """Load the FastText model from the specified path."""
        try:
            import fasttext
            print(f"Checking for model at: {self.model_path}")
            if os.path.exists(self.model_path):
                print("Model file found, loading...")
                self.model = fasttext.load_model(self.model_path)
                # Test the model with multiple examples
                test_texts = [
                    "Come funziona l'IVA?",
                    "Come funziona l'IVA per un libero professionista?",
                    "Quanto costa aprire un'attività?",
                    "Che tempo farà domani?"
                ]
                for text in test_texts:
                    predictions = self.model.predict(text, k=-1)
                    print(f"Model test prediction for '{text}': {predictions}")
                logging.info(f"Successfully loaded FastText model from {self.model_path}")
            else:
                print(f"❌ Model file not found at {self.model_path}")
                print("Current working directory:", os.getcwd())
                print("Directory contents:", os.listdir(os.path.dirname(self.model_path)))
                logging.warning(f"Model file not found at {self.model_path}")
                self.model = None
        except ImportError as e:
            print(f"❌ Error importing fasttext: {e}")
            logging.error(f"Error importing fasttext: {e}")
            self.model = None
        except Exception as e:
            print(f"❌ Error loading FastText model: {e}")
            logging.error(f"Error loading FastText model: {e}")
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
            'context_relevance': False
        }
        
        # Store the query in conversation history for testing
        self._conversation_history.append(text)
        
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
            
            # 2. Check for queries starting with conjunctions
            conjunctions = {'e', 'ma', 'però', 'oppure', 'invece'}
            first_word = processed_text.split()[0] if processed_text else ''
            if first_word in conjunctions:
                is_followup = True
            
            # 3. Check for incomplete sentences that rely on context
            if len(words) <= 4 and not any(word in self.tax_keywords for word in words):
                is_followup = True
            
            # If this is a follow-up and we have relevant context, consider it relevant
            if is_followup and max_context_score >= 0.6:
                details['context_relevance'] = True
                details['context_score'] = max_context_score
                details['context_keywords'] = context_keywords
                # Boost keyword score for follow-up questions
                details['keyword_score'] = max(keyword_score, max_context_score * 0.8)
                return True, details
        
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