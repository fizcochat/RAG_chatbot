"""
FastText-based relevance checker for tax/IVA related queries.

This module provides functionality to determine if text is relevant to tax/IVA topics
using a combination of FastText classifier and keyword matching.
"""

import os
import re
import logging
import time
from typing import List, Tuple, Optional, Set, Dict
import streamlit as st

# Set NumPy environment variable early, before any imports
os.environ['NUMPY_ARRAY_FUNCTION_LIKE_CURRENT'] = '1'

# Global model instance
_FASTTEXT_MODEL = None
_FASTTEXT_MODEL_PATH = None
_FASTTEXT_LOAD_ATTEMPT_TIME = 0
_FASTTEXT_LOAD_RETRY_INTERVAL = 3600  # 1 hour in seconds

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
    
    def __init__(self, model_path: str = None, lazy_load: bool = True):
        """Initialize the relevance checker with a FastText model.
        
        Args:
            model_path: Path to the FastText model file
            lazy_load: If True, delay loading the model until it's needed
        """
        global _FASTTEXT_MODEL, _FASTTEXT_MODEL_PATH, _FASTTEXT_AVAILABLE
        
        self.model_path = model_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'tax_classifier.bin')
        self.relevant_labels = {"Tax"}
        self._conversation_history = []  # Store conversation history for testing
        self.model = None
        self.lazy_load = lazy_load
        self.model_loaded = False
        
        # Keywords and their weights - Enhanced list for better coverage
        self.tax_keywords = {
            # High confidence keywords
            'iva': 1.0,
            'fiscozen': 1.0,
            'imposta': 0.9,
            'tasse': 0.9,
            'tassa': 0.9,
            'imposte': 0.9,
            'forfettario': 0.9,
            'partita iva': 1.0,
            'p iva': 1.0,
            'p.iva': 1.0,
            
            # Medium confidence keywords
            'fiscale': 0.8,
            'dichiarazione': 0.8,
            'fatture': 0.8,
            'fattura': 0.8,
            'detrarre': 0.8,
            'detrazioni': 0.8,
            'detrazione': 0.8,
            'deduzione': 0.8,
            'deducibile': 0.8,
            'detraibile': 0.8,
            'redditi': 0.8,
            'reddito': 0.8,
            'tributario': 0.8,
            'tributi': 0.8,
            'irpef': 0.9,
            'irap': 0.9,
            'imu': 0.8,
            'inps': 0.8,
            'inail': 0.8,
            '730': 0.9,
            'f24': 0.8,
            'agenzia entrate': 0.9,
            'ade': 0.8,
            'codice fiscale': 0.8,
            'evasione': 0.8,
            'esenzione': 0.8,
            
            # Lower confidence keywords
            'contributi': 0.7,
            'contribuente': 0.7,
            'commercialista': 0.7,
            'contabile': 0.7,
            'contabilità': 0.7,
            'rimborso': 0.7,
            'aliquote': 0.7,
            'aliquota': 0.7,
            'spese': 0.6,
            'pagare': 0.5,
            'pagamento': 0.5,
            'attività': 0.5,
            'impresa': 0.6,
            'azienda': 0.6,
            'freelancer': 0.7,
            'professionista': 0.7,
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
            'spese deducibili': 0.9,
            'fattura elettronica': 0.9,
            'fatturazione elettronica': 0.9,
            'modello 730': 0.9,
            'calcolo imposte': 0.9,
            'scadenze fiscali': 0.9,
            'tassazione': 0.9,
            'conto fiscale': 0.8,
            'detrazione fiscale': 0.9,
            'agevolazione fiscale': 0.9,
            'regime dei minimi': 0.9,
            'flat tax': 0.8,
            'adempimenti fiscali': 0.9,
            'cassetto fiscale': 0.9,
            'bollo fattura': 0.8,
            'ritenuta acconto': 0.8,
            'regime agevolato': 0.8,
            'split payment': 0.8,
            'reverse charge': 0.8
        }
        
        # Tax-related question starters
        self.tax_question_starters = [
            'come funziona', 'come si calcola', 'quanto costa', 'quando scade',
            'cosa serve per', 'chi deve', 'quali sono', 'come posso', 'come si fa',
            'quali documenti', 'come si paga', 'come dichiarare', 'come detrarre'
        ]
        
        # Only load the model now if not using lazy loading
        if not lazy_load:
                    self._load_model()
    
    def _should_attempt_load(self) -> bool:
        """Determine if we should attempt to load the model again after a failure."""
        global _FASTTEXT_LOAD_ATTEMPT_TIME, _FASTTEXT_LOAD_RETRY_INTERVAL
        
        current_time = time.time()
        if current_time - _FASTTEXT_LOAD_ATTEMPT_TIME > _FASTTEXT_LOAD_RETRY_INTERVAL:
            return True
        return False
    
    def _load_model(self) -> bool:
        """Load the FastText model with improved error handling and logging.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        global _FASTTEXT_MODEL, _FASTTEXT_MODEL_PATH, _FASTTEXT_AVAILABLE
        global _FASTTEXT_LOAD_ATTEMPT_TIME
        
        # Update the load attempt time
        _FASTTEXT_LOAD_ATTEMPT_TIME = time.time()
        
        if not _FASTTEXT_AVAILABLE:
            print("FastText module not available. Cannot load model.")
            return False
            
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
                return False
            
            print("Model file found, loading...")
            try:
                # Safely try to load the model
                import numpy as np
                old_numpy_array_function_like = os.environ.get('NUMPY_ARRAY_FUNCTION_LIKE_CURRENT')
                os.environ['NUMPY_ARRAY_FUNCTION_LIKE_CURRENT'] = '1'
                
                # Import with patched setting
                self.model = fasttext.load_model(self.model_path)
                
                # Restore original setting
                if old_numpy_array_function_like:
                    os.environ['NUMPY_ARRAY_FUNCTION_LIKE_CURRENT'] = old_numpy_array_function_like
                else:
                    os.environ.pop('NUMPY_ARRAY_FUNCTION_LIKE_CURRENT', None)
                
            except Exception as numpy_error:
                print(f"Error with NumPy compatibility: {str(numpy_error)}")
                print("Using keyword-based relevance checking only.")
                return False
            
            # Store model in global variable
            _FASTTEXT_MODEL = self.model
            _FASTTEXT_MODEL_PATH = self.model_path
            
            print("Model loaded successfully")
            self.model_loaded = True
            
            # Don't test the model to avoid NumPy issues
            return True
            
        except Exception as e:
            print(f"Error loading FastText model: {str(e)}")
            print(f"Model path: {self.model_path}")
            print(f"Current working directory: {os.getcwd()}")
            self.model = None
            self.model_loaded = False
            return False
    
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
        
        # Check for tax question patterns (e.g., "Come funziona l'IVA?")
        for starter in self.tax_question_starters:
            if text.startswith(starter):
                # Check if any tax keyword follows the starter pattern
                remaining_text = text[len(starter):].strip()
                if any(keyword in remaining_text for keyword in self.tax_keywords):
                    max_score = max(max_score, 0.8)  # Higher weight for well-formed tax questions
                    found_phrases.add(f"tax question pattern ({0.8:.1f})")
        
        # Check for phrases first (they take precedence)
        for phrase, weight in self.tax_phrases.items():
            if phrase in text:
                max_score = max(max_score, weight)
                found_phrases.add(f"{phrase} ({weight:.1f})")
        
        # Check for individual keywords
        for keyword, weight in self.tax_keywords.items():
            if keyword in text:
                max_score = max(max_score, weight)
                found_keywords.add(f"{keyword} ({weight:.1f})")
        
        return max_score, found_keywords, found_phrases
    
    def _fasttext_predict(self, text: str) -> Tuple[bool, float]:
        """Use FastText model to predict if text is relevant."""
        # Check if we should load the model
        if self.lazy_load and not self.model_loaded:
            if not self._load_model() and not self._should_attempt_load():
                # If loading failed and we shouldn't retry yet
                print("Using keyword-based classification since FastText model is not loaded")
                return False, 0.0
                
        # If the model is still not loaded, give up and use keywords only
        if not self.model:
            return False, 0.0
            
        try:
            # Make sure NumPy array functions work correctly
            import os
            os.environ['NUMPY_ARRAY_FUNCTION_LIKE_CURRENT'] = '1'

            # Get prediction
            labels, scores = self.model.predict(text)
            
            # Extract the label and score
            label = labels[0].replace("__label__", "")
            score = float(scores[0])
            
            # Map the score to a relevance score (0-1)
            is_relevant = label in self.relevant_labels
            return is_relevant, score
            
        except Exception as e:
            print(f"Error in FastText prediction: {str(e)}")
            # For tax-related terms, default to True
            if any(term in text.lower() for term in ['tass', 'fiscale', 'iva', 'fiscozen', 'partita']):
                return True, 0.75
            return False, 0.0
    
    def is_relevant(self, text: str, threshold: float = 0.6) -> Tuple[bool, dict]:
        """Determine if text is relevant to tax/IVA topics.
        
        Args:
            text: The text to check for relevance
            threshold: The minimum relevance score to consider text relevant
            
        Returns:
            Tuple of (is_relevant, details_dict)
        """
        try:
            # Preprocess the text
            processed_text = self._preprocess_text(text)
        
            # Calculate keyword score
            keyword_score, found_keywords, found_phrases = self._calculate_keyword_score(processed_text)
        
            # Initialize result dictionary
            result = {
                "keyword_score": keyword_score,
                "keywords": list(found_keywords),
                "phrases": list(found_phrases),
                "threshold": threshold,
                "is_relevant": False,
                "fasttext": {"used": False, "score": 0.0}
            }
            
            # If keyword score is high enough, mark as relevant immediately
            if keyword_score >= 0.6:  # Lowered from 0.9/0.7 to be even more permissive
                result["is_relevant"] = True
                result["final_score"] = keyword_score
                result["determination_method"] = "high confidence keywords"
                return True, result
            
            # If we have the FastText model, use it for additional classification
            fasttext_result, fasttext_score = self._fasttext_predict(processed_text)
            result["fasttext"] = {
                "used": self.model is not None,
                "score": fasttext_score,
                "prediction": fasttext_result
                }
        
            # Calculate final score - weighted combination of keyword and FastText scores
            if self.model:
                # If we have the model, use a combination
                final_score = (keyword_score * 0.6) + (fasttext_score * 0.4)
            else:
                # Otherwise rely only on keywords, but with a more generous threshold
                # For questions about taxes, fiscal matters, or Fiscozen services
                contains_tax_terms = any(term in processed_text for term in ['tass', 'fisca', 'iva', 'fiscozen', 'partita', 'forfet', 'detra', 'dedu', 'imposta'])
                if contains_tax_terms and keyword_score >= 0.4:  # Lower threshold for tax terms
                    final_score = max(keyword_score + 0.2, 0.6)  # Boost score for tax terms
                else:
                    final_score = keyword_score
                
            # Set the final score
            result["final_score"] = final_score
            
            # Determine if the text is relevant
            is_relevant = final_score >= threshold
            result["is_relevant"] = is_relevant
            
            # Set determination method
            if self.model:
                result["determination_method"] = "combined keyword and FastText"
            else:
                result["determination_method"] = "keywords only"
                
            return is_relevant, result
            
        except Exception as e:
            print(f"Error in relevance checking: {str(e)}")
            # Default to True with an error message (we assume it's relevant by default when in doubt)
            return True, {
                "error": str(e),
                "is_relevant": True,  # Default to relevant to avoid blocking valid queries
                "final_score": 0.7,
                "threshold": threshold
            } 