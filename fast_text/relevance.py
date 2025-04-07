"""
FastText-based relevance checker for tax/IVA related queries.

This module provides functionality to determine if text is relevant to tax/IVA topics
using a FastText classifier.
"""

import os
import re
import logging
from typing import List, Tuple, Optional, Set

class FastTextRelevanceChecker:
    """Class to determine if text is relevant to tax/IVA topics using FastText classifier."""
    
    def __init__(self, model_path: str = "fast_text/models/tax_classifier.bin"):
        """
        Initialize the FastTextRelevanceChecker.
        
        Args:
            model_path (str): Path to the FastText model file
        """
        self.model_path = model_path
        self.model = None
        self.relevant_labels = {"IVA", "Fiscozen"}
        self.load_model()
        
        # Keywords for heuristic fallback
        self._tax_keywords = {
            'iva', 'tax', 'tassa', 'tasse', 'fiscale', 'fiscali', 'imposta',
            'imposte', 'fattura', 'fatture', 'detrazioni', 'rimborso',
            'dichiarazione', 'partita iva', 'p.iva', 'piva', 'fiscozen'
        }
    
    def load_model(self) -> None:
        """
        Load the FastText model from the specified path.
        
        If the model fails to load, logs an error and continues without the model.
        """
        try:
            import fasttext
            if os.path.exists(self.model_path):
                self.model = fasttext.load_model(self.model_path)
                logging.info(f"Successfully loaded FastText model from {self.model_path}")
            else:
                logging.warning(f"Model file not found at {self.model_path}")
        except Exception as e:
            logging.error(f"Error loading FastText model: {e}")
            self.model = None
    
    def train_with_data(self, training_data: List[Tuple[str, str]], save: bool = True) -> bool:
        """
        Train the FastText model with provided data.
        
        Args:
            training_data (List[Tuple[str, str]]): List of (text, label) pairs
            save (bool): Whether to save the model after training
        
        Returns:
            bool: True if training was successful, False otherwise
        """
        try:
            import fasttext
            import tempfile
            
            # Create temporary training file
            with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
                for text, label in training_data:
                    temp_file.write(f"__label__{label} {text}\n")
                temp_file_path = temp_file.name
            
            # Train model
            self.model = fasttext.train_supervised(
                input=temp_file_path,
                epoch=10,
                lr=0.1,
                wordNgrams=2,
                verbose=1,
                minCount=1
            )
            
            # Save model if requested
            if save:
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                self.model.save_model(self.model_path)
            
            # Cleanup
            os.unlink(temp_file_path)
            return True
            
        except Exception as e:
            logging.error(f"Error training FastText model: {e}")
            if 'temp_file_path' in locals():
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            return False
    
    def is_relevant(self, text: str, threshold: float = 0.5) -> bool:
        """
        Check if the given text is relevant to tax/IVA topics.
        
        Args:
            text (str): Text to check for relevance
            threshold (float): Confidence threshold for FastText predictions
        
        Returns:
            bool: True if the text is relevant, False otherwise
        """
        # Preprocess text
        processed_text = self._preprocess_text(text)
        
        # Try using FastText model first
        if self.model is not None:
            try:
                labels, probs = self.model.predict(processed_text, k=1)
                label = labels[0].replace('__label__', '')
                prob = probs[0]
                
                return label in self.relevant_labels and prob >= threshold
            except Exception as e:
                logging.warning(f"Error using FastText model for prediction: {e}")
        
        # Fallback to heuristic method if model fails or isn't available
        return self._is_relevant_heuristic(processed_text)
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for relevance checking.
        
        Args:
            text (str): Text to preprocess
        
        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text
    
    def _is_relevant_heuristic(self, text: str) -> bool:
        """
        Use keyword matching to determine if text is relevant.
        
        Args:
            text (str): Preprocessed text to check
        
        Returns:
            bool: True if the text contains relevant keywords
        """
        words = set(text.split())
        return bool(words & self._tax_keywords) 