"""
FastText-based relevance checker for the Fiscozen chatbot.
This module provides functionality to determine if a user query is related to tax matters,
specifically IVA and Fiscozen topics.
"""

import os
import re
import random
import fasttext
import numpy as np
from pathlib import Path
import tempfile
import shutil


class FastTextRelevanceChecker:
    """
    A class that uses FastText to determine if a user's query is relevant to tax matters.
    FastText is more lightweight and faster than BERT, while still providing good classification results.
    """

    def __init__(self, model_path=None):
        """
        Initialize the FastText relevance checker.
        
        Args:
            model_path (str, optional): Path to a pre-trained FastText model. 
                                        If None, a default model will be used or initialized.
        """
        self.model_path = model_path
        self.model = None
        self.labels = ["IVA", "Fiscozen", "Other"]
        
        # Create model directory if it doesn't exist
        if self.model_path:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        else:
            # Default path
            self.model_path = "fasttext/models/tax_classifier.bin"
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Load the model if it exists
        self._load_model()
    
    def _load_model(self):
        """Load the FastText model if it exists, or create a default model."""
        try:
            if os.path.exists(self.model_path):
                print(f"Loading FastText model from {self.model_path}")
                self.model = fasttext.load_model(self.model_path)
                return True
            else:
                print(f"No FastText model found at {self.model_path}. Model will need to be trained.")
                return False
        except Exception as e:
            print(f"Error loading FastText model: {e}")
            return False
    
    def preprocess_text(self, text):
        """
        Clean and normalize text for better relevance detection.
        
        Args:
            text (str): The input text to preprocess.
            
        Returns:
            str: The preprocessed text.
        """
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Replace common abbreviations and variants
        replacements = {
            "iva's": "iva",
            "i.v.a": "iva",
            "i.v.a.": "iva",
            "fiscozen's": "fiscozen",
            "fisco zen": "fiscozen",
            "fisco-zen": "fiscozen",
            "fisco zen's": "fiscozen",
            "v.a.t": "vat",
            "v.a.t.": "vat"
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def check_relevance(self, text, tax_threshold=0.5):
        """
        Check if the given text is relevant to tax matters.
        
        Args:
            text (str): The input text to check.
            tax_threshold (float, optional): The threshold above which a text is considered tax-related.
            
        Returns:
            dict: A dictionary with relevance information.
        """
        # If model is not loaded, return a default response
        if self.model is None:
            print("FastText model not loaded. Please train the model first.")
            return {
                "is_relevant": False,
                "topic": "Unknown",
                "tax_related_probability": 0.0,
                "confidence": 0.0,
                "probabilities": {"IVA": 0.0, "Fiscozen": 0.0, "Other": 1.0}
            }
        
        # Preprocess the text
        processed_text = self.preprocess_text(text)
        
        # FastText requires text to end with a newline for predict_proba
        if not processed_text.endswith('\n'):
            processed_text += '\n'
        
        # Get predictions
        labels, probabilities = self.model.predict(processed_text, k=3)
        
        # Process the results
        label_probs = {}
        for i, label in enumerate(labels):
            # FastText labels are in format '__label__X', so we extract X
            label_name = label.replace('__label__', '')
            label_probs[label_name] = float(probabilities[i])
        
        # Ensure all expected labels have a probability
        for label in self.labels:
            if label not in label_probs:
                label_probs[label] = 0.0
        
        # Determine the most likely topic
        topic = max(label_probs, key=lambda k: label_probs[k])
        
        # Calculate tax-related probability (IVA + Fiscozen)
        tax_related_prob = label_probs.get("IVA", 0.0) + label_probs.get("Fiscozen", 0.0)
        
        # Determine if the text is relevant based on the threshold
        is_relevant = tax_related_prob >= tax_threshold
        
        # Calculate confidence (probability of the predicted class)
        confidence = label_probs.get(topic, 0.0)
        
        return {
            "is_relevant": is_relevant,
            "topic": topic,
            "tax_related_probability": tax_related_prob,
            "confidence": confidence,
            "probabilities": label_probs
        }
    
    def train_with_data(self, training_data, output_path=None, epochs=20, lr=0.1):
        """
        Train the FastText model with custom data.
        
        Args:
            training_data (list): List of (text, label) tuples.
            output_path (str, optional): Path to save the model.
            epochs (int, optional): Number of training epochs.
            lr (float, optional): Learning rate.
            
        Returns:
            bool: True if training was successful, False otherwise.
        """
        if not training_data:
            print("No training data provided.")
            return False
        
        try:
            # Create a temporary file for training data
            with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
                for text, label in training_data:
                    # FastText expects format: __label__LABEL text
                    temp_file.write(f"__label__{label} {text}\n")
                temp_file_path = temp_file.name
            
            # Train the model
            print(f"Training FastText model with {len(training_data)} examples...")
            self.model = fasttext.train_supervised(
                input=temp_file_path,
                epoch=epochs,
                lr=lr,
                wordNgrams=2,
                verbose=2,
                minCount=1
            )
            
            # Save the model
            if output_path:
                self.model.save_model(output_path)
                self.model_path = output_path
                print(f"Model saved to {output_path}")
            else:
                self.model.save_model(self.model_path)
                print(f"Model saved to {self.model_path}")
            
            # Clean up
            os.unlink(temp_file_path)
            
            return True
        
        except Exception as e:
            print(f"Error training FastText model: {e}")
            # Clean up
            if 'temp_file_path' in locals():
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            return False
    
    def save_model(self, output_path=None):
        """
        Save the FastText model.
        
        Args:
            output_path (str, optional): Path to save the model.
            
        Returns:
            bool: True if saving was successful, False otherwise.
        """
        if self.model is None:
            print("No model to save.")
            return False
        
        try:
            if output_path:
                self.model.save_model(output_path)
                print(f"Model saved to {output_path}")
            else:
                self.model.save_model(self.model_path)
                print(f"Model saved to {self.model_path}")
            return True
        except Exception as e:
            print(f"Error saving FastText model: {e}")
            return False 