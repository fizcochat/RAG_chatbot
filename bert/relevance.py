"""
BERT-based relevance checker for tax-related queries.
This module provides a RelevanceChecker class that can determine if user queries 
are relevant to tax matters, specifically for IVA and Fiscozen topics.
"""

import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import numpy as np
import re
from typing import Dict, List, Any, Optional


class RelevanceChecker:
    """A class for checking if messages are relevant to tax matters with advanced detection methods"""
    
    def __init__(self, model_path=None):
        """
        Initialize the relevance checker
        
        Args:
            model_path: Path to the model directory. If None, will use default path.
        """
        # Use default path if none provided
        if model_path is None:
            model_path = "bert/models/enhanced_bert"
        
        # Initialize the tokenizer and model
        try:
            if os.path.exists(model_path):
                print(f"Loading model from {model_path}")
                self.tokenizer = BertTokenizer.from_pretrained(model_path)
                self.model = BertForSequenceClassification.from_pretrained(model_path)
            else:
                print(f"Model path {model_path} not found.")
                print("Falling back to default BERT model from Hugging Face")
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                self.model = BertForSequenceClassification.from_pretrained(
                    'bert-base-uncased', num_labels=3
                )
                
                # Create directory and save the model for future use
                os.makedirs(model_path, exist_ok=True)
                print(f"Created directory {model_path}")
                self.save_model(model_path)
                print(f"Saved default model to {model_path} for future use")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to default BERT model from Hugging Face")
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased', num_labels=3
            )
        
        # Mapping from index to topic
        self.topics = {
            0: "IVA",
            1: "Fiscozen",
            2: "Other"
        }
        
        # Move model to device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def preprocess_text(self, text: str) -> str:
        """
        Clean and normalize text for better relevance detection
        
        Args:
            text: The input text to clean
            
        Returns:
            Cleaned and normalized text
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
            "v.a.t.": "vat",
            "partita iva": "partita iva",
            "p. iva": "partita iva",
            "p.iva": "partita iva",
            "imposta sul valore aggiunto": "iva"
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
        
    def check_relevance(self, text: str, tax_threshold: float = 0.6, apply_preprocessing: bool = True) -> Dict[str, Any]:
        """
        Check if the text is relevant to tax matters using combined tax probability
        
        Args:
            text: The text to check
            tax_threshold: Threshold for combined tax-related probabilities
            apply_preprocessing: Whether to preprocess the text before checking
            
        Returns:
            Dictionary with results:
                is_relevant: Whether the text is relevant to tax matters
                topic: The predicted topic ("IVA", "Fiscozen", or "Other")
                confidence: Confidence score
                tax_related_probability: Combined probability of tax classes
                probabilities: Raw class probabilities
        """
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Optionally preprocess the text
        if apply_preprocessing:
            text = self.preprocess_text(text)
        
        # Tokenize the text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            
        # Convert to numpy for easier handling
        probs = probs.cpu().numpy()[0]
        
        # Get the predicted class and confidence
        pred_class = np.argmax(probs)
        confidence = probs[pred_class]
        
        # Calculate the combined probability of tax-related classes (IVA and Fiscozen)
        tax_related_prob = probs[0] + probs[1]  # Sum of IVA and Fiscozen probabilities
        
        # Determine if the query is relevant to tax matters
        # A query is relevant if the combined probability of tax-related classes exceeds the threshold
        is_relevant = tax_related_prob >= tax_threshold
        
        # If it's relevant, determine the specific topic
        if is_relevant:
            # If one tax class has higher probability, use that as the topic
            if probs[0] > probs[1]:
                topic = self.topics[0]  # IVA
                topic_confidence = probs[0]
            else:
                topic = self.topics[1]  # Fiscozen
                topic_confidence = probs[1]
        else:
            topic = self.topics[2]  # Other
            topic_confidence = probs[2]
        
        return {
            "is_relevant": is_relevant,
            "topic": topic,
            "confidence": float(topic_confidence),
            "tax_related_probability": float(tax_related_prob),
            "probabilities": {
                "IVA": float(probs[0]),
                "Fiscozen": float(probs[1]),
                "Other": float(probs[2])
            }
        }
    
    def train_with_data(self, texts, labels, batch_size=16, epochs=3):
        """
        Train the model with custom data
        
        Args:
            texts: List of text samples
            labels: List of corresponding labels (0=IVA, 1=Fiscozen, 2=Other)
            batch_size: Batch size for training
            epochs: Number of training epochs
        """
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import AdamW
        
        # Ensure model is in training mode
        self.model.train()
        
        # Tokenize all texts
        encodings = self.tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
        
        # Create dataset and dataloader
        dataset = TensorDataset(
            encodings["input_ids"], 
            encodings["attention_mask"], 
            torch.tensor(labels)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer
        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in dataloader:
                # Get batch and move to device
                b_input_ids, b_attention_mask, b_labels = [b.to(self.device) for b in batch]
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(
                    input_ids=b_input_ids,
                    attention_mask=b_attention_mask,
                    labels=b_labels
                )
                
                # Get loss and perform backprop
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Print epoch results
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")
        
        # Move model back to CPU for saving
        self.model.to(torch.device('cpu'))
        
        print("Training completed!")
    
    def save_model(self, output_dir):
        """
        Save the model and tokenizer to the specified directory
        
        Args:
            output_dir: Directory to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")

# Example usage
if __name__ == "__main__":
    # Create the relevance checker
    checker = RelevanceChecker()
    
    # Option 1: Test with individual messages
    print("\nTesting individual messages:")
    test_texts = [
        "What is the current IVA rate in Italy?",
        "Can you help me with my Fiscozen account?",
        "What's the weather like today in Rome?",
        "How do I register for IVA?",
        "What services does Fiscozen offer?",
        "I need to book a flight to Milan"
    ]
    
    for text in test_texts:
        result = checker.check_relevance(text)
        relevance = "Relevant" if result["is_relevant"] else "Not relevant"
        print(f"\nText: '{text}'")
        print(f"Result: {relevance} ({result['topic']}, confidence: {result['confidence']:.4f})")
        print(f"Tax-related probability: {result['tax_related_probability']:.4f}")
        print(f"Probabilities: IVA: {result['probabilities']['IVA']:.4f}, "
              f"Fiscozen: {result['probabilities']['Fiscozen']:.4f}, "
              f"Other: {result['probabilities']['Other']:.4f}")