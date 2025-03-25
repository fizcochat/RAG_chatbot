"""
Model initialization script for Fiscozen Chatbot

This script downloads and sets up the BERT model for relevance checking.
Run this before starting the chatbot to ensure the model is properly initialized.
"""

import os
import sys
from transformers import BertTokenizer, BertForSequenceClassification

def initialize_bert_model():
    """Download and set up the BERT model for relevance checking"""
    model_path = "models/enhanced_bert"
    
    # Create directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)
    print(f"✓ Created directory {model_path}")
    
    # Check if model already exists
    if os.path.exists(os.path.join(model_path, "config.json")):
        print(f"✓ Model already exists at {model_path}")
        return True
    
    try:
        print("⏳ Downloading BERT model from Hugging Face...")
        # Initialize the tokenizer and model
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
        
        # Save the model and tokenizer
        print(f"⏳ Saving model to {model_path}...")
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        
        print(f"✓ Model successfully initialized and saved to {model_path}")
        return True
    except Exception as e:
        print(f"✗ Error initializing model: {e}")
        return False

if __name__ == "__main__":
    print("\n=== Fiscozen Chatbot Model Initialization ===\n")
    success = initialize_bert_model()
    
    if success:
        print("\n✓ Model initialization complete! You can now run the chatbot.")
        print("   Run: python run_prod.py\n")
    else:
        print("\n✗ Model initialization failed. Please check the error message above.")
        sys.exit(1) 