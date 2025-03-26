#!/usr/bin/env python3
"""
BERT Classifier Training Script

This script trains the BERT relevance classifier using documents from:
1. data_documents/ - General document data
2. argilla_data_49/ - Argilla-specific document data

The trained model is saved to bert/models/enhanced_bert for use in the chatbot.
"""

import os
import re
import glob
import random
import pandas as pd
import PyPDF2
from tqdm import tqdm
from bert.relevance import RelevanceChecker

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + " "
            return text.strip()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def preprocess_text(text):
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def categorize_document(filename):
    """Categorize a document based on its filename"""
    filename = filename.lower()
    
    if 'iva' in filename or 'vat' in filename or 'tax' in filename:
        return 0  # IVA category
    elif 'fiscozen' in filename or 'fisco' in filename:
        return 1  # Fiscozen category
    else:
        return 2  # Other category

def extract_training_data():
    """Extract training data from document folders"""
    print("Extracting training data from documents...")
    
    texts = []
    labels = []
    
    # Process data_documents folder if it exists
    data_documents_path = "data_documents"
    if os.path.exists(data_documents_path):
        print(f"Processing documents in {data_documents_path}...")
        pdf_files = glob.glob(os.path.join(data_documents_path, "**", "*.pdf"), recursive=True)
        
        for pdf_path in tqdm(pdf_files, desc="Processing data_documents"):
            text = extract_text_from_pdf(pdf_path)
            
            if text:
                # Split into chunks to create more training examples
                chunks = [text[i:i+500] for i in range(0, len(text), 500)]
                
                for chunk in chunks:
                    if len(chunk) > 50:  # Only use chunks with sufficient content
                        texts.append(preprocess_text(chunk))
                        labels.append(categorize_document(os.path.basename(pdf_path)))
    
    # Process argilla_data_49 folder if it exists
    argilla_data_path = "argilla_data_49"
    if os.path.exists(argilla_data_path):
        print(f"Processing documents in {argilla_data_path}...")
        pdf_files = glob.glob(os.path.join(argilla_data_path, "**", "*.pdf"), recursive=True)
        
        for pdf_path in tqdm(pdf_files, desc="Processing argilla_data_49"):
            text = extract_text_from_pdf(pdf_path)
            
            if text:
                # Split into chunks to create more training examples
                chunks = [text[i:i+500] for i in range(0, len(text), 500)]
                
                for chunk in chunks:
                    if len(chunk) > 50:  # Only use chunks with sufficient content
                        texts.append(preprocess_text(chunk))
                        labels.append(categorize_document(os.path.basename(pdf_path)))
    
    # Create some negative examples (off-topic content)
    print("Generating negative examples...")
    negative_examples = [
        "What's the weather like today?",
        "Can you recommend a good restaurant in Rome?",
        "What's the capital of France?",
        "How tall is Mount Everest?",
        "When was the last World Cup?",
        "Can you help me book a flight to Milan?",
        "What's the best way to learn Italian?",
        "Tell me about the history of pizza",
        "How do I make pasta carbonara?",
        "What movies are playing in theaters this weekend?",
        "Can you recommend a good hotel in Florence?",
        "What's the exchange rate between Euro and Dollar?",
        "How do I get to the Colosseum from Vatican City?",
        "What's the population of Italy?",
        "Who is the current prime minister of Italy?",
        "What's the best time to visit Sicily?",
        "Can you translate 'hello' to Italian?",
        "What are the most popular tourist attractions in Venice?",
        "How long does it take to drive from Rome to Naples?",
        "What's the typical Italian breakfast?",
    ]
    
    texts.extend(negative_examples)
    labels.extend([2] * len(negative_examples))  # Label 2 for "Other"
    
    # Add some explicit IVA examples
    iva_examples = [
        "What is the current IVA rate in Italy?",
        "How do I register for IVA?",
        "What expenses can I deduct from my IVA?",
        "When do I need to pay IVA?",
        "What's the IVA exemption threshold?",
        "How do I apply for an IVA number?",
        "What is the reduced IVA rate for food?",
        "How often do I need to file IVA returns?",
        "What happens if I pay my IVA late?",
        "Can I reclaim IVA on business expenses?",
    ]
    
    texts.extend(iva_examples)
    labels.extend([0] * len(iva_examples))  # Label 0 for "IVA"
    
    # Add some explicit Fiscozen examples
    fiscozen_examples = [
        "What services does Fiscozen offer?",
        "How can Fiscozen help with my taxes?",
        "What is the cost of Fiscozen's services?",
        "How do I sign up for Fiscozen?",
        "Can Fiscozen help me with my tax return?",
        "What makes Fiscozen different from other tax services?",
        "Does Fiscozen offer consultations?",
        "How quickly does Fiscozen process tax returns?",
        "Can Fiscozen help with tax audits?",
        "How do I contact Fiscozen support?",
    ]
    
    texts.extend(fiscozen_examples)
    labels.extend([1] * len(fiscozen_examples))  # Label 1 for "Fiscozen"
    
    print(f"Total training examples: {len(texts)}")
    print(f"  - IVA examples: {labels.count(0)}")
    print(f"  - Fiscozen examples: {labels.count(1)}")
    print(f"  - Other examples: {labels.count(2)}")
    
    return texts, labels

def train_model():
    """Train the BERT model with the extracted data"""
    print("\n=== Training BERT Classifier ===\n")
    
    # Extract training data
    texts, labels = extract_training_data()
    
    if len(texts) < 10:
        print("Not enough training data. Please make sure your document folders contain PDF files.")
        return False
    
    # Initialize the relevance checker
    model_path = "bert/models/enhanced_bert"
    checker = RelevanceChecker(model_path=model_path)
    
    # Shuffle the data while keeping texts and labels aligned
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    
    # Train the model
    print(f"\nTraining model with {len(texts)} examples...")
    checker.train_with_data(texts, labels, batch_size=8, epochs=3)
    
    # Save the trained model
    checker.save_model(model_path)
    print(f"\nModel trained and saved to {model_path}")
    
    # Test the model with some examples
    print("\nTesting the trained model with examples:")
    test_examples = [
        "What is the current IVA rate?",
        "How can Fiscozen help me?",
        "What's the weather like today?",
        "I need help with my tax return",
        "When is the deadline for paying IVA?",
    ]
    
    for text in test_examples:
        result = checker.check_relevance(text)
        relevance = "Relevant" if result["is_relevant"] else "Not relevant"
        print(f"\nText: '{text}'")
        print(f"Result: {relevance} ({result['topic']}, confidence: {result['confidence']:.4f})")
        print(f"Tax-related probability: {result['tax_related_probability']:.4f}")
    
    return True

if __name__ == "__main__":
    # Make sure the output directory exists
    os.makedirs("bert/models/enhanced_bert", exist_ok=True)
    
    # Train the model
    success = train_model()
    
    if success:
        print("\n✅ BERT classifier successfully trained with document data!")
        print("   The model has been saved to bert/models/enhanced_bert")
        print("   Your chatbot will now use this enhanced model for relevance checking.")
    else:
        print("\n❌ Failed to train the BERT classifier.")
        print("   Please check the error messages above and ensure your document folders contain PDF files.") 