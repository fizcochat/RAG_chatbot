"""
Utility to enhance your existing relevance checker by training with PDF documents.
This augments your existing system without replacing it.
"""

import os
import glob
import json
from typing import List, Tuple
import random
from relevance import RelevanceChecker
from pdfminer.high_level import extract_text

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file"""
    try:
        return extract_text(pdf_path)
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def generate_training_samples_from_text(text: str, category: int, n_samples: int = 10) -> List[Tuple[str, int]]:
    """
    Generate training samples from extracted text.
    
    Args:
        text: The extracted text from a document
        category: Category label (0=IVA, 1=Fiscozen, 2=Other Tax)
        n_samples: Number of samples to generate
        
    Returns:
        List of (text, label) tuples
    """
    # Split text into paragraphs and filter out empty ones
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    # If we don't have enough paragraphs, use sentences
    if len(paragraphs) < n_samples:
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 30]
        paragraphs = sentences
    
    # Ensure we have content to work with
    if not paragraphs:
        return []
    
    # Select random paragraphs as samples
    samples = []
    for _ in range(min(n_samples, len(paragraphs))):
        idx = random.randint(0, len(paragraphs) - 1)
        paragraph = paragraphs[idx]
        
        # Limit length to avoid excessively long samples
        if len(paragraph) > 500:
            paragraph = paragraph[:500]
            
        samples.append((paragraph, category))
        
    return samples

def create_iva_questions(n_samples: int = 10) -> List[Tuple[str, int]]:
    """Create IVA-specific questions"""
    iva_questions = [
        "What is my IVA balance?",
        "How do I calculate my IVA payments?",
        "When is my next IVA payment due?",
        "Can I reduce my IVA payments?",
        "What happens if I miss an IVA payment?",
        "How long does an IVA last?",
        "Is IVA right for my situation?",
        "What debts can be included in an IVA?",
        "How will an IVA affect my credit score?",
        "Can I get an IVA if I'm self-employed?",
        "How to register for IVA?",
        "What are the IVA deadlines?",
        "How do I file IVA paperwork?",
        "What's the current IVA rate?",
        "Who needs to pay IVA tax?",
    ]
    
    # Ensure we don't try to get more questions than we have
    n_samples = min(n_samples, len(iva_questions))
    
    # Return selected questions with category 0 (IVA)
    return [(iva_questions[i], 0) for i in range(n_samples)]

def create_fiscozen_questions(n_samples: int = 10) -> List[Tuple[str, int]]:
    """Create Fiscozen-specific questions"""
    fiscozen_questions = [
        "Can Fiscozen help with my tax return?",
        "What services does Fiscozen offer?",
        "How much does Fiscozen charge?",
        "Is Fiscozen available in my region?",
        "How do I contact Fiscozen support?",
        "Can Fiscozen handle corporate taxes?",
        "Does Fiscozen offer tax consulting?",
        "What are Fiscozen's business hours?",
        "How do I schedule a consultation with Fiscozen?",
        "Can Fiscozen help with international taxation?",
        "How do I sign up with Fiscozen?",
        "What documents do I need to provide to Fiscozen?",
        "What is Fiscozen's privacy policy?",
        "Does Fiscozen handle tax disputes?",
        "Can Fiscozen represent me to tax authorities?",
    ]
    
    # Ensure we don't try to get more questions than we have
    n_samples = min(n_samples, len(fiscozen_questions))
    
    # Return selected questions with category 1 (Fiscozen)
    return [(fiscozen_questions[i], 1) for i in range(n_samples)]

def create_non_relevant_questions(n_samples: int = 5) -> List[Tuple[str, int]]:
    """Create non-relevant questions"""
    non_relevant_questions = [
        "What's the weather like today?",
        "Can you recommend a good restaurant?",
        "How do I book a flight?",
        "What's the latest sports news?",
        "Where can I buy concert tickets?",
        "When is the next football match?",
        "What's a good movie to watch?",
        "How do I change my email password?",
        "What's the best smartphone to buy?",
        "Can you tell me a joke?",
    ]
    
    # Ensure we don't try to get more questions than we have
    n_samples = min(n_samples, len(non_relevant_questions))
    
    # Return selected questions with category 2 (Other)
    return [(non_relevant_questions[i], 2) for i in range(n_samples)]

def enhance_relevance_checker(argilla_folder: str = "argilla_data_49", 
                          docs_folder: str = "data_documents", 
                          model_path: str = "models/bert_classifier"):
    """
    Enhance your existing relevance checker with document data
    
    Args:
        argilla_folder: Path to the folder containing Argilla data and PDFs
        docs_folder: Path to additional folder containing documents
        model_path: Path where to save the trained model
    """
    # Make sure the model directory exists
    os.makedirs("models", exist_ok=True)
    
    # Collect training data
    training_data = []
    
    # 1. Find and process PDFs from argilla_data_49 folder
    print(f"Processing PDFs from {argilla_folder}...")
    argilla_pdfs = glob.glob(os.path.join(argilla_folder, "*.pdf"), recursive=True)
    
    for pdf_path in argilla_pdfs:
        print(f"Extracting text from {pdf_path}")
        text = extract_text_from_pdf(pdf_path)
        
        # Determine category based on filename
        filename = os.path.basename(pdf_path).lower()
        if "iva" in filename or "agenzia" in filename:
            category = 0  # IVA
        elif "fiscozen" in filename:
            category = 1  # Fiscozen
        else:
            category = 2  # Other tax
            
        samples = generate_training_samples_from_text(text, category)
        training_data.extend(samples)
    
    # 2. Also check the regular data_documents folder if it exists
    if os.path.exists(docs_folder):
        print(f"Processing additional documents from {docs_folder}...")
        doc_pdfs = glob.glob(os.path.join(docs_folder, "*.pdf"), recursive=True)
        
        for pdf_path in doc_pdfs:
            print(f"Extracting text from {pdf_path}")
            text = extract_text_from_pdf(pdf_path)
            
            # Determine category based on filename
            filename = os.path.basename(pdf_path).lower()
            if "iva" in filename or "agenzia" in filename:
                category = 0  # IVA
            elif "fiscozen" in filename:
                category = 1  # Fiscozen
            else:
                category = 2  # Other tax
                
            samples = generate_training_samples_from_text(text, category)
            training_data.extend(samples)
    
    # 3. Add manually created questions
    print("Adding predefined questions...")
    training_data.extend(create_iva_questions(10))
    training_data.extend(create_fiscozen_questions(10))
    training_data.extend(create_non_relevant_questions(5))
    
    # 4. Shuffle the training data
    random.shuffle(training_data)
    
    # 5. Split into texts and labels
    texts = [x[0] for x in training_data]
    labels = [x[1] for x in training_data]
    
    # 6. Train the model (enhancing your existing RelevanceChecker)
    print(f"Training with {len(texts)} examples")
    print(f"Categories: IVA: {labels.count(0)}, Fiscozen: {labels.count(1)}, Other: {labels.count(2)}")
    
    # Using your existing RelevanceChecker
    relevance_checker = RelevanceChecker()
    relevance_checker.train_with_data(texts, labels)
    
    # 7. Save the enhanced model
    relevance_checker.save_model(model_path)
    print(f"Enhanced model trained and saved to {model_path}")
    print("Your existing relevance checker has been enhanced with document data!")

if __name__ == "__main__":
    enhance_relevance_checker() 