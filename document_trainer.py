"""
Utility to enhance your existing relevance checker by training with PDF documents.
This augments your existing system without replacing it.
"""

import os
import glob
import json
import re
from typing import List, Tuple, Dict, Any
import random
import torch
from tqdm import tqdm
from pdfminer.high_level import extract_text
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import numpy as np

# For direct imports within RelevanceChecker implementation
# Make sure these imports match what's used in the relevance checker
from relevance import RelevanceChecker

def preprocess_text(text):
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

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file"""
    try:
        text = extract_text(pdf_path)
        # Preprocess extracted text
        return preprocess_text(text)
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
        "I need information about I.V.A",
        "Tell me about the Italian VAT system",
        "How is IVA calculated?",
        "When do I need to pay my IVA?",
        "What happens if I don't pay IVA?",
        "What is the current partita IVA rate?",
        "How do I open a partita IVA?",
        "Is my business required to have a P.IVA?",
        "Can I have multiple partita IVA numbers?",
        "How to close a partita IVA account?",
    ]
    
    # Ensure we don't try to get more questions than we have
    n_samples = min(n_samples, len(iva_questions))
    
    # Preprocess all questions
    processed_questions = [(preprocess_text(iva_questions[i]), 0) for i in range(n_samples)]
    
    # Return selected questions with category 0 (IVA)
    return processed_questions

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
        "Tell me about Fisco-Zen services",
        "I want to use Fiscozen for my taxes",
        "Does Fiscozen's platform support English?",
        "What benefits does Fiscozen offer freelancers?",
        "How long has Fiscozen been operating?",
        "Does Fiscozen have an app?",
        "What makes Fiscozen different from traditional accountants?",
        "Can Fiscozen help with tax deductions?",
        "Is Fiscozen suitable for small businesses?",
        "How do I upload my receipts to Fiscozen?",
    ]
    
    # Ensure we don't try to get more questions than we have
    n_samples = min(n_samples, len(fiscozen_questions))
    
    # Preprocess all questions
    processed_questions = [(preprocess_text(fiscozen_questions[i]), 1) for i in range(n_samples)]
    
    # Return selected questions with category 1 (Fiscozen)
    return processed_questions

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
        "What's the capital of France?",
        "How do I make pasta carbonara?",
        "Tell me about the history of Rome",
        "What are the top tourist attractions in Milan?",
        "How tall is the Eiffel Tower?",
        "What time is it in New York?",
        "Who won the last World Cup?",
        "How do I learn to speak Italian?",
        "What's the best way to get to the airport?",
        "Can you recommend a good book?",
    ]
    
    # Ensure we don't try to get more questions than we have
    n_samples = min(n_samples, len(non_relevant_questions))
    
    # Preprocess all questions
    processed_questions = [(preprocess_text(non_relevant_questions[i]), 2) for i in range(n_samples)]
    
    # Return selected questions with category 2 (Other)
    return processed_questions

def train_model_with_advanced_techniques(relevance_checker, texts, labels, validation_split=0.2, 
                                         batch_size=16, epochs=4, learning_rate=5e-5, 
                                         warmup_proportion=0.1, max_grad_norm=1.0, 
                                         patience=2):
    """
    Train the model with advanced techniques for improved stability on noisy data:
    - Learning rate scheduler with warmup
    - Early stopping
    - Gradient clipping
    
    Args:
        relevance_checker: The RelevanceChecker instance
        texts: List of text samples
        labels: List of corresponding labels
        validation_split: Proportion of data to use for validation
        batch_size: Training batch size
        epochs: Maximum number of epochs
        learning_rate: Initial learning rate
        warmup_proportion: Proportion of steps to warm up the learning rate
        max_grad_norm: Maximum gradient norm for gradient clipping
        patience: Number of epochs with no improvement after which training will stop
    
    Returns:
        Dict with training metrics
    """
    # Create the tokenizer from the BERT model
    tokenizer = relevance_checker.tokenizer
    
    # Split the data into training and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=validation_split, stratify=labels, random_state=42
    )
    
    print(f"Training on {len(train_texts)} samples, validating on {len(val_texts)} samples")
    
    # Convert to tensors and create Dataset
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
    
    train_dataset = torch.utils.data.TensorDataset(
        train_encodings['input_ids'], 
        train_encodings['attention_mask'], 
        torch.tensor(train_labels)
    )
    
    val_dataset = torch.utils.data.TensorDataset(
        val_encodings['input_ids'], 
        val_encodings['attention_mask'], 
        torch.tensor(val_labels)
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    
    # Load model and move to device
    model = relevance_checker.model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Configure optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Create learning rate scheduler with warmup
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Early stopping variables
    best_val_loss = float('inf')
    no_improve_epochs = 0
    best_model_state = None
    
    # Training loop
    metrics = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in train_progress:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_progress.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Calculate average training loss for this epoch
        avg_train_loss = train_loss / len(train_loader)
        metrics['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for batch in val_progress:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                
                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                val_progress.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Calculate average validation loss and accuracy
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        metrics['val_loss'].append(avg_val_loss)
        metrics['val_accuracy'].append(accuracy)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.4f} - "
              f"Val Loss: {avg_val_loss:.4f} - "
              f"Val Accuracy: {accuracy:.2f}%")
        
        # Check for improvement for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            no_improve_epochs = 0
            print(f"New best validation loss: {best_val_loss:.4f}")
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epochs")
            
        # Early stopping check
        if no_improve_epochs >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    # Load the best model state if we have one
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model from validation")
    
    # Move model back to CPU for saving
    model.to('cpu')
    
    # Update the model in the relevance checker
    relevance_checker.model = model
    
    return metrics

def enhance_relevance_checker(argilla_folder: str = "argilla_data_49", 
                          docs_folder: str = "data_documents", 
                          model_path: str = "models/enhanced_bert",
                          advanced_training: bool = True):
    """
    Enhance your existing relevance checker with document data
    
    Args:
        argilla_folder: Path to the folder containing Argilla data and PDFs
        docs_folder: Path to additional folder containing documents
        model_path: Path where to save the trained model
        advanced_training: Whether to use advanced training techniques
    """
    # Make sure the model directory exists
    os.makedirs("models", exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Collect training data
    training_data = []
    
    # 1. Find and process PDFs from argilla_data_49 folder
    print(f"Processing PDFs from {argilla_folder}...")
    if os.path.exists(argilla_folder):
        argilla_pdfs = glob.glob(os.path.join(argilla_folder, "*.pdf"), recursive=True)
        
        for pdf_path in argilla_pdfs:
            print(f"Extracting text from {pdf_path}")
            text = extract_text_from_pdf(pdf_path)
            
            # Skip empty or very short texts
            if len(text.strip()) < 100:
                print(f"  Skipping {pdf_path} - insufficient text content")
                continue
                
            # Determine category based on filename
            filename = os.path.basename(pdf_path).lower()
            if "iva" in filename or "agenzia" in filename or "partita" in filename:
                category = 0  # IVA
                print(f"  Categorized as IVA")
            elif "fiscozen" in filename:
                category = 1  # Fiscozen
                print(f"  Categorized as Fiscozen")
            else:
                category = 2  # Other tax
                print(f"  Categorized as Other tax")
                
            samples = generate_training_samples_from_text(text, category)
            print(f"  Generated {len(samples)} training samples")
            training_data.extend(samples)
    else:
        print(f"Warning: Folder {argilla_folder} does not exist")
    
    # 2. Also check the regular data_documents folder if it exists
    if os.path.exists(docs_folder):
        print(f"Processing additional documents from {docs_folder}...")
        doc_pdfs = glob.glob(os.path.join(docs_folder, "*.pdf"), recursive=True)
        
        for pdf_path in doc_pdfs:
            print(f"Extracting text from {pdf_path}")
            text = extract_text_from_pdf(pdf_path)
            
            # Skip empty or very short texts
            if len(text.strip()) < 100:
                print(f"  Skipping {pdf_path} - insufficient text content")
                continue
            
            # Determine category based on filename
            filename = os.path.basename(pdf_path).lower()
            if "iva" in filename or "agenzia" in filename or "partita" in filename:
                category = 0  # IVA
                print(f"  Categorized as IVA")
            elif "fiscozen" in filename:
                category = 1  # Fiscozen
                print(f"  Categorized as Fiscozen")
            else:
                category = 2  # Other tax
                print(f"  Categorized as Other tax")
                
            samples = generate_training_samples_from_text(text, category)
            print(f"  Generated {len(samples)} training samples")
            training_data.extend(samples)
    else:
        print(f"Warning: Folder {docs_folder} does not exist")
    
    # 3. Add manually created questions
    print("Adding predefined questions...")
    training_data.extend(create_iva_questions(25))
    training_data.extend(create_fiscozen_questions(25))
    training_data.extend(create_non_relevant_questions(20))
    
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
    
    # Option to use advanced training techniques or the default training method
    if advanced_training and len(texts) > 20:  # Ensure we have enough data for validation split
        print("Using advanced training techniques (learning rate scheduler, early stopping, gradient clipping)")
        metrics = train_model_with_advanced_techniques(
            relevance_checker=relevance_checker,
            texts=texts,
            labels=labels,
            validation_split=0.2,
            batch_size=16,
            epochs=4,
            learning_rate=5e-5,
            warmup_proportion=0.1,
            max_grad_norm=1.0,
            patience=2
        )
        
        # Print final metrics
        final_epoch = len(metrics['val_accuracy'])
        print(f"Final validation accuracy: {metrics['val_accuracy'][-1]:.2f}%")
        print(f"Training completed in {final_epoch} epochs")
        
        # Save metrics for analysis if needed
        metrics_path = os.path.join(os.path.dirname(model_path), "training_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    else:
        print("Using default training method")
        relevance_checker.train_with_data(texts, labels)
    
    # 7. Save the enhanced model
    relevance_checker.save_model(model_path)
    print(f"Enhanced model trained and saved to {model_path}")
    print("Your existing relevance checker has been enhanced with document data!")

if __name__ == "__main__":
    enhance_relevance_checker() 