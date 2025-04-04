"""
Train a FastText classifier for tax relevance detection.

This script processes documents from data folders and trains a FastText model
to classify text as either IVA-related, Fiscozen-related, or Other.
"""

import os
import sys
import re
import random
import shutil
from pathlib import Path
import PyPDF2
from tqdm import tqdm

# Add parent directory to path for imports
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Create __init__.py file if it doesn't exist to make fasttext a proper package
init_file = os.path.join(os.path.dirname(__file__), "__init__.py")
if not os.path.exists(init_file):
    with open(init_file, "w") as f:
        f.write("# FastText package initialization\n")
    print(f"Created {init_file} to make fasttext directory a proper package")

# Try importing FastText RelevanceChecker with more explicit path handling
try:
    # First try direct import in case path is already set up
    from fasttext.relevance import FastTextRelevanceChecker
    print("Successfully imported FastTextRelevanceChecker")
except ImportError:
    try:
        # Try with relative import using full path
        sys.path.append(project_root)
        from fasttext.relevance import FastTextRelevanceChecker
        print("Successfully imported FastTextRelevanceChecker using explicit path")
    except ImportError:
        # If still not found, try copying the relevance.py file to the local directory
        relevance_path = os.path.join(os.path.dirname(__file__), "relevance.py")
        
        if os.path.exists(relevance_path):
            print(f"Found relevance.py at {relevance_path}")
            # Import directly from local file
            import importlib.util
            spec = importlib.util.spec_from_file_location("relevance", relevance_path)
            relevance = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(relevance)
            FastTextRelevanceChecker = relevance.FastTextRelevanceChecker
            print("Successfully imported FastTextRelevanceChecker from local file")
        else:
            print("Could not find relevance.py. Checking if fasttext is installed...")
            
            # Check and install fasttext if needed
            try:
                import fasttext
                print("FastText is already installed.")
            except ImportError:
                print("Installing fasttext package...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "fasttext"])
                import fasttext
                print("FastText installed successfully.")
            
            print("Error: Still cannot find FastTextRelevanceChecker.")
            print("Creating a simplified version for training...")
            
            # Create a simplified version of the FastTextRelevanceChecker for training
            import fasttext
            
            class SimplifiedFastTextRelevanceChecker:
                def __init__(self, model_path=None):
                    self.model_path = model_path or "fasttext/models/tax_classifier.bin"
                    self.model = None
                    self.labels = ["IVA", "Fiscozen", "Other"]
                    
                    # Create model directory if it doesn't exist
                    os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                
                def train_with_data(self, training_data, output_path=None, epochs=20, lr=0.1):
                    if not training_data:
                        print("No training data provided.")
                        return False
                    
                    try:
                        # Create a temporary file for training data
                        import tempfile
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
                        save_path = output_path or self.model_path
                        self.model.save_model(save_path)
                        print(f"Model saved to {save_path}")
                        
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
            
            FastTextRelevanceChecker = SimplifiedFastTextRelevanceChecker
            print("Using simplified FastTextRelevanceChecker for training")

def extract_text_from_pdf(pdf_path):
    """Extract text content from a PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
    return text

def preprocess_text(text):
    """Clean and normalize text for better training."""
    if not text:
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep letters, numbers, and spaces
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def categorize_documents():
    """
    Categorize documents from data folders into training categories.
    Returns training data as (text, label) tuples.
    """
    training_data = []
    
    # Define data directories - updated to include the correct directories
    data_dirs = ["data_documents",  "argilla_data_49"]
    categories = {
        "iva": "IVA",           # Documents about Italian VAT
        "fiscozen": "Fiscozen", # Documents about Fiscozen services
        "other": "Other"        # Documents about other topics
    }
    
    print(f"Looking for documents in these directories: {', '.join(data_dirs)}")
    
    # Process documents from all data directories
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            print(f"Directory {data_dir} not found. Skipping.")
            continue
        
        print(f"Processing documents from {data_dir}...")
        
        # Check if directory is empty
        if not os.listdir(data_dir):
            print(f"Directory {data_dir} is empty. Skipping.")
            continue
            
        # First try to find subdirectories for categories
        has_subdirs = False
        for item in os.listdir(data_dir):
            if os.path.isdir(os.path.join(data_dir, item)):
                has_subdirs = True
                break
        
        if has_subdirs:
            # Process each subdirectory for categorized documents
            for category_dir in os.listdir(data_dir):
                category_path = os.path.join(data_dir, category_dir)
                
                if not os.path.isdir(category_path):
                    continue
                
                # Determine the category from directory name
                category_name = category_dir.lower()
                if "iva" in category_name:
                    label = "IVA"
                elif "fiscozen" in category_name:
                    label = "Fiscozen"
                elif "other" in category_name or "misc" in category_name:
                    label = "Other"
                else:
                    # Default to "Other" if not explicitly categorized
                    label = "Other"
                
                # Process files in this category
                for file_name in os.listdir(category_path):
                    file_path = os.path.join(category_path, file_name)
                    
                    if os.path.isfile(file_path):
                        # Extract text based on file type
                        if file_name.endswith('.pdf'):
                            text = extract_text_from_pdf(file_path)
                        elif file_name.endswith('.txt'):
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                text = f.read()
                        else:
                            continue  # Skip unsupported file types
                        
                        # Preprocess text
                        processed_text = preprocess_text(text)
                        
                        if processed_text:
                            # Add to training data
                            training_data.append((processed_text, label))
                            
                            # For better training, split long documents into paragraphs
                            paragraphs = processed_text.split('\n\n')
                            if len(paragraphs) > 1:
                                for para in paragraphs:
                                    if len(para.split()) > 10:  # Only use paragraphs with at least 10 words
                                        training_data.append((para, label))
        else:
            # If no subdirectories, try to guess categories from filenames
            print(f"No category subdirectories found in {data_dir}. Processing files directly.")
            for file_name in os.listdir(data_dir):
                file_path = os.path.join(data_dir, file_name)
                
                if os.path.isfile(file_path):
                    # Determine category based on filename
                    file_name_lower = file_name.lower()
                    if "iva" in file_name_lower:
                        label = "IVA"
                    elif "fiscozen" in file_name_lower:
                        label = "Fiscozen"
                    else:
                        label = "Other"
                    
                    # Extract text based on file type
                    if file_name.endswith('.pdf'):
                        text = extract_text_from_pdf(file_path)
                    elif file_name.endswith('.txt'):
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            text = f.read()
                    else:
                        continue  # Skip unsupported file types
                    
                    # Preprocess text
                    processed_text = preprocess_text(text)
                    
                    if processed_text:
                        # Add to training data
                        training_data.append((processed_text, label))
                        
                        # For better training, split long documents into paragraphs
                        paragraphs = processed_text.split('\n\n')
                        if len(paragraphs) > 1:
                            for para in paragraphs:
                                if len(para.split()) > 10:  # Only use paragraphs with at least 10 words
                                    training_data.append((para, label))
    
    print(f"Processed {len(training_data)} documents/sections for training")
    
    # If we don't have enough data, create some synthetic examples
    if len(training_data) < 30:
        print("Not enough real training data. Adding synthetic examples...")
        
        # Synthetic IVA examples
        iva_examples = [
            "Come funziona l'IVA in Italia?",
            "Quali sono le aliquote IVA in Italia?",
            "Come si registra per l'IVA?",
            "Quando devo presentare la dichiarazione IVA?",
            "Posso recuperare l'IVA sugli acquisti aziendali?",
            "Ho bisogno di aiuto con il mio rimborso IVA",
            "Qual è la differenza tra IVA ordinaria e ridotta?",
            "Come gestire l'IVA per le vendite online?",
            "Devo applicare l'IVA ai clienti esteri?",
            "Vorrei sapere di più sulle esenzioni IVA"
        ]
        
        # Synthetic Fiscozen examples
        fiscozen_examples = [
            "Come Fiscozen può aiutarmi con la contabilità?",
            "Quali servizi offre Fiscozen?",
            "Come funziona la piattaforma Fiscozen?",
            "Quanto costa usare Fiscozen?",
            "Fiscozen gestisce anche la dichiarazione dei redditi?",
            "Vorrei aprire un account Fiscozen",
            "È possibile parlare con un consulente Fiscozen?",
            "Come Fiscozen gestisce le fatture elettroniche?",
            "Quali sono i vantaggi di usare Fiscozen?",
            "Posso gestire più attività con Fiscozen?"
        ]
        
        # Synthetic Other examples
        other_examples = [
            "Come è il tempo oggi?",
            "Quali sono i migliori ristoranti in zona?",
            "Mi puoi consigliare un buon film?",
            "Come si dice ciao in francese?",
            "Qual è la capitale della Spagna?",
            "Chi ha vinto l'ultimo mondiale di calcio?",
            "Raccontami una barzelletta",
            "Mi serve una ricetta per la pasta",
            "Quanto costa un biglietto per Parigi?",
            "Come si coltivano i pomodori?"
        ]
        
        # Add synthetic examples to training data
        for example in iva_examples:
            training_data.append((example, "IVA"))
        
        for example in fiscozen_examples:
            training_data.append((example, "Fiscozen"))
        
        for example in other_examples:
            training_data.append((example, "Other"))
    
    # Shuffle the training data
    random.shuffle(training_data)
    
    return training_data

def train_fasttext_model():
    """Train a FastText model with document data."""
    # Output model path
    output_dir = "fasttext/models"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "tax_classifier.bin")
    
    # Get training data
    print("Categorizing and processing documents...")
    training_data = categorize_documents()
    
    if not training_data:
        print("No training data available. Creating a minimal training set...")
        # Create minimal training set if no data is available
        minimal_data = [
            ("iva calcolo aliquota", "IVA"),
            ("rimborso iva", "IVA"),
            ("fiscozen servizi", "Fiscozen"),
            ("consulente fiscozen", "Fiscozen"),
            ("tempo oggi", "Other"),
            ("ricetta pasta", "Other")
        ]
        training_data = minimal_data
    
    # Initialize and train the model
    print(f"Training FastText model with {len(training_data)} examples...")
    relevance_checker = FastTextRelevanceChecker(model_path=model_path)
    
    # Train the model
    success = relevance_checker.train_with_data(
        training_data=training_data,
        output_path=model_path,
        epochs=25,  # Increase epochs for better learning
        lr=0.1
    )
    
    if success:
        print(f"✅ FastText model trained and saved to {model_path}")
        return True
    else:
        print("❌ Failed to train FastText model")
        return False

if __name__ == "__main__":
    print("\n🔹🔹🔹 TRAINING FASTTEXT CLASSIFIER 🔹🔹🔹\n")
    
    # Create necessary directories
    os.makedirs("fasttext/models", exist_ok=True)
    
    # Train the model
    success = train_fasttext_model()
    
    if success:
        print("\n✅ FastText classifier training completed successfully!")
    else:
        print("\n❌ FastText classifier training failed!")
        sys.exit(1) 