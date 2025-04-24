"""
FastText model training script for tax relevance classification.

This script extracts text from PDF documents and Excel files in the data directories,
and trains a FastText model to classify text as tax-related or not.
"""

import os
import re
import string
import random
import pandas as pd
from pathlib import Path
import fasttext
import PyPDF2
import shutil
from tqdm import tqdm
import argparse
import nltk
from nltk.corpus import stopwords
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create directories if they don't exist
os.makedirs("fast_text/models", exist_ok=True)

# Download NLTK resources if not available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Define constants
TRAINING_DATA_PATH = "fasttext_training_data.txt"
VALIDATION_DATA_PATH = "fasttext_validation_data.txt"
MODEL_PATH = "fast_text/models/tax_classifier.bin"
PDF_DATA_DIRS = ["argilla_data_49", "data_documents"]

# Define Italian tax-related keywords for synthetic data generation
TAX_KEYWORDS = [
    'iva', 'imposta', 'tasse', 'forfettario', 'partita iva', 'fiscale', 'dichiarazione',
    'fattura', 'detrazioni', 'redditi', 'irpef', 'inps', 'agenzia entrate', 'f24',
    'regime forfettario', 'contabilità', 'commercialista', 'fiscozen', 'contributi',
    'scadenze fiscali', 'fatturazione elettronica'
]

# Define non-tax keywords for generating negative examples
NON_TAX_KEYWORDS = [
    'cinema', 'sport', 'musica', 'vacanza', 'ristorante', 'libro', 'film', 'arte',
    'concerto', 'cibo', 'ricetta', 'tempo libero', 'hobby', 'viaggiare', 'cucinare',
    'serie tv', 'calcio', 'automobile', 'giardino', 'tecnologia', 'moda', 'salute',
    'medicina', 'esercizio', 'università', 'scuola', 'bambini', 'famiglia', 'amore',
    'amicizia', 'lavoro', 'dormire', 'natura', 'montagna', 'mare', 'weekend'
]

def clean_text(text):
    """Clean and normalize text for training."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

def extract_data_from_excel(excel_path):
    """Extract labeled text data from Excel file if available."""
    try:
        df = pd.read_excel(excel_path)
        
        # Check if this looks like a labeled dataset
        if 'text' in df.columns or 'Text' in df.columns:
            text_col = 'text' if 'text' in df.columns else 'Text'
            
            # Look for a label column
            label_col = None
            for col in df.columns:
                if 'label' in col.lower() or 'category' in col.lower() or 'class' in col.lower():
                    label_col = col
                    break
            
            texts = df[text_col].tolist()
            
            if label_col:
                labels = df[label_col].tolist()
                return list(zip(texts, labels))
            else:
                # Assume all texts from excel are tax-related (positive examples)
                return [(text, 'Tax') for text in texts if isinstance(text, str) and len(text) > 10]
        else:
            return []
    except Exception as e:
        logging.error(f"Error processing Excel file {excel_path}: {e}")
        return []

def create_training_chunks(text, chunk_size=200, overlap=50):
    """Split text into overlapping chunks of specified size."""
    words = text.split()
    chunks = []
    
    if len(words) <= chunk_size:
        return [text]
    
    for i in range(0, len(words) - chunk_size + 1, chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks

def generate_synthetic_examples(count=500):
    """Generate synthetic examples for training."""
    italian_stopwords = set(stopwords.words('italian'))
    examples = []
    
    # Generate positive examples (tax-related)
    for _ in range(count // 2):
        # Select 2-4 random tax keywords
        kw_count = random.randint(2, 4)
        keywords = random.sample(TAX_KEYWORDS, kw_count)
        
        # Generate a sentence with these keywords
        words = []
        for _ in range(random.randint(5, 15)):
            words.append(random.choice(list(italian_stopwords)))
        
        # Insert keywords at random positions
        for kw in keywords:
            pos = random.randint(0, len(words))
            words.insert(pos, kw)
        
        sentence = ' '.join(words)
        examples.append((sentence, 'Tax'))
    
    # Generate negative examples (not tax-related)
    for _ in range(count // 2):
        # Select 2-4 random non-tax keywords
        kw_count = random.randint(2, 4)
        keywords = random.sample(NON_TAX_KEYWORDS, kw_count)
        
        # Generate a sentence with these keywords
        words = []
        for _ in range(random.randint(5, 15)):
            words.append(random.choice(list(italian_stopwords)))
        
        # Insert keywords at random positions
        for kw in keywords:
            pos = random.randint(0, len(words))
            words.insert(pos, kw)
        
        sentence = ' '.join(words)
        examples.append((sentence, 'NonTax'))
    
    random.shuffle(examples)
    return examples

def prepare_training_data():
    """Prepare training data from PDFs and Excel files."""
    all_examples = []
    
    # Process PDF files
    pdf_paths = []
    for data_dir in PDF_DATA_DIRS:
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.lower().endswith('.pdf'):
                    pdf_paths.append(os.path.join(data_dir, file))
    
    # Extract text from PDFs
    logging.info(f"Processing {len(pdf_paths)} PDF files...")
    for pdf_path in tqdm(pdf_paths):
        text = extract_text_from_pdf(pdf_path)
        if text:
            # Split into chunks and label as Tax (positive examples)
            chunks = create_training_chunks(clean_text(text))
            all_examples.extend([(chunk, 'Tax') for chunk in chunks])
    
    # Look for Excel files that might contain labeled data
    excel_paths = []
    for data_dir in PDF_DATA_DIRS:
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.lower().endswith(('.xlsx', '.xls')):
                    excel_paths.append(os.path.join(data_dir, file))
    
    # Process Excel files if any
    logging.info(f"Processing {len(excel_paths)} Excel files...")
    for excel_path in excel_paths:
        labeled_data = extract_data_from_excel(excel_path)
        all_examples.extend(labeled_data)
    
    # Generate synthetic examples to balance the dataset
    logging.info("Generating synthetic examples...")
    synthetic_examples = generate_synthetic_examples(count=1000)
    all_examples.extend(synthetic_examples)
    
    # Shuffle examples
    random.shuffle(all_examples)
    
    # Split into training and validation sets (80/20)
    split_idx = int(len(all_examples) * 0.8)
    train_examples = all_examples[:split_idx]
    validation_examples = all_examples[split_idx:]
    
    # Write to training file
    with open(TRAINING_DATA_PATH, 'w', encoding='utf-8') as f:
        for text, label in train_examples:
            f.write(f"__label__{label} {text}\n")
    
    # Write to validation file
    with open(VALIDATION_DATA_PATH, 'w', encoding='utf-8') as f:
        for text, label in validation_examples:
            f.write(f"__label__{label} {text}\n")
    
    logging.info(f"Created training data with {len(train_examples)} examples")
    logging.info(f"Created validation data with {len(validation_examples)} examples")
    
    return len(train_examples), len(validation_examples)

def train_model():
    """Train the FastText model for classification."""
    logging.info("Training FastText model...")
    
    # Train the model
    model = fasttext.train_supervised(
        input=TRAINING_DATA_PATH,
        epoch=25,
        lr=0.5,
        wordNgrams=2,
        dim=100,
        loss='softmax'
    )
    
    # Save the model
    model.save_model(MODEL_PATH)
    logging.info(f"Model saved to {MODEL_PATH}")
    
    # Evaluate on validation set
    result = model.test(VALIDATION_DATA_PATH)
    precision = result[1]
    recall = result[2]
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    logging.info(f"Model evaluation results:")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1_score:.4f}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Train a FastText model for tax relevance classification")
    parser.add_argument('--force', action='store_true', help='Force retraining even if model exists')
    args = parser.parse_args()
    
    if os.path.exists(MODEL_PATH) and not args.force:
        logging.info(f"Model already exists at {MODEL_PATH}. Use --force to retrain.")
        return
    
    train_size, val_size = prepare_training_data()
    
    if train_size > 0:
        model = train_model()
        
        # Clean up temporary files
        for file in [TRAINING_DATA_PATH, VALIDATION_DATA_PATH]:
            if os.path.exists(file):
                os.remove(file)
        
        logging.info("Training complete!")
    else:
        logging.error("No training data was generated. Check data directories.")

if __name__ == "__main__":
    main() 