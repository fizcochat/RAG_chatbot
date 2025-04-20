"""
Improved FastText model training for the Fiscozen tax chatbot.

This script enhances the FastText model training by:
1. Extracting labeled text chunks from Pinecone (with metadata)
2. Adding manually crafted tax-related examples to improve recall
3. Generating negative examples to balance the dataset
4. Implementing better text preprocessing for Italian tax documents
5. Using cross-validation to evaluate and tune the model
6. Optimizing hyperparameters for better classification
"""

import os
import sys
import re
import glob
import pandas as pd
import tempfile
import logging
import random
from typing import List, Tuple, Dict, Set, Optional
from collections import Counter
from sklearn.model_selection import train_test_split
import fasttext
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Pinecone setup
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index = os.getenv("PINECONE_INDEX", "ragtest")
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index)

# Constants
MODEL_PATH = os.path.join("fast_text", "models", "tax_classifier.bin")

#ARGILLA_DATA_DIR = "argilla_data_49"
#DOCUMENTS_DATA_DIR = "data_documents"
#EXCEL_PATH = os.path.join(DOCUMENTS_DATA_DIR, "worker_conversations_labeled_translated_file.xlsx")

# Define Italian tax domain-specific stop words
ITALIAN_STOPWORDS = set([
    "il", "lo", "la", "i", "gli", "le", "un", "una", "del", "della", "dello", 
    "dei", "degli", "delle", "al", "alla", "ai", "agli", "alle", "dal", "dalla",
    "dallo", "dai", "dagli", "dalle", "di", "a", "da", "in", "con", "su", "per",
    "tra", "fra", "e", "ed", "o", "ma", "se", "perché", "che", "chi", "come",
    "dove", "quando", "quale", "quali"
])

# Define tax-related terms to preserve during preprocessing
TAX_TERMS = {
    "iva", "partita iva", "fisco", "fiscale", "tasse", "imposte", "dichiarazione", 
    "fattura", "fatture", "detrazioni", "detraibili", "deducibili", "spese", 
    "redditi", "tributario", "tributi", "contributi", "agenzia entrate", 
    "forfettario", "fiscozen", "commercialista", "contabilità", "rimborso",
    "aliquota", "aliquote"
}

def clean_text(text: str) -> str:
    if not text or not isinstance(text, str): return ""
    text = text.lower()
    has_tax_terms = any(term in text for term in TAX_TERMS)
    replacements = {"art.": "articolo", "p.iva": "partita iva", "c.f.": "codice fiscale", "f24": "modello f24", "f23": "modello f23", "€": "euro"}
    for abbr, full in replacements.items(): text = text.replace(abbr, full)
    # Remove URLs, email, and other unwanted patterns
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Preserve key terms in the text
    if "iva" in text.split(): text += " imposta_valore_aggiunto tasse fiscale"
    if has_tax_terms: text += " tasse fiscale iva dichiarazione"
    return text

def extract_data_from_pinecone(k: int = 10000) -> List[Tuple[str, str]]:
    training_data = []
    try:
        for ns in [pinecone_index]:
            response = index.query(namespace=ns, top_k=k, include_metadata=True, vector=[0.0]*1536)
            for match in response.get("matches", []):
                metadata = match.get("metadata", {})
                text = metadata.get("text", "")
                plan_type = metadata.get("plan_type", "") or ""
                if not text: continue
                label = "Tax" if any(term in text.lower() for term in TAX_TERMS) or plan_type.lower() == "tax" else "Other"
                cleaned = clean_text(text)
                if cleaned: training_data.append((cleaned, label))
    except Exception as e:
        logger.error(f"Failed to extract from Pinecone: {e}")
    return training_data

'''
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from a PDF file.
    Args:
        pdf_path: Path to the PDF file
    Returns:
        Extracted text
    """
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return ""
'''
'''''
def load_labeled_data_from_excel() -> List[Tuple[str, str]]:
    """
    Load labeled conversation data from Excel file.
    
    Returns:
        List of (text, label) tuples
    """
    try:
        logger.info(f"Loading labeled data from {EXCEL_PATH}")
        
        if not os.path.exists(EXCEL_PATH):
            logger.warning(f"Excel file not found at {EXCEL_PATH}")
            return []
            
        df = pd.read_excel(EXCEL_PATH)
        logger.info(f"Loaded Excel with columns: {df.columns.tolist()}")
        
        training_data = []
        
        for idx, row in df.iterrows():
            try:
                text = str(row.get('Conversazione', '')).strip()
                label = str(row.get('Label', '')).strip()
                
                if text and label:
                    # Force text with these keywords to be tax-related
                    tax_keywords = ["iva", "tasse", "fiscale", "fattura", "dichiarazione", "fiscozen"]
                    
                    # Convert numeric label to category - Fix: Always assign IVA if label is 1
                    if label == '1':  # Tax/IVA related
                        category = 'Tax'
                    elif any(keyword in text.lower() for keyword in tax_keywords):
                        category = 'Tax'
                    else:
                        category = 'Other'
                    
                    # Clean up the text and normalize
                    text = clean_text(text)
                    if text:  # Only add if text is not empty after cleaning
                        training_data.append((text, category))
                    
            except Exception as row_error:
                logger.error(f"Error processing row {idx}: {row_error}")
                continue
        
        logger.info(f"Processed {len(training_data)} examples from Excel")
        return training_data
        
    except Exception as e:
        logger.error(f"Error loading Excel data: {e}")
        return []
'''
'''
def extract_data_from_pdfs() -> List[Tuple[str, str]]:
    """
    Extract text from PDFs and create training examples.
    
    Returns:
        List of (text, label) tuples
    """
    training_data = []
    
    # Process Argilla data (tax-related documents)
    argilla_pdf_paths = glob.glob(os.path.join(ARGILLA_DATA_DIR, "*.pdf"))
    logger.info(f"Found {len(argilla_pdf_paths)} PDFs in Argilla directory")
    
    for pdf_path in tqdm(argilla_pdf_paths, desc="Processing Argilla PDFs"):
        text = extract_text_from_pdf(pdf_path)
        if text:
            # Split into paragraphs for more granular examples
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            # Take each paragraph as a separate training example
            for paragraph in paragraphs:
                if len(paragraph.split()) >= 5:  # Only use paragraphs with at least 5 words
                    clean_paragraph = clean_text(paragraph)
                    if clean_paragraph:
                        # All Argilla data is tax-related
                        training_data.append((clean_paragraph, 'Tax'))
    
    # Process other documents
    doc_pdf_paths = glob.glob(os.path.join(DOCUMENTS_DATA_DIR, "*.pdf"))
    logger.info(f"Found {len(doc_pdf_paths)} PDFs in Documents directory")
    
    for pdf_path in tqdm(doc_pdf_paths, desc="Processing Documents PDFs"):
        text = extract_text_from_pdf(pdf_path)
        
        # Check if this is a tax document by looking for keywords
        tax_keywords = ['iva', 'fiscale', 'tasse', 'agenzia entrate', 'dichiarazione', 'fattura']
        is_tax_related = any(keyword in text.lower() for keyword in tax_keywords)
        
        # Update: Always label tax-related documents as 'Tax'
        label = 'Tax' if is_tax_related else 'Other'
        
        if text:
            # Split into paragraphs for more granular examples
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            # Take each paragraph as a separate training example
            for paragraph in paragraphs:
                if len(paragraph.split()) >= 5:  # Only use paragraphs with at least 5 words
                    clean_paragraph = clean_text(paragraph)
                    if clean_paragraph:
                        # For tax documents, ensure they get the Tax label
                        training_data.append((clean_paragraph, label))
    
    logger.info(f"Extracted {len(training_data)} examples from PDFs")
    return training_data
'''

def generate_negative_examples(positive_examples: List[Tuple[str, str]], count: int = 1000) -> List[Tuple[str, str]]:
    """
    Generate negative examples (non-tax related text).
    Args:
        positive_examples: List of positive training examples
        count: Number of negative examples to generate
    Returns:
        List of (text, label) tuples with non-tax topics
    """
    common_topics = [
        "Quale ristorante consigli per cena?",
        "Mi puoi dire qual è il meteo di oggi?",
        "Quando è il prossimo concerto a Milano?",
        "Dove posso trovare un buon hotel?",
        "Ho bisogno di informazioni sul trasporto pubblico",
        "Qual è il migliore film in uscita?",
        "Come posso acquistare biglietti del treno?",
        "Posso prenotare un tavolo al ristorante?",
        "Mi puoi dire la ricetta per la pasta alla carbonara?",
        "Dove posso comprare dei vestiti a buon prezzo?",
        "Come posso migliorare il mio italiano?",
        "Qual è la migliore università in Italia?",
        "Puoi consigliarmi un buon libro da leggere?",
        "Come posso raggiungere il centro della città?"
    ]
    # Use pre-defined common topics
    negative_examples = [(random.choice(common_topics), 'Other') for _ in range(min(count // 2, len(common_topics)))]
    # Generate more diverse negative examples
    words_collection = [word for text, _ in positive_examples for word in text.split()]
    # Count word frequencies
    word_counter = Counter(words_collection)
    common_words = [word for word, count in word_counter.most_common(100) if word not in ITALIAN_STOPWORDS and len(word) > 3 and word not in TAX_TERMS]
    # Generate random sentences using common words but in different contexts
    for _ in range(count - len(negative_examples)):
        if common_words:
            sentence = " ".join(random.sample(common_words, min(random.randint(5, 12), len(common_words))))
            negative_examples.append((sentence, 'Other'))
    return negative_examples

'''
def prepare_training_data() -> List[Tuple[str, str]]:
    """
    Prepare combined training data from all sources.
    
    Returns:
        List of (text, label) tuples
    """
    # Load data from Excel (higher quality)
    excel_data = load_labeled_data_from_excel()
    
    # Extract data from PDFs
    pdf_data = extract_data_from_pdfs()
    
    # Combine all data sources
    all_data = excel_data + pdf_data
    
    # Check if we have positive examples
    positive_examples = [item for item in all_data if item[1] == 'Tax']
    
    # Generate negative examples if needed
    if positive_examples:
        # Create more negative examples to balance the dataset
        negative_examples = generate_negative_examples(
            positive_examples, 
            count=min(1000, len(positive_examples))
        )
        all_data.extend(negative_examples)
    
    # Add manual examples to strengthen tax classification
    manual_tax_examples = [
        ("Come funziona l'IVA per le fatture elettroniche?", "Tax"),
        ("Quali sono le aliquote IVA in Italia?", "Tax"),
        ("Devo pagare l'IVA su questo acquisto?", "Tax"),
        ("Come si registra una fattura a fini IVA?", "Tax"),
        ("Vorrei sapere come compilare la dichiarazione dei redditi", "Tax"),
        ("Mi spieghi il regime forfettario?", "Tax"),
        ("Qual è la differenza tra spese deducibili e detraibili?", "Tax"),
        ("Devo aprire partita IVA come freelancer", "Tax"),
        ("Ho ricevuto un avviso dall'Agenzia delle Entrate", "Tax"),
        ("Come funziona il rimborso IVA?", "Tax"),
        ("Fiscozen mi può aiutare con la dichiarazione?", "Tax"),
        ("Devo fare la fattura elettronica", "Tax"),
        ("Consigli su come gestire la mia partita IVA", "Tax"),
        ("Quali sono le scadenze fiscali di questo mese?", "Tax"),
        ("Come posso ridurre il carico fiscale?", "Tax")
    ]
    
    # Add each example multiple times to increase their weight
    for _ in range(5):
        all_data.extend(manual_tax_examples)
    
    # Shuffle the data
    random.shuffle(all_data)
    
    logger.info(f"Total training examples: {len(all_data)}")
    
    # Count examples by label
    labels_count = Counter([label for _, label in all_data])
    logger.info(f"Distribution by label: {labels_count}")
    
    return all_data
'''


def prepare_training_data() -> List[Tuple[str, str]]:
    """
    Prepare training data using Pinecone data and manual tax examples.
    Returns:
        List of (text, label) tuples
    """
    data = extract_data_from_pinecone()
    # Add manual examples to strengthen tax classification
    manual_tax_examples = [
        ("Come funziona l'IVA per le fatture elettroniche?", "Tax"),
        ("Quali sono le aliquote IVA in Italia?", "Tax"),
        ("Devo pagare l'IVA su questo acquisto?", "Tax"),
        ("Come si registra una fattura a fini IVA?", "Tax"),
        ("Vorrei sapere come compilare la dichiarazione dei redditi", "Tax"),
        ("Mi spieghi il regime forfettario?", "Tax"),
        ("Qual è la differenza tra spese deducibili e detraibili?", "Tax"),
        ("Devo aprire partita IVA come freelancer", "Tax"),
        ("Ho ricevuto un avviso dall'Agenzia delle Entrate", "Tax"),
        ("Come funziona il rimborso IVA?", "Tax"),
        ("Fiscozen mi può aiutare con la dichiarazione?", "Tax"),
        ("Devo fare la fattura elettronica", "Tax"),
        ("Consigli su come gestire la mia partita IVA", "Tax"),
        ("Quali sono le scadenze fiscali di questo mese?", "Tax"),
        ("Come posso ridurre il carico fiscale?", "Tax")
    ]
    for _ in range(5): data.extend(manual_tax_examples)  # Add each 5 times to weight them more heavily
    positives = [item for item in data if item[1] == 'Tax']
    # Generate negatives
    if positives: data.extend(generate_negative_examples(positives, count=min(1000, len(positives))))
    random.shuffle(data)
    logger.info(f"Total training examples: {len(data)}")
    return data

def train_fasttext_model(training_data: List[Tuple[str, str]], output_path: str, hyperparams: Dict = None) -> bool:
    """
    Train FastText model with optimized hyperparameters.
    
    Args:
        training_data: List of (text, label) tuples
        output_path: Path to save the trained model
        hyperparams: Dictionary of hyperparameters to use (optional)
        
    Returns:
        True if training succeeded, False otherwise
    """
    try:
        # Default hyperparameters (if none provided)
        hyperparams = hyperparams or {'epoch': 25, 'lr': 0.1, 'wordNgrams': 2, 'minCount': 1, 'dim': 100, 'loss': 'softmax'}
        # Create temporary file for training data
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8') as train_file:
            for text, label in training_data:
                # FastText expects format: __label__LABEL text
                train_file.write(f"__label__{label} {text}\n")
            train_file_path = train_file.name
        # Train the model
        model = fasttext.train_supervised(input=train_file_path, **hyperparams)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        model.save_model(output_path)
        os.unlink(train_file_path)
        
        # Test the model with some sample text to verify it works correctly
        sample_text = "Come funziona l'IVA per le fatture elettroniche?"
        prediction = model.predict(sample_text)
        logger.info(f"Sample prediction for '{sample_text}': {prediction}")
        
        # Check if prediction makes sense
        label = prediction[0][0].replace("__label__", "")
        if label != "Tax":
            logger.warning(f"⚠️ Model predicted '{label}' for a tax-related query. This might indicate a problem.")
        
        return True
    except Exception as e:
        logger.error(f"Error training FastText model: {e}")
        # Clean up
        if 'train_file_path' in locals():
            try:
                os.unlink(train_file_path)
            except:
                pass
        return False


def evaluate_model(model_path: str, test_data: List[Tuple[str, str]]) -> Dict:
    """
    Evaluate the trained model on test data.
    
    Args:
        model_path: Path to the trained model
        test_data: List of (text, label) tuples for testing
        
    Returns:
        Dictionary with evaluation metrics
    """
    try:
        import fasttext
        
        model = fasttext.load_model(model_path)
        
        # Prepare test data
        correct = 0
        total = len(test_data)
        results = {
            'predicted_labels': [],
            'true_labels': [],
            'texts': []
        }
        
        # Track metrics for each label
        label_metrics = {}
        
        for text, true_label in test_data:
            pred_label, prob = model.predict(text)
            pred_label = pred_label[0].replace('__label__', '')
            prob = prob[0]
            
            # Store results
            results['predicted_labels'].append(pred_label)
            results['true_labels'].append(true_label)
            results['texts'].append(text)
            
            # Update metrics
            if pred_label == true_label:
                correct += 1
            
            # Update per-label metrics
            if true_label not in label_metrics:
                label_metrics[true_label] = {'correct': 0, 'total': 0}
            label_metrics[true_label]['total'] += 1
            if pred_label == true_label:
                label_metrics[true_label]['correct'] += 1
        
        # Calculate accuracy
        accuracy = correct / total if total > 0 else 0
        
        # Calculate per-label metrics
        for label, metrics in label_metrics.items():
            metrics['accuracy'] = metrics['correct'] / metrics['total'] if metrics['total'] > 0 else 0
        
        # Return evaluation results
        return {
            'accuracy': accuracy,
            'total_examples': total,
            'correct_predictions': correct,
            'label_metrics': label_metrics,
            'detailed_results': results
        }
        
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return {
            'accuracy': 0,
            'error': str(e)
        }

def perform_hyperparameter_tuning(training_data: List[Tuple[str, str]], 
                                 validation_data: List[Tuple[str, str]]) -> Dict:
    """
    Perform hyperparameter tuning using a simple grid search.
    
    Args:
        training_data: Training data examples
        validation_data: Validation data examples
        
    Returns:
        Dictionary with best hyperparameters
    """
    try:
        # Temporary model path for tuning
        temp_model_path = os.path.join(tempfile.gettempdir(), "temp_fasttext_model.bin")
        best_accuracy = 0
        best_params = None
        # Try different combinations (simplified grid search)
        # Only test a small set of combinations to save time
        combinations = [
            {'epoch': 25, 'lr': 0.1, 'wordNgrams': 2, 'dim': 100, 'minCount': 1},
            {'epoch': 50, 'lr': 0.1, 'wordNgrams': 2, 'dim': 100, 'minCount': 1},
            {'epoch': 25, 'lr': 0.2, 'wordNgrams': 2, 'dim': 100, 'minCount': 1},
            {'epoch': 25, 'lr': 0.1, 'wordNgrams': 3, 'dim': 100, 'minCount': 1}
        ]
        
        for params in combinations:
            # Train model with these parameters
            success = train_fasttext_model(
                training_data=training_data,
                output_path=temp_model_path,
                hyperparams=params
            )
            
            if success:
                # Evaluate model
                eval_results = evaluate_model(temp_model_path, validation_data)
                accuracy = eval_results['accuracy']
                # Check if this is the best so far
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = params
        
        # Clean up temp model
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
        
        if best_params:
            logger.info(f"Best parameters found: {best_params} with accuracy: {best_accuracy:.4f}")
            return best_params
        else:
            logger.warning("No successful parameter combination found, using defaults")
            return {
                'epoch': 25,
                'lr': 0.1,
                'wordNgrams': 2,
                'dim': 100,
                'minCount': 1,
                'loss': 'softmax'
            }
            
    except Exception as e:
        logger.error(f"Error during hyperparameter tuning: {e}")
        return {
            'epoch': 25,
            'lr': 0.1, 
            'wordNgrams': 2,
            'dim': 100,
            'minCount': 1,
            'loss': 'softmax'
        }

def main():
    """Main function to train and evaluate the FastText model."""
    logger.info("🔹🔹🔹 IMPROVED FASTTEXT MODEL TRAINING 🔹🔹🔹")
    
    # Prepare training data from all sources
    all_data = prepare_training_data()
    
    if not all_data:
        logger.error("No training data available. Exiting.")
        return False
    
    # Split data for training and testing
    train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)
    
    # Further split training data for validation during hyperparameter tuning
    train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)
    # Perform hyperparameter tuning
    best_params = perform_hyperparameter_tuning(train_data, val_data)
    
    # Train final model with best parameters using all training data
    final_train_data = train_data + val_data  # Combine training and validation for final model
    success = train_fasttext_model(
        training_data=final_train_data,
        output_path=MODEL_PATH,
        hyperparams=best_params
    )
    
    if not success:
        logger.error("Failed to train final model")
        return False
    
    # Evaluate final model
    eval_results = evaluate_model(MODEL_PATH, test_data)
    
    # Report results
    logger.info(f"Final model accuracy: {eval_results['accuracy']:.4f}")
    logger.info("Per-label performance:")
    for label, metrics in eval_results['label_metrics'].items():
        logger.info(f"  {label}: {metrics['correct']}/{metrics['total']} = {metrics['accuracy']:.4f}")
    
    # Success!
    logger.info("✅ FastText model trained and evaluated successfully!")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        logger.error("❌ FastText model training failed")
        sys.exit(1)
    sys.exit(0) 