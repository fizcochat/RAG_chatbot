"""
Train the FastText model using real data from Excel and PDF files.
"""

import os
import sys
import pandas as pd
import PyPDF2
import tempfile
import logging
import re
import random
from typing import List, Tuple, Set

# Configure logging
logging.basicConfig(level=logging.INFO)

# Add non-tax examples for better balance
NON_TAX_EXAMPLES = [
    ("Come si prepara una buona pasta alla carbonara?", "Other"),
    ("Che tempo farà domani a Roma?", "Other"),
    ("Quali sono i migliori ristoranti in città?", "Other"),
    ("Mi puoi consigliare un buon film da vedere?", "Other"),
    ("Come posso migliorare il mio inglese?", "Other"),
    ("Dove posso trovare un idraulico?", "Other"),
    ("Quali sono gli orari del supermercato?", "Other"),
    ("Come si allena un cane?", "Other"),
    ("Quali sono i migliori hotel a Milano?", "Other"),
    ("Come si fa il pane in casa?", "Other"),
    ("Quali sono i monumenti più famosi di Roma?", "Other"),
    ("Come si coltivano i pomodori?", "Other"),
    ("Dove posso comprare dei mobili usati?", "Other"),
    ("Come si cambia una ruota dell'auto?", "Other"),
    ("Quali sono i migliori libri del 2024?", "Other"),
    ("Come si fa la pizza napoletana?", "Other"),
    ("Dove posso trovare un buon parrucchiere?", "Other"),
    ("Come si prepara il tiramisù?", "Other"),
    ("Quali sono le migliori spiagge in Sicilia?", "Other"),
    ("Come si fa il compost?", "Other"),
    ("Quali sono i migliori vini italiani?", "Other"),
    ("Come si prepara il caffè con la moka?", "Other"),
    ("Dove posso comprare prodotti biologici?", "Other"),
    ("Come si fa la pasta fatta in casa?", "Other"),
    ("Quali sono le sagre più famose in Italia?", "Other"),
    ("Come si coltivano le erbe aromatiche?", "Other"),
    ("Dove posso trovare un corso di yoga?", "Other"),
    ("Come si prepara il pesto alla genovese?", "Other"),
    ("Quali sono i migliori mercati rionali?", "Other"),
    ("Come si fa il limoncello?", "Other"),
    ("Quali sono i migliori ristoranti di pesce?", "Other"),
    ("Come si prenota un biglietto aereo?", "Other"),
    ("Dove posso trovare un dentista?", "Other"),
    ("Come si fa la marmellata in casa?", "Other"),
    ("Quali sono i migliori musei di Firenze?", "Other"),
    ("Come si coltivano le orchidee?", "Other"),
    ("Dove posso comprare vestiti vintage?", "Other"),
    ("Come si prepara un cocktail mojito?", "Other"),
    ("Quali sono le migliori palestre in zona?", "Other"),
    ("Come si fa il gelato in casa?", "Other"),
    ("Dove posso trovare un veterinario?", "Other"),
    ("Come si prepara la focaccia genovese?", "Other"),
    ("Quali sono i migliori parchi a Milano?", "Other"),
    ("Come si fa la birra artigianale?", "Other"),
    ("Dove posso comprare piante da giardino?", "Other"),
    ("Come si prepara il risotto ai funghi?", "Other"),
    ("Quali sono i migliori concerti del 2024?", "Other"),
    ("Come si fa il sapone naturale?", "Other"),
    ("Dove posso trovare un meccanico?", "Other"),
    ("Come si prepara il pane al cioccolato?", "Other"),
    ("Come si scrive un business plan?", "Other"),
    ("Quali sono le migliori strategie di marketing?", "Other"),
    ("Come si gestisce un team di lavoro?", "Other"),
    ("Dove posso trovare investitori?", "Other"),
    ("Come si fa un'analisi di mercato?", "Other"),
    ("Quali sono i migliori software di project management?", "Other"),
    ("Come si registra un marchio?", "Other"),
    ("Dove posso trovare fornitori affidabili?", "Other"),
    ("Come si fa un piano finanziario?", "Other"),
    ("Quali sono le migliori tecniche di vendita?", "Other"),
]

def load_excel_data(excel_path: str) -> List[Tuple[str, str]]:
    """Load and prepare training data from the Excel file."""
    try:
        # Load the Excel file
        df = pd.read_excel(excel_path)
        
        # Prepare training data
        training_data = []
        
        # Process each row
        for _, row in df.iterrows():
            text = str(row.get('Conversazione', '')).strip()
            label = str(row.get('Label', '')).strip()
            
            if text and label:
                # Convert numeric label to category
                if label == '1':  # Tax/IVA related
                    category = 'IVA'
                elif 'fiscozen' in text.lower():  # Check text for Fiscozen mentions
                    category = 'IVA'
                else:
                    category = 'Other'
                
                # Clean up the text - remove "User:" and "Chatbot:" prefixes
                text = text.replace('User:', '').replace('Chatbot:', '')
                text = ' '.join(text.split())  # Normalize whitespace
                
                training_data.append((text, category))
        
        logging.info(f"Loaded {len(training_data)} examples from Excel file")
        return training_data
        
    except Exception as e:
        logging.error(f"Error loading Excel data: {e}")
        return []

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        logging.error(f"Error reading PDF {pdf_path}: {e}")
        return ""

def process_pdf_text(text: str) -> List[str]:
    """Process PDF text into sentences or meaningful chunks."""
    # Remove extra whitespace and normalize line endings
    text = re.sub(r'\s+', ' ', text)
    
    # Split into sentences (simple approach)
    sentences = re.split(r'[.!?]+', text)
    
    # Filter and clean sentences
    valid_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        # Only keep sentences with reasonable length and content
        if len(sentence.split()) >= 5 and len(sentence) >= 20:
            valid_sentences.append(sentence)
    
    return valid_sentences

def process_pdf_directory(directory: str, label: str) -> List[Tuple[str, str]]:
    """Process all PDFs in a directory and create training examples."""
    training_data = []
    
    try:
        for filename in os.listdir(directory):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(directory, filename)
                logging.info(f"Processing PDF: {filename}")
                
                # Extract and process text
                text = extract_text_from_pdf(pdf_path)
                sentences = process_pdf_text(text)
                
                # Add to training data
                for sentence in sentences:
                    training_data.append((sentence, label))
                
                logging.info(f"Added {len(sentences)} examples from {filename}")
    except Exception as e:
        logging.error(f"Error processing directory {directory}: {e}")
    
    return training_data

def balance_dataset(training_data: List[Tuple[str, str]], max_ratio: float = 3.0) -> List[Tuple[str, str]]:
    """Balance the dataset to avoid extreme class imbalance."""
    # Count examples per class
    class_counts = {}
    for _, label in training_data:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    # Find minority and majority classes
    min_class = min(class_counts.items(), key=lambda x: x[1])[0]
    max_class = max(class_counts.items(), key=lambda x: x[1])[0]
    
    # Calculate target size for majority class
    target_size = int(class_counts[min_class] * max_ratio)
    
    # Separate examples by class
    examples_by_class = {label: [] for label in class_counts.keys()}
    for example in training_data:
        examples_by_class[example[1]].append(example)
    
    # Balance the dataset
    balanced_data = []
    for label, examples in examples_by_class.items():
        if label == max_class and len(examples) > target_size:
            # Randomly sample from majority class
            balanced_data.extend(random.sample(examples, target_size))
        else:
            balanced_data.extend(examples)
    
    # Shuffle the balanced dataset
    random.shuffle(balanced_data)
    return balanced_data

def train_fasttext_model(training_data: List[Tuple[str, str]], output_path: str) -> bool:
    """Train FastText model with the provided data."""
    try:
        import fasttext
        
        # Create temporary file for training data
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            for text, label in training_data:
                # FastText expects format: __label__LABEL text
                temp_file.write(f"__label__{label} {text}\n")
            temp_file_path = temp_file.name
        
        # Train the model
        print("\nTraining FastText model with real data...")
        model = fasttext.train_supervised(
            input=temp_file_path,
            epoch=20,      # Increased epochs for better learning
            lr=0.8,        # Increased learning rate for better convergence
            wordNgrams=3,  # Increased to capture more context
            verbose=1,
            minCount=2,    # Keep minCount at 2 to reduce noise
            loss='hs',     # Use hierarchical softmax
            dim=200,       # Increased embedding dimension
            bucket=200000  # Increased number of buckets
        )
        
        # Save the model
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        model.save_model(output_path)
        print(f"FastText model trained and saved to {output_path}")
        
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

if __name__ == "__main__":
    print("\n🔹🔹🔹 TRAINING FASTTEXT MODEL WITH REAL DATA 🔹🔹🔹\n")
    
    # Set paths
    excel_path = "data_documents/worker_conversations_labeled_translated_file.xlsx"
    agenzia_dir = "data_documents"
    argilla_dir = "argilla_data_49"
    model_path = "fast_text/models/tax_classifier.bin"
    
    # Load training data from different sources
    training_data = []
    
    # 1. Load Excel data
    print("Loading data from Excel file...")
    excel_data = load_excel_data(excel_path)
    training_data.extend(excel_data)
    
    # 2. Process Agenzia Entrate PDFs
    print("\nProcessing Agenzia Entrate PDFs...")
    agenzia_data = process_pdf_directory(agenzia_dir, "IVA")
    training_data.extend(agenzia_data)
    
    # 3. Process Argilla PDFs
    print("\nProcessing Argilla PDFs...")
    argilla_data = process_pdf_directory(argilla_dir, "IVA")
    training_data.extend(argilla_data)
    
    # 4. Add non-tax examples
    print("\nAdding non-tax examples...")
    training_data.extend(NON_TAX_EXAMPLES)
    
    # 5. Balance the dataset
    print("\nBalancing the dataset...")
    balanced_data = balance_dataset(training_data, max_ratio=1.5)
    
    # Print statistics
    tax_count = sum(1 for _, label in balanced_data if label == "IVA")
    other_count = sum(1 for _, label in balanced_data if label == "Other")
    
    print("\n📊 Training Data Statistics:")
    print(f"Tax-related examples: {tax_count}")
    print(f"Non-tax examples: {other_count}")
    print(f"Total training examples: {len(balanced_data)}")
    print(f"Class ratio (Tax:Other): {tax_count/other_count:.2f}")
    
    # Train the model
    if balanced_data:
        success = train_fasttext_model(balanced_data, model_path)
        
        if success:
            print("\n✅ FastText model trained successfully!")
        else:
            print("\n❌ FastText model training failed!")
            sys.exit(1)
    else:
        print("\n❌ No training data available!")
        sys.exit(1) 