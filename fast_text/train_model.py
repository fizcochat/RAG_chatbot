"""
Train the FastText model using real data from the labeled conversations.
"""

import os
import sys
import pandas as pd
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Add parent directory to path for imports
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def load_training_data():
    """Load and prepare training data from the Excel file."""
    try:
        # Load the Excel file
        excel_path = 'data_documents/worker_conversations_labeled_translated_file.xlsx'
        logging.info(f"Loading Excel file from: {excel_path}")
        
        df = pd.read_excel(excel_path)
        
        # Log DataFrame info
        logging.info("DataFrame columns:")
        logging.info(df.columns.tolist())
        logging.info(f"\nFirst few rows:\n{df.head()}")
        
        # Prepare training data
        training_data = []
        
        # Process each row
        for idx, row in df.iterrows():
            try:
                text = str(row.get('Conversazione', '')).strip()
                label = str(row.get('Label', '')).strip()
                
                if text and label:
                    # Convert numeric label to category
                    if label == '1':  # Tax/IVA related
                        category = 'IVA'
                    elif 'fiscozen' in text.lower():  # Check text for Fiscozen mentions
                        category = 'Fiscozen'
                    else:
                        category = 'Other'
                    
                    # Clean up the text - remove "User:" and "Chatbot:" prefixes
                    text = text.replace('User:', '').replace('Chatbot:', '')
                    text = ' '.join(text.split())  # Normalize whitespace
                    
                    training_data.append((text, category))
                    
            except Exception as row_error:
                logging.error(f"Error processing row {idx}: {row_error}")
                continue
        
        logging.info(f"Processed {len(training_data)} training examples")
        if training_data:
            # Log some examples
            logging.info("\nFirst few training examples:")
            for text, label in training_data[:3]:
                logging.info(f"Text: {text[:100]}...")
                logging.info(f"Label: {label}\n")
        
        return training_data
        
    except Exception as e:
        logging.error(f"Error loading training data: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def train_fasttext_model(training_data, output_path):
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
        print("Training FastText model with real data...")
        model = fasttext.train_supervised(
            input=temp_file_path,
            epoch=25,  # More epochs for better training
            lr=0.1,
            wordNgrams=2,
            verbose=1,
            minCount=1
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
    
    # Set model path
    model_path = "fast_text/models/tax_classifier.bin"
    
    # Load training data
    print("Loading training data...")
    training_data = load_training_data()
    
    if training_data:
        print(f"Loaded {len(training_data)} training examples")
        
        # Train the model
        success = train_fasttext_model(training_data, model_path)
        
        if success:
            print("\n✅ FastText model trained successfully!")
        else:
            print("\n❌ FastText model training failed!")
            sys.exit(1)
    else:
        print("\n❌ Failed to load training data!")
        sys.exit(1) 