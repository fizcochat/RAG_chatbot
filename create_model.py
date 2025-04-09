import fasttext
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_training_data():
    """Create training data for the FastText model."""
    training_data = [
        "__label__IVA Come funziona l'IVA?",
        "__label__IVA Come si calcola l'IVA?",
        "__label__IVA Quali sono le aliquote IVA?",
        "__label__IVA Come si paga l'IVA?",
        "__label__IVA Come funziona la partita IVA?",
        "__label__Other Che tempo fa oggi?",
        "__label__Other Come si fa la pasta?",
        "__label__Other Qual è la capitale della Francia?",
        "__label__Other Come funziona un motore?",
        "__label__Other Qual è il tuo colore preferito?"
    ]
    
    with open('tax_classifier.txt', 'w', encoding='utf-8') as f:
        for line in training_data:
            f.write(line + '\n')
    
    logger.info("Training data created successfully")

def train_model():
    """Train the FastText model and save it."""
    try:
        # Train model
        model = fasttext.train_supervised('tax_classifier.txt')
        
        # Save model
        model_path = '/app/fast_text/models/tax_classifier.bin'
        model.save_model(model_path)
        
        # Verify model exists
        if os.path.exists(model_path):
            logger.info(f"Model created successfully at {model_path}")
        else:
            raise FileNotFoundError(f"Model file not created at {model_path}")
            
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        create_training_data()
        train_model()
    except Exception as e:
        logger.error(f"Error in model creation process: {str(e)}")
        raise 