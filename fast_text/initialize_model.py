"""
Initialize a FastText model for text classification.

This script creates a basic FastText model with default settings if no trained model exists.
"""

import os
import sys
import tempfile

# Add parent directory to path for imports
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def initialize_fasttext_model():
    """Initialize a basic FastText model if none exists."""
    # Set output directory and model path
    output_dir = "fast_text/models"
    model_path = os.path.join(output_dir, "tax_classifier.bin")
    
    # Check if model already exists
    if os.path.exists(model_path):
        print(f"FastText model already exists at {model_path}")
        return True
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating a basic FastText model with minimal training data...")
    
    # Create minimal training data
    minimal_training_data = [
        ("Come funziona l'IVA in Italia?", "IVA"),
        ("Quali sono le aliquote IVA?", "IVA"),
        ("Come registrarsi per l'IVA?", "IVA"),
        ("Rimborso IVA per acquisti", "IVA"),
        ("IVA detraibile per le aziende", "IVA"),
        
        ("Servizi offerti da Fiscozen", "Fiscozen"),
        ("Come funziona la piattaforma Fiscozen?", "Fiscozen"),
        ("Quanto costa usare Fiscozen?", "Fiscozen"),
        ("Consulenti fiscali Fiscozen", "Fiscozen"),
        ("Vantaggi di Fiscozen per partite IVA", "Fiscozen"),
        
        ("Che tempo fa oggi?", "Other"),
        ("Mi racconti una storia?", "Other"),
        ("Qual è la capitale della Francia?", "Other"),
        ("Come si prepara la pasta?", "Other"),
        ("Consigliami un film da guardare", "Other")
    ]
    
    try:
        import fasttext
        
        # Create a temporary file for training data
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            for text, label in minimal_training_data:
                # FastText expects format: __label__LABEL text
                temp_file.write(f"__label__{label} {text}\n")
            temp_file_path = temp_file.name
        
        # Train a simple model
        print("Training model with minimal data...")
        model = fasttext.train_supervised(
            input=temp_file_path,
            epoch=10,
            lr=0.1,
            wordNgrams=2,
            verbose=1,
            minCount=1
        )
        
        # Save the model
        model.save_model(model_path)
        print(f"Basic FastText model initialized and saved to {model_path}")
        
        # Clean up
        os.unlink(temp_file_path)
        
        return True
    
    except Exception as e:
        print(f"Error initializing FastText model: {e}")
        # Clean up
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        return False

if __name__ == "__main__":
    print("\n🔹🔹🔹 INITIALIZING FASTTEXT MODEL 🔹🔹🔹\n")
    
    # Initialize the model
    success = initialize_fasttext_model()
    
    if success:
        print("\n✅ FastText model initialized successfully!")
    else:
        print("\n❌ FastText model initialization failed!")
        sys.exit(1) 