"""
Production deployment script for Fiscozen Chatbot
"""

import os
import sys
import subprocess
import shutil
from dotenv import load_dotenv

load_dotenv()

def check_environment():
    """Check environment and dependencies for production"""
    required_files = ["main.py", "utils.py", "relevance.py"]
    for file in required_files:
        if not os.path.exists(file):
            print(f"Error: Required file {file} not found!")
            return False
    
    # Check model directory
    if not os.path.exists("models/enhanced_bert"):
        print(f"Warning: models/enhanced_bert directory not found.")
        print("Will initialize model during startup.")
        print("Creating models directory...")
        os.makedirs("models/enhanced_bert", exist_ok=True)
    
    # Check environment variables
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("PINECONE_API_KEY"):
        print("Warning: API keys not found in environment variables.")
        print("Please set OPENAI_API_KEY and PINECONE_API_KEY environment variables.")
        if os.path.exists(".env"):
            print("Detected .env file. Loading variables from it...")
            try:
                from dotenv import load_dotenv
                load_dotenv()
                print("Loaded environment variables from .env file.")
            except ImportError:
                print("python-dotenv not installed. Installing...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])
                from dotenv import load_dotenv
                load_dotenv()
                print("Loaded environment variables from .env file.")
    
    # Remove any unnecessary integration files
    integration_files = [
        "simple_integration.py", 
        "integration_example.py",
        "test_improved_relevance.py",
        "document_trainer.py",
        "enhance_and_use.py",
        "run_local.py"
    ]
    
    for file in integration_files:
        if os.path.exists(file):
            print(f"Removing development file: {file}")
            os.remove(file)
    
    return True

def install_production_dependencies():
    """Install minimum required dependencies for production"""
    production_packages = [
        "streamlit",
        "streamlit-chat",
        "openai",
        "pinecone",
        "langchain",
        "langchain-openai",
        "langchain-pinecone",
        "transformers",
        "torch",
        "python-dotenv"
    ]
    
    print("Installing production dependencies...")
    for package in production_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError:
            print(f"Warning: Failed to install {package}")
    
    # Remove deprecated package
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "pinecone-client"])
    except:
        pass
    
    print("Dependencies installed!")

def initialize_model():
    """Initialize the BERT model for relevance checking"""
    model_path = "models/enhanced_bert"
    
    # Check if model already exists
    if os.path.exists(os.path.join(model_path, "config.json")):
        print(f"Model already exists at {model_path}")
        return True
    
    print("Initializing BERT model...")
    try:
        # Try using the dedicated initialization script if it exists
        if os.path.exists("initialize_model.py"):
            subprocess.check_call([sys.executable, "initialize_model.py"])
            return True
        
        # Fallback to manual initialization
        from transformers import BertTokenizer, BertForSequenceClassification
        
        # Create directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        
        # Download and save the model
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
        
        # Save the model and tokenizer
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        
        print(f"Model successfully initialized and saved to {model_path}")
        return True
    except Exception as e:
        print(f"Error initializing model: {e}")
        print("You can try running the initialize_model.py script separately before starting the server.")
        return False

def start_production_server():
    """Start the production server"""
    if not os.path.exists("main.py"):
        print("Error: main.py not found")
        return False
    
    print("\n=== Starting Fiscozen Chatbot (Production Mode) ===\n")
    
    # Set production flag
    os.environ["PRODUCTION"] = "true"
    
    # Run with Streamlit
    subprocess.call([sys.executable, "-m", "streamlit", "run", "main.py"])
    return True

if __name__ == "__main__":
    print("\n=== Fiscozen Chatbot Production Deployment ===\n")
    
    if not check_environment():
        print("Error: Environment check failed. Please fix the issues above.")
        sys.exit(1)
    
    install_production_dependencies()
    
    # Initialize model if necessary
    initialize_model()
    
    start_production_server() 