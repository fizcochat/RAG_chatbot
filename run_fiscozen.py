#!/usr/bin/env python3
"""
Fiscozen Chatbot - Main Entry Point

This script provides a simple way to initialize and run the Fiscozen chatbot.
It handles all the setup, dependency installation, and model initialization.

Usage:
    python run_fiscozen.py

This will:
1. Check and install required dependencies
2. Initialize the BERT model if needed
3. Start the Streamlit server with the chatbot interface
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_header():
    """Print a fancy header for the chatbot"""
    print("\n" + "=" * 60)
    print("                FISCOZEN TAX CHATBOT")
    print("=" * 60)
    print("  A specialized chatbot for Italian tax matters")
    print("  Focusing on IVA regulations and Fiscozen services")
    print("=" * 60 + "\n")

def check_environment():
    """Check if the environment is properly set up"""
    print("🔍 Checking environment...")
    
    # Check for required files
    required_files = ["main.py", "utils.py", "bert/relevance.py", "bert/__init__.py"]
    missing_files = [file for file in required_files if not os.path.exists(file)]
    
    if missing_files:
        print(f"❌ Error: Missing required files: {', '.join(missing_files)}")
        return False
    
    # Check for .env file
    if not os.path.exists(".env"):
        print("⚠️ Warning: No .env file found. API keys may need to be set manually.")
    else:
        # Try to load .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
            print("✅ Loaded environment variables from .env file")
        except ImportError:
            print("⚠️ Warning: python-dotenv not installed, can't load .env file automatically")
    
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Warning: OPENAI_API_KEY not found in environment variables")
    if not os.getenv("PINECONE_API_KEY"):
        print("⚠️ Warning: PINECONE_API_KEY not found in environment variables")
    
    # Create model directories if they don't exist
    os.makedirs("bert/models/enhanced_bert", exist_ok=True)
    
    print("✅ Environment check completed\n")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Checking and installing dependencies...")
    
    required_packages = [
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
    
    # Check which packages need installation
    to_install = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} is already installed")
        except ImportError:
            to_install.append(package)
    
    # Install missing packages
    if to_install:
        print(f"📦 Installing {len(to_install)} missing packages...")
        for package in to_install:
            print(f"   Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"   ✅ {package} installed successfully")
            except subprocess.CalledProcessError:
                print(f"   ❌ Failed to install {package}")
    else:
        print("✅ All required packages are already installed")
    
    # Try to remove deprecated packages
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "pinecone-client"], 
                             stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        print("✅ Removed deprecated pinecone-client package")
    except:
        pass
    
    print("✅ Dependency check completed\n")

def initialize_bert_model():
    """Initialize the BERT model if needed"""
    print("🤖 Checking BERT model...")
    
    model_path = "bert/models/enhanced_bert"
    
    # Check if model files exist
    if os.path.exists(os.path.join(model_path, "config.json")):
        print("✅ BERT model is already initialized")
        return True
    
    print("⏳ Initializing BERT model (this may take a moment)...")
    
    try:
        # Try running the dedicated initialization script
        if os.path.exists("bert/initialize_model.py"):
            subprocess.check_call([sys.executable, "bert/initialize_model.py"])
            print("✅ BERT model initialization completed")
            return True
        
        # Fallback to manual initialization
        print("⏳ Downloading BERT model from Hugging Face...")
        from transformers import BertTokenizer, BertForSequenceClassification
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
        
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        
        print("✅ BERT model initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Error initializing BERT model: {e}")
        return False

def start_chatbot():
    """Start the Streamlit server with the chatbot"""
    print("🚀 Starting Fiscozen Chatbot...")
    
    if not os.path.exists("main.py"):
        print("❌ Error: main.py not found")
        return False
    
    print("\n" + "=" * 60)
    print("  Fiscozen Chatbot is starting up!")
    print("  You can access it in your web browser shortly.")
    print("=" * 60 + "\n")
    
    # Set production environment variable
    os.environ["PRODUCTION"] = "true"
    
    # Start Streamlit server
    subprocess.call([sys.executable, "-m", "streamlit", "run", "main.py"])
    return True

if __name__ == "__main__":
    print_header()
    
    if not check_environment():
        print("❌ Environment check failed. Please fix the issues above.")
        sys.exit(1)
    
    install_dependencies()
    
    if not initialize_bert_model():
        print("⚠️ Warning: BERT model initialization failed. The chatbot may not function correctly.")
    
    start_chatbot() 