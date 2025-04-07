"""
Fiscozen Tax Chatbot - Main Application
"""

import os
import sys
import subprocess
import importlib
import pkg_resources
import re
import uuid
import time
import random
from pathlib import Path

# === Global Variables and Environment Detection ===
# Check if we're running in a test environment
is_test_environment = "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ

# Check if we're running directly (not through Streamlit)
is_streamlit_running = 'STREAMLIT_BROWSER_GATHER_USAGE_STATS' in os.environ

# === Core Setup Functions ===
def ensure_dependencies():
    """Check and install all required dependencies from requirements.txt"""
    print("\n=== Checking and Installing Dependencies ===")
    
    required_packages = [
        "streamlit", "streamlit-chat", "openai", "pinecone-client", 
        "python-dotenv", "langchain", "langchain-openai", "langchain-pinecone", 
        "fasttext", "tqdm", "PyPDF2"
    ]
    
    # Check if packages are installed
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package.replace("-", "_").split(">=")[0].split("==")[0])
        except ImportError:
            missing_packages.append(package)
    
    # Install missing packages
    if missing_packages:
        print(f"📦 Installing {len(missing_packages)} missing packages...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])
            print("✅ All required packages installed successfully!")
        except Exception as e:
            print(f"❌ Error installing dependencies: {e}")
            print("Please install dependencies manually with:")
            print("pip install streamlit streamlit-chat openai pinecone-client python-dotenv langchain langchain-openai langchain-pinecone fasttext tqdm PyPDF2")
            sys.exit(1)
    else:
        print("✅ All required packages are already installed!")

def setup_environment():
    """Set up the environment variables and directories"""
    print("\n=== Setting Up Environment ===")
    
    # Add current directory to path for imports
    project_root = os.path.abspath(os.path.dirname(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Create required directories
    os.makedirs("fasttext/models", exist_ok=True)
    os.makedirs("data_documents", exist_ok=True)
    os.makedirs("dtaa-documents", exist_ok=True)
    os.makedirs("argilla_data_49", exist_ok=True)
    os.makedirs("argilla-data", exist_ok=True)
    print("✅ Created required directories")
    
    # Create .env file if it doesn't exist
    if not os.path.exists(".env"):
        print("⚠️ No .env file found")
        print("Creating template .env file...")
        with open(".env", "w") as f:
            f.write("OPENAI_API_KEY=your_openai_api_key_here\nPINECONE_API_KEY=your_pinecone_api_key_here\n")
        print("✅ Created template .env file")
        print("⚠️ Please edit .env file with your actual API keys before continuing")
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Loaded environment variables from .env file")
    except Exception as e:
        print(f"❌ Error loading .env file: {e}")
    
    # Check for API keys
    api_keys_set = bool(os.getenv("OPENAI_API_KEY") and os.getenv("PINECONE_API_KEY"))
    if not api_keys_set and not is_test_environment:
        print("⚠️ API keys not found in environment!")
        print("Please edit the .env file with your actual API keys")
        choice = input("Do you want to continue anyway? (y/n): ")
        if choice.lower() != 'y':
            sys.exit(1)

def train_fasttext_model():
    """Train the FastText model with documents if available"""
    print("\n=== FastText Model Preparation ===")
    
    # Skip in test environment
    if is_test_environment:
        print("🧪 Test environment detected - skipping FastText training")
        return
    
    # Check for document folders
    data_dirs = {
        "data_documents": os.path.exists("data_documents") and any(os.listdir("data_documents")),
        "argilla_data_49": os.path.exists("argilla_data_49") and any(os.listdir("argilla_data_49"))
    }
    
    # Check if any directory has data
    has_data = any(data_dirs.values())
    
    print("Checking for document data in these directories:")
    for dir_name, has_content in data_dirs.items():
        status = "✅ Found data" if has_content else "❌ No data found"
        print(f"  - {dir_name}: {status}")
    
    if has_data:
        # Provide detailed info about the data sources found
        data_sources = []
        for dir_name, has_content in data_dirs.items():
            if has_content:
                doc_count = len(os.listdir(dir_name))
                data_sources.append(f"{dir_name} ({doc_count} files)")
            
        print(f"📚 Document data found in: {', '.join(data_sources)}")
        print("Preparing to train FastText classifier with all available data...")
        
        # Create missing directories that might be expected by the training script
        for dir_name in data_dirs.keys():
            os.makedirs(dir_name, exist_ok=True)
        
        # Check if training script exists
        if os.path.exists("fast_text/train_with_real_data.py"):
            print("🔄 Starting FastText training process...")
            try:
                # Run the training script
                subprocess.check_call([sys.executable, "fast_text/train_with_real_data.py"])
                print("✅ FastText classifier training completed successfully!")
            except subprocess.CalledProcessError as e:
                print(f"❌ FastText classifier training failed: {e}")
                choice = input("Do you want to continue without training? (y/n): ")
                if choice.lower() != 'y':
                    sys.exit(1)
        else:
            print("❌ Training script not found: fast_text/train_with_real_data.py")
            print("Continuing with default FastText model...")
    else:
        print("ℹ️ No document data found in any data directory.")
        print("Creating sample document directories for future use...")
        
        # Create all possible data directories
        for dir_name in data_dirs.keys():
            os.makedirs(dir_name, exist_ok=True)
        
        # Create a README file explaining how to add documents
        with open("data_documents/README.txt", "w") as f:
            f.write("""
FISCOZEN CHATBOT TRAINING DATA
==============================

Place your training documents in this directory to train the FastText classifier.

Supported formats:
- PDF documents (.pdf)
- Excel files (.xlsx) with labeled conversations
- Text files (.txt)

For best results:
1. Organize documents by topic
2. Include a variety of text samples for each category (IVA, Other)
3. Make sure documents are in Italian
4. Aim for at least 10-20 documents per category

After adding documents, run `python main.py` to retrain the model.
""")
        
        print("✅ Created document directories with instructions")
        print("Using default FastText model without custom training")
    
    # Ensure FastText model exists
    model_path = "fast_text/models/tax_classifier.bin"
    if not os.path.exists(model_path):
        print("❌ FastText model not found after training")
        print("Please ensure the training script completed successfully")
        sys.exit(1)
    else:
        print("✅ FastText model ready for use")

def launch_chatbot():
    """Launch the Streamlit chatbot"""
    print("\n=== Launching Fiscozen Tax Chatbot ===")
    
    if is_streamlit_running:
        print("🚀 Streamlit is already running!")
        return
    
    print("🚀 Starting Streamlit server...")
    try:
        # Kill any existing Streamlit processes
        subprocess.call(["pkill", "-f", "streamlit"])
        time.sleep(2)  # Wait for processes to be killed
        
        # Set headless mode for server
        os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
        
        # Launch Streamlit with specific port and app file
        subprocess.call([
            sys.executable, 
            "-m", 
            "streamlit", 
            "run",
            "app.py",  # Use the new app file
            "--server.port=8501",
            "--server.address=localhost"
        ])
    except Exception as e:
        print(f"❌ Failed to start Streamlit server: {e}")
        sys.exit(1)

# === Main Execution Logic ===
if not is_streamlit_running and not is_test_environment:
    # We're running directly (not via Streamlit or tests)
    print("\n🔹🔹🔹 FISCOZEN TAX CHATBOT SETUP 🔹🔹🔹\n")
    
    # Step 1: Install dependencies
    ensure_dependencies()
    
    # Step 2: Set up environment (directories, .env file)
    setup_environment()
    
    # Step 3: Train FastText model if document data is available
    train_fasttext_model()
    
    # Step 4: Launch the chatbot
    launch_chatbot()
    
    # Exit this execution (Streamlit will start a new process)
    sys.exit(0)

# === API Function ===
def get_response(user_input: str, conversation_id: str = "api_user") -> str:
    """Get response from the chatbot"""
    try:
        # Initialize relevance checker
        relevance_checker = FastTextRelevanceChecker()
        
        # Check if the query is relevant to tax/IVA topics
        is_relevant, details = relevance_checker.is_relevant(user_input)
        
        if not is_relevant:
            # If not relevant, provide a polite response
            return "Mi dispiace, ma posso rispondere solo a domande relative a tasse, IVA e questioni fiscali. Posso aiutarti con domande su questi argomenti?"
        
        # If relevant, proceed with normal chatbot response
        conversation_string = get_conversation_string(conversation_id)
        refined_query = query_refiner(conversation_string, user_input)
        response = find_match(refined_query, 2)
        return response
    except Exception as e:
        print(f"Error in get_response: {e}")
        return "Mi dispiace, si è verificato un errore. Per favore, riprova."
