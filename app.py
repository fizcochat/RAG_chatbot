"""
Fiscozen Tax Chatbot - Streamlit App
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
import psutil
from pathlib import Path

import streamlit as st
from streamlit_chat import message
from fast_text.relevance import FastTextRelevanceChecker
from utils import initialize_services, find_match, query_refiner, get_conversation_string, translate_to_italian, translate_from_italian
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv
import base64

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

def train_fasttext_model():
    """Train the FastText model with documents if available"""
    print("\n=== FastText Model Preparation ===")
    
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
        
        print("✅ Created document directories")
        print("Using default FastText model without custom training")
    
    # Ensure FastText model exists
    model_path = "fast_text/models/tax_classifier.bin"
    if not os.path.exists(model_path):
        print("❌ FastText model not found after training")
        print("Please ensure the training script completed successfully")
    else:
        print("✅ FastText model ready for use")

# Run setup functions if this is being run directly
if __name__ == "__main__":
    # Check if environment variable indicates we're running through streamlit
    if 'STREAMLIT_BROWSER_GATHER_USAGE_STATS' in os.environ:
        print("🚀 Running via streamlit run app.py")
        # Run setup only when directly called with streamlit
        ensure_dependencies()
        setup_environment()
        train_fasttext_model()

# Load environment variables
load_dotenv()

# Initialize services
try:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    
    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        st.error("Please set up your API keys in the .env file")
        st.stop()
        
    # Initialize services and store in session state
    if 'vectorstore' not in st.session_state or 'openai_client' not in st.session_state:
        vectorstore, client = initialize_services(OPENAI_API_KEY, PINECONE_API_KEY)
        st.session_state['vectorstore'] = vectorstore
        st.session_state['openai_client'] = client
    
    # Initialize LLM
    if 'llm' not in st.session_state:
        st.session_state['llm'] = ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY)
    
    # Initialize conversation memory
    if 'memory' not in st.session_state:
        st.session_state['memory'] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
except Exception as e:
    st.error(f"Error initializing services: {e}")
    st.stop()

# Initialize relevance checker
try:
    if 'relevance_checker' not in st.session_state:
        st.session_state['relevance_checker'] = FastTextRelevanceChecker()
except Exception as e:
    st.error(f"Error initializing relevance checker: {e}")
    st.stop()

# Function to process a query and get a response
def process_query(query, language="it"):
    try:
        # Translate the query to Italian if the user is using English
        original_query = query
        if language == "en":
            query = translate_to_italian(query)
            print(f"Translated query: {query}")
        
        # Check relevance
        is_relevant, details = st.session_state['relevance_checker'].is_relevant(query)
        
        if not is_relevant:
            if language == "it":
                response = "Mi dispiace, ma posso rispondere solo a domande relative a tasse, IVA e questioni fiscali. Posso aiutarti con domande su questi argomenti?"
            else:  # English
                response = "I'm sorry, but I can only answer questions about taxes, VAT, and fiscal matters. Can I help you with questions on these topics?"
        else:
            # Process relevant query
            conversation_string = get_conversation_string()
            refined_query = query_refiner(conversation_string, query)
            italian_response = find_match(refined_query)
            
            # Translate the response if the user is using English
            if language == "en":
                response = translate_from_italian(italian_response)
            else:
                response = italian_response
            
            # Update conversation memory (store in original language)
            st.session_state['memory'].save_context(
                {"input": original_query},
                {"answer": response}
            )
        
        return response
    except Exception as e:
        st.error(f"Error processing query: {e}")
        if language == "it":
            return "Mi dispiace, si è verificato un errore. Per favore, riprova."
        else:  # English
            return "I'm sorry, an error occurred. Please try again."

# Page config
st.set_page_config(
    page_title="Fiscozen Tax Chatbot",
    page_icon="images/fiscozen_small.png",
    layout="centered"
)

# Function to convert image to base64 for HTML embedding
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Initialize language selection in session state
if 'language' not in st.session_state:
    st.session_state.language = "it"  # Default to Italian

# Initialize session state for chat
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'processing' not in st.session_state:
    st.session_state.processing = False

# Get base64 encoded images for chat icons
try:
    fiscozen_logo_base64 = get_image_base64("images/fiscozen_logo.jpeg")
    fiscozen_small_base64 = get_image_base64("images/fiscozen_small.png")
except Exception as e:
    print(f"Error loading images: {e}")
    fiscozen_logo_base64 = ""
    fiscozen_small_base64 = ""

# Add minimalist CSS to hide profile images
st.markdown("""
<style>
img[alt="profile"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# Sidebar for language selection
with st.sidebar:
    st.image("images/fiscozen_small.png", width=50)
    st.title("Settings")
    
    # Language selector
    selected_language = st.selectbox(
        "Language / Lingua",
        options=["Italian / Italiano", "English / Inglese"],
        index=0 if st.session_state.language == "it" else 1
    )
    
    # Update language in session state
    if selected_language == "Italian / Italiano" and st.session_state.language != "it":
        st.session_state.language = "it"
        st.rerun()
    elif selected_language == "English / Inglese" and st.session_state.language != "en":
        st.session_state.language = "en"
        st.rerun()
    
    st.divider()
    
    # Add information about the chatbot
    if st.session_state.language == "it":
        st.write("**Informazioni**")
        st.write("Questo assistente risponde a domande su tasse, IVA e questioni fiscali in Italia.")
    else:
        st.write("**Information**")
        st.write("This assistant answers questions about taxes, VAT, and fiscal matters in Italy.")

# Logo only, no title, larger and centered
st.image("images/fiscozen_logo.jpeg", width=200)

# Welcome Section - change text based on language
if st.session_state.language == "it":
    st.markdown("""
    ### Benvenuto nel Tax Assistant di Fiscozen!
    Sono qui per aiutarti con le tue domande fiscali. Posso supportarti su:
    """)
    
    # List of features with checkmarks in Italian
    st.markdown("""
    - ✓ IVA e regime fiscale italiano
    - ✓ Detrazioni e deduzioni fiscali
    - ✓ Dichiarazione dei redditi
    - ✓ Servizi offerti da Fiscozen
    """)
    
    # Restart button text in Italian
    restart_label = "↻ Ricomincia"
    
    # Input placeholder in Italian
    input_placeholder = "Chiedimi qualcosa sul fisco italiano..."
    
    # Suggested questions in Italian
    suggested_questions = [
        "Come funziona l'IVA per i liberi professionisti?",
        "Quali detrazioni fiscali posso avere per i figli?",
        "Come gestire le fatture elettroniche?",
        "Cosa offre Fiscozen?"
    ]
    
    suggested_title = "Domande Frequenti:"
else:
    st.markdown("""
    ### Welcome to Fiscozen's Tax Assistant!
    I'm here to help you with your tax questions. I can support you on:
    """)
    
    # List of features with checkmarks in English
    st.markdown("""
    - ✓ Italian VAT and tax regime
    - ✓ Tax deductions and allowances
    - ✓ Income tax declarations
    - ✓ Services offered by Fiscozen
    """)
    
    # Restart button text in English
    restart_label = "↻ Restart"
    
    # Input placeholder in English
    input_placeholder = "Ask me something about Italian taxation..."
    
    # Suggested questions in English
    suggested_questions = [
        "How does VAT work for freelancers?",
        "What tax deductions can I get for children?",
        "How to manage electronic invoices?",
        "What does Fiscozen offer?"
    ]
    
    suggested_title = "Frequently Asked Questions:"

# Restart button at the top
restart_clicked = st.button(restart_label)
if restart_clicked:
    st.session_state.chat_history = []
    if 'memory' in st.session_state:
        st.session_state['memory'].clear()
    st.rerun()

# Chat container
st.divider()

# Display chat history
for i, message_obj in enumerate(st.session_state.chat_history):
    message(
        message_obj["message"],
        is_user=message_obj["is_user"],
        key=message_obj["key"],
        avatar_style="none"  # This disables the avatar
    )

# Show typing indicator during processing
if st.session_state.processing:
    if st.session_state.language == "it":
        st.write("Elaborazione in corso...")
    else:
        st.write("Processing...")

# Suggested questions (only show if chat is empty)
if not st.session_state.chat_history:
    st.subheader(suggested_title)
    col1, col2 = st.columns(2)
    
    for i, question in enumerate(suggested_questions):
        col = col1 if i < 2 else col2
        if col.button(question, key=f"suggested_{i}"):
            user_msg_key = f"user_msg_{len(st.session_state.chat_history)}"
            st.session_state.chat_history.append({
                "message": question,
                "is_user": True,
                "key": user_msg_key
            })
            st.session_state.processing = True
            st.rerun()

# Input field for user queries
user_input = st.chat_input(input_placeholder)

# Process user input
if user_input and not st.session_state.processing:
    user_msg_key = f"user_msg_{len(st.session_state.chat_history)}"
    st.session_state.chat_history.append({
        "message": user_input,
        "is_user": True,
        "key": user_msg_key
    })
    st.session_state.processing = True
    st.rerun()

# Process the response
if st.session_state.processing:
    last_user_message = next((msg["message"] for msg in reversed(st.session_state.chat_history) if msg["is_user"]), None)
    
    if last_user_message:
        response = process_query(last_user_message, st.session_state.language)
        bot_msg_key = f"bot_msg_{len(st.session_state.chat_history)}"
        st.session_state.chat_history.append({
            "message": response,
            "is_user": False,
            "key": bot_msg_key
        })
    
    st.session_state.processing = False
    st.rerun() 