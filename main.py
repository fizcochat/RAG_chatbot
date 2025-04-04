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
    
    # Check for document folders - ensure we check all possible data sources
    data_dirs = {
        "data_documents": os.path.exists("data_documents") and any(os.listdir("data_documents")),
        "dtaa-documents": os.path.exists("dtaa-documents") and any(os.listdir("dtaa-documents")),
        "argilla_data_49": os.path.exists("argilla_data_49") and any(os.listdir("argilla_data_49")),
        "argilla-data": os.path.exists("argilla-data") and any(os.listdir("argilla-data"))
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
        if os.path.exists("fasttext/train_classifier.py"):
            print("🔄 Starting FastText training process...")
            try:
                # Run the training script
                subprocess.check_call([sys.executable, "fasttext/train_classifier.py"])
                print("✅ FastText classifier training completed successfully!")
            except subprocess.CalledProcessError as e:
                print(f"❌ FastText classifier training failed: {e}")
                choice = input("Do you want to continue without training? (y/n): ")
                if choice.lower() != 'y':
                    sys.exit(1)
        else:
            print("❌ Training script not found: fasttext/train_classifier.py")
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
- Text files (.txt)

For best results:
1. Organize documents by topic
2. Include a variety of text samples for each category (IVA, Fiscozen, Other)
3. Make sure documents are in Italian or English
4. Aim for at least 10-20 documents per category

After adding documents, run `python main.py` to retrain the model.
""")
        
        print("✅ Created document directories with instructions")
        print("Using default FastText model without custom training")
    
    # Ensure FastText model exists
    model_path = "fasttext/models/tax_classifier.bin"
    if not os.path.exists(model_path):
        print("🔄 Initializing FastText model...")
        try:
            # Check for initialization script
            if os.path.exists("fasttext/initialize_model.py"):
                subprocess.check_call([sys.executable, "fasttext/initialize_model.py"])
                print("✅ FastText model initialized successfully!")
            else:
                print("❌ Initialization script not found: fasttext/initialize_model.py")
                print("Please run 'python fasttext/initialize_model.py' before starting the chatbot.")
                sys.exit(1)
        except Exception as e:
            print(f"❌ Failed to initialize FastText model: {e}")
            sys.exit(1)
    else:
        print("✅ FastText model already initialized")

def launch_chatbot():
    """Launch the Streamlit chatbot"""
    print("\n=== Launching Fiscozen Tax Chatbot ===")
    
    if is_streamlit_running:
        print("🚀 Streamlit is already running!")
        return
    
    print("🚀 Starting Streamlit server...")
    try:
        # Set headless mode for server
        os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
        # Launch Streamlit
        subprocess.call([sys.executable, "-m", "streamlit", "run", __file__])
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

# === Streamlit App Code Starts Here ===
try:
    import streamlit as st
    from streamlit_chat import message
    from fasttext.relevance import FastTextRelevanceChecker
    from dotenv import load_dotenv
    from utils import initialize_services, find_match, query_refiner, get_conversation_string
    from langchain_openai import ChatOpenAI
    from langchain.chains import ConversationChain
    from langchain.chains.conversation.memory import ConversationBufferWindowMemory
    from langchain.prompts import (
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
        ChatPromptTemplate,
        MessagesPlaceholder
    )
except ImportError as e:
    if not is_streamlit_running:
        print(f"Error importing required modules: {e}")
        print("Please run with 'python main.py' to automatically install dependencies")
        sys.exit(1)
    raise

# === Streamlit Initialization ===
def initialize_environment_for_streamlit():
    """Initialize the environment for the Streamlit chatbot"""
    # Skip complex initialization in test environments
    if is_test_environment:
        return
        
    # Load environment variables from .env if it exists
    if os.path.exists(".env"):
        load_dotenv()
        
    # Check for required API keys
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("PINECONE_API_KEY"):
        st.error("⚠️ API keys not found! Please set OPENAI_API_KEY and PINECONE_API_KEY in your environment or .env file.")
        st.stop()
    
    # Create model directories if they don't exist
    os.makedirs("fasttext/models", exist_ok=True)
    
    # Check if FastText model is initialized
    model_path = "fasttext/models/tax_classifier.bin"
    if not os.path.exists(model_path):
        st.warning("⚠️ FastText model not initialized. Attempting to initialize now...")
        try:
            # Try dedicated initialization script
            if os.path.exists("fasttext/initialize_model.py"):
                subprocess.check_call([sys.executable, "fasttext/initialize_model.py"])
                st.success("FastText model initialized successfully!")
            else:
                st.error("❌ Initialization script not found: fasttext/initialize_model.py")
                st.info("Please run 'python main.py' before starting the chatbot.")
                st.stop()
        except Exception as e:
            st.error(f"Error initializing FastText model: {e}")
            st.info("Please run 'python fasttext/initialize_model.py' before starting the chatbot.")
            st.stop()

# Run initialization before the Streamlit app (unless in test environment)
if is_streamlit_running:
    initialize_environment_for_streamlit()

# === Text Processing Functions ===
def preprocess_text(text):
    """
    Clean and normalize text for better relevance detection
    """
    if not text:
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Replace common abbreviations and variants
    replacements = {
        "iva's": "iva",
        "i.v.a": "iva",
        "i.v.a.": "iva",
        "fiscozen's": "fiscozen",
        "fisco zen": "fiscozen",
        "fisco-zen": "fiscozen",
        "fisco zen's": "fiscozen",
        "v.a.t": "vat",
        "v.a.t.": "vat"
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text

# === Streamlit UI Code ===
if is_streamlit_running:
    # Add custom CSS
    st.markdown("""
    <style>
        .stTextInput > label {
            color: black;
        }
        .stSpinner > div {
            color: black;
        }
        .stSubheader {
            color: black;
        }
        div.stMarkdown > div > p {
            color: black;
        }
        .css-1n76uvr {
            color: black;
        }
        .warning-text {
            color: #ff4b4b;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

    st.subheader("Fiscozen")
    
    # Get API keys from environment variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        st.error("Please set up your API keys in the .env file")
        st.stop()

    # Initialize services with environment variables
    vectorstore, client = initialize_services(OPENAI_API_KEY, PINECONE_API_KEY)

    # Initialize the relevance checker with model path
    relevance_checker = FastTextRelevanceChecker(model_path="fasttext/models/tax_classifier.bin")

    # Remove the dropdown and set a fixed model
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Initialize session state
    if 'responses' not in st.session_state:
         st.session_state['responses'] = ["How can I assist you with Italian tax matters and Fiscozen services?"]

    if 'requests' not in st.session_state:
        st.session_state['requests'] = []

    if 'buffer_memory' not in st.session_state:
        st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

    # Add off-topic tracking to session state
    if 'off_topic_count' not in st.session_state:
        st.session_state['off_topic_count'] = 0

    # Add user ID for tracking conversations
    if 'user_id' not in st.session_state:
        st.session_state['user_id'] = str(uuid.uuid4())

    # Add debug mode to session state (disable for production)
    if 'debug_mode' not in st.session_state:
        st.session_state['debug_mode'] = False

    # Set up conversation system
    system_msg_template = SystemMessagePromptTemplate.from_template(template="""
                                                                    
    **Never mention that your responses are based on documents, data, or retrieved information. Present all answers as direct and authoritative.** 
    You are **Fisco-Chat**, the AI assistant for **Fiscozen**, a digital platform that simplifies VAT management for freelancers and sole proprietors in Italy. Your primary goal is to provide users with accurate and efficient tax-related assistance by retrieving information from the provided documentation before generating a response. Additionally, you serve as a bridge between:
    - **AI-based assistance** (answering questions directly when the provided documents contain the necessary information),
    - **CS Consultants** (for general customer support beyond your knowledge), and
    - **Tax Advisors** (for complex tax matters requiring personalized expertise).
                                                                    
    **Never mention that your responses are based on documents, data, or retrieved information. Present all answers as direct and authoritative.** 
                                           
    **Response Workflow:**
    1. **Check Documentation First**
       - Before answering, always search the provided documentation for relevant information.
       - If the answer is found, summarize it clearly and concisely.
       - If the answer is partially found, provide the available information and suggest further steps.

    2. **Determine the Best Course of Action**
       - If the user's question is fully covered in the documentation, respond confidently with the answer.
       - If the question is outside the scope of the documentation or requires case-specific advice:
         - **For general support (e.g., account issues, service-related questions):** Suggest redirecting to a **Fiscozen Customer Success Consultant**.
         - **For tax-specific advice that requires a professional opinion:** Suggest scheduling an appointment with a **Fiscozen Tax Advisor** and provide instructions to do so.
       - **If the user explicitly requests to speak with a human (CS Consultant or Tax Advisor), immediately suggest the appropriate redirection** without attempting to resolve the issue further.

    **Tone & Interaction Guidelines:**
    - Maintain a **professional, clear, and friendly** tone. 
    - Be **precise and concise** in your responses—users appreciate efficiency.
    - Use simple language where possible to make complex tax topics easy to understand.
    - If redirecting to a consultant or advisor, explain **why** the transfer is necessary
    - **Never mention that your responses are based on documents, data, or retrieved information. Present all answers as direct and authoritative.** 

    **Limitations & Boundaries:**
    - Do not make assumptions beyond the provided documentation.
    - Do not offer legal, financial, or tax advice beyond the scope of Fiscozen's services.
    - If uncertain, guide the user toward professional assistance rather than providing speculative answers.
    """)

    human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

    prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

    conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

    # UI Container setup
    response_container = st.container()
    textcontainer = st.container()
    debug_container = st.container()  # Debug container

    # Input handling
    with textcontainer:
        query = st.chat_input("Type here...")
        if query:
            with st.spinner("Typing..."):
                # Store the original query for display
                original_query = query
                
                # Preprocess the query for relevance checking
                preprocessed_query = preprocess_text(query)
                
                # Check if the query is relevant to tax matters
                result = relevance_checker.check_relevance(preprocessed_query, tax_threshold=0.5)
                
                # Debug information if needed
                if st.session_state['debug_mode']:
                    with debug_container:
                        st.write("## Debug Information")
                        st.write(f"**Original query:** {original_query}")
                        st.write(f"**Is relevant:** {result['is_relevant']}")
                        st.write(f"**Topic:** {result['topic']}")
                        st.write(f"**Tax probability:** {result['tax_related_probability']:.4f}")
                        st.write(f"**Session off-topic count:** {st.session_state['off_topic_count']}")
                        
                        # Show probabilities
                        st.write("### Class Probabilities")
                        st.write(f"- IVA: {result['probabilities'].get('IVA', 0):.4f}")
                        st.write(f"- Fiscozen: {result['probabilities'].get('Fiscozen', 0):.4f}")
                        st.write(f"- Other: {result['probabilities'].get('Other', 0):.4f}")
                
                if not result['is_relevant']:
                    # Increment off-topic count
                    st.session_state['off_topic_count'] += 1
                    
                    # Check if we need to redirect
                    if st.session_state['off_topic_count'] >= 2:
                        response = (
                            "<span class='warning-text'>OFF-TOPIC CONVERSATION DETECTED:</span> "
                            "I notice our conversation has moved away from tax-related topics. "
                            "I'm specialized in Italian tax and Fiscozen-related matters only. "
                            "Let me redirect you to a Customer Success Consultant who can help with general inquiries."
                        )
                        st.session_state['off_topic_count'] = 0  # Reset after redirecting
                    else:
                        # Just warn the user
                        response = (
                            "<span class='warning-text'>OFF-TOPIC DETECTED:</span> "
                            "I'm specialized in Italian tax matters and Fiscozen services. "
                            "Could you please ask something related to taxes, IVA, or Fiscozen?"
                        )
                else:
                    # Reset off-topic count for relevant queries
                    st.session_state['off_topic_count'] = 0
                    
                    # Process relevant query normally
                    conversation_string = get_conversation_string()
                    refined_query = query_refiner(client, conversation_string, query)
                    print("\nRefined Query:", refined_query)
                    context = find_match(vectorstore, refined_query)
                    response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
                    
                    # Add some topic-specific context if we have high confidence
                    if result['confidence'] > 0.6:
                        topic = result['topic']
                        if topic == "IVA" and "IVA" not in response.upper():
                            response = f"Regarding IVA (Italian VAT): {response}"
                        elif topic == "Fiscozen" and "Fiscozen" not in response:
                            response = f"About Fiscozen services: {response}"
                        
            # Use the original query for display
            st.session_state.requests.append(original_query)
            st.session_state.responses.append(response)
            st.rerun()

    # Display conversation history
    with response_container:
        if st.session_state['responses']:
            for i in range(len(st.session_state['responses'])):
                message(st.session_state['responses'][i], 
                       avatar_style="no-avatar",
                       key=str(i),
                       allow_html=True)  # Allow HTML for warning formatting
                if i < len(st.session_state['requests']):
                    message(st.session_state["requests"][i], 
                           is_user=True,
                           avatar_style="no-avatar",
                           key=str(i) + '_user')

# === API Function ===
def get_response(user_input: str, conversation_id: str = "api_user") -> str:
    """
    Process user input and generate response with basic off-topic detection
    
    Args:
        user_input: The user's message
        conversation_id: Identifier for the conversation
        
    Returns:
        Response text
    """
    if not user_input:
        return "Please enter a valid question."
    
    # Static counter for off-topic messages (for API usage)
    if not hasattr(get_response, 'off_topic_count'):
        setattr(get_response, 'off_topic_count', {})
    
    if conversation_id not in get_response.off_topic_count:
        get_response.off_topic_count[conversation_id] = 0
    
    # Check if the query is relevant to tax matters
    preprocessed_input = preprocess_text(user_input)
    result = relevance_checker.check_relevance(preprocessed_input, tax_threshold=0.5)
    
    if not result['is_relevant']:
        # Increment off-topic count
        get_response.off_topic_count[conversation_id] += 1
        
        # Check if we need to redirect
        if get_response.off_topic_count[conversation_id] >= 2:
            # Reset the counter
            get_response.off_topic_count[conversation_id] = 0
            
            return (
                "OFF-TOPIC CONVERSATION DETECTED: "
                "I notice our conversation has moved away from tax-related topics. "
                "I'm specialized in Italian tax and Fiscozen-related matters only. "
                "Let me redirect you to a Customer Success Consultant who can help with general inquiries."
            )
        else:
            # Just warn the user
            return (
                "OFF-TOPIC DETECTED: "
                "I'm specialized in Italian tax matters and Fiscozen services. "
                "Could you please ask something related to taxes, IVA, or Fiscozen?"
            )
    
    # Reset off-topic count for relevant queries
    get_response.off_topic_count[conversation_id] = 0
    
    # Process relevant query normally
    conversation_string = ""  # No conversation context in API mode
    refined_query = query_refiner(client, conversation_string, user_input)
    context = find_match(vectorstore, refined_query)
    response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{user_input}")
    
    # Add topic-specific context if we have high confidence
    if result['confidence'] > 0.6:
        topic = result['topic']
        if topic == "IVA" and "IVA" not in response.upper():
            response = f"Regarding IVA (Italian VAT): {response}"
        elif topic == "Fiscozen" and "Fiscozen" not in response:
            response = f"About Fiscozen services: {response}"
    
    return response
