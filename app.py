"""
Fiscozen Tax Chatbot - Streamlit App
"""

import os
import streamlit as st
from streamlit_chat import message
from fast_text.relevance import FastTextRelevanceChecker
from utils import initialize_services, find_match, query_refiner, get_conversation_string
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv
import uuid
import time
import base64

# Load environment variables
load_dotenv()

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

# Initialize session state for chat
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'processing' not in st.session_state:
    st.session_state.processing = False

# Get base64 encoded images for chat icons
try:
    fiscozen_logo_base64 = get_image_base64("images/fiscozen_logo.jpeg")
    fiscozen_small_base64 = get_image_base64("images/fiscozen_small.png")
    # Default user icon (blue professional avatar)
    user_icon = """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 512" width="24" height="24" fill="#4a4e69">
        <path d="M224 256A128 128 0 1 0 224 0a128 128 0 1 0 0 256zm-45.7 48C79.8 304 0 383.8 0 482.3C0 498.7 13.3 512 29.7 512H418.3c16.4 0 29.7-13.3 29.7-29.7C448 383.8 368.2 304 269.7 304H178.3z"/>
    </svg>
    """
except Exception as e:
    print(f"Error loading images: {e}")
    fiscozen_logo_base64 = ""
    fiscozen_small_base64 = ""
    user_icon = ""

# Custom CSS with animations and enhanced styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary-color: #3c4c9c;
        --secondary-color: #6c76af;
        --accent-color: #8a92c7;
        --background-color: #f8f9fd;
        --chat-user-bg: #e8eaf7;
        --chat-bot-bg: #f0f2ff;
        --text-color: #333;
        --light-text: #666;
    }
    
    /* Base styles */
    body {
        font-family: 'Poppins', sans-serif;
        background-color: var(--background-color);
        color: var(--text-color);
        line-height: 1.6;
    }
    
    h1, h2, h3 {
        font-weight: 600;
        color: var(--primary-color);
    }
    
    /* Header animation */
    .title-animation {
        display: inline-block;
        position: relative;
        animation: fadeIn 1.5s ease-out;
    }
    
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(-10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    /* Logo animation */
    .logo-container {
        display: flex;
        justify-content: center;
        margin-bottom: 1rem;
        animation: scaleIn 0.8s ease-out;
    }
    
    @keyframes scaleIn {
        0% { transform: scale(0.9); opacity: 0; }
        100% { transform: scale(1); opacity: 1; }
    }
    
    /* Chat messages styling */
    .stChatMessage {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        animation: slideIn 0.3s ease-out;
        transition: all 0.3s ease;
    }
    
    .stChatMessage:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    @keyframes slideIn {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    .stChatMessage[data-testid*="user"] {
        background-color: var(--chat-user-bg);
        border-left: 4px solid var(--primary-color);
    }
    
    .stChatMessage[data-testid*="assistant"] {
        background-color: var(--chat-bot-bg);
        border-left: 4px solid var(--secondary-color);
    }
    
    /* Avatar styling */
    /* For user avatar (initials) */
    [data-testid="avatar-user"] {
        background-color: var(--primary-color) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
    }
    
    /* For bot avatar */
    [data-testid="avatar-assistant"] {
        background-color: var(--secondary-color) !important;
        border: none !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        font-weight: 500;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        background-color: var(--secondary-color);
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }
    
    .stButton>button:active {
        transform: translateY(0);
    }
    
    /* Input field styling */
    .stChatInput>div {
        border-radius: 30px !important;
        border: 2px solid #e0e3f4 !important;
        padding: 0.2rem 1rem !important;
        background-color: white !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05) !important;
    }
    
    .stChatInput>div:focus-within {
        border-color: var(--primary-color) !important;
        box-shadow: 0 2px 10px rgba(60, 76, 156, 0.15) !important;
    }
    
    /* Welcome section styling */
    .welcome-section {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
        border-left: 5px solid var(--primary-color);
        animation: fadeIn 1s ease-out;
    }
    
    .features-list li {
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
    }
    
    .features-list li:before {
        content: "✓";
        color: var(--primary-color);
        font-weight: bold;
        margin-right: 0.5rem;
    }
    
    /* Chat container */
    .chat-container {
        border-radius: 12px;
        background-color: white;
        box-shadow: 0 3px 15px rgba(0,0,0,0.08);
        padding: 1rem;
        margin-top: 1rem;
    }
    
    /* Animated typing indicator */
    .typing-indicator {
        display: inline-flex;
        align-items: center;
        margin: 1rem 0;
    }
    
    .typing-indicator span {
        height: 8px;
        width: 8px;
        background-color: var(--secondary-color);
        border-radius: 50%;
        margin: 0 2px;
        display: inline-block;
        animation: typing 1.4s infinite ease-in-out;
    }
    
    .typing-indicator span:nth-child(1) { animation-delay: 0s; }
    .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
    .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes typing {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-6px); }
    }
    
    /* Fix for chat disappearing */
    .element-container {
        opacity: 1 !important;
    }
    
    </style>
""", unsafe_allow_html=True)

# App Header with Animation
st.markdown(f'<div class="logo-container"><img src="data:image/jpeg;base64,{fiscozen_logo_base64}" width="200"></div>', unsafe_allow_html=True)
st.markdown('<h1 class="title-animation">Fiscozen Tax Chatbot</h1>', unsafe_allow_html=True)

# Welcome Section with better formatting
st.markdown("""
    <div class="welcome-section">
        <h3>Benvenuto nel Tax Assistant di Fiscozen!</h3>
        <p>Sono qui per aiutarti con le tue domande fiscali. Posso supportarti su:</p>
        <ul class="features-list">
            <li>IVA e regime fiscale italiano</li>
            <li>Detrazioni e deduzioni fiscali</li>
            <li>Dichiarazione dei redditi</li>
            <li>Servizi offerti da Fiscozen</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

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
        st.session_state['llm'] = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
    
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
def process_query(query):
    try:
        # Check relevance
        is_relevant, details = st.session_state['relevance_checker'].is_relevant(query)
        
        if not is_relevant:
            return "Mi dispiace, ma posso rispondere solo a domande relative a tasse, IVA e questioni fiscali. Posso aiutarti con domande su questi argomenti?"
        else:
            # Process relevant query
            conversation_string = get_conversation_string()
            refined_query = query_refiner(conversation_string, query)
            response = find_match(refined_query)
            
            # Update conversation memory
            st.session_state['memory'].save_context(
                {"input": query},
                {"answer": response}
            )
            
            return response
    except Exception as e:
        st.error(f"Error processing query: {e}")
        return "Mi dispiace, si è verificato un errore. Per favore, riprova."

# Chat interface
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display chat history
for i, message_obj in enumerate(st.session_state.chat_history):
    if message_obj["is_user"]:
        message(
            message_obj["message"],
            is_user=True,
            key=message_obj["key"],
            avatar_style="initials",
            seed="U"
        )
    else:
        message(
            message_obj["message"],
            is_user=False,
            key=message_obj["key"],
            avatar_style="bottts-neutral",
            seed="F"
        )

# Show typing indicator during processing
if st.session_state.processing:
    st.markdown("""
        <div class="typing-indicator">
            <span></span><span></span><span></span>
        </div>
    """, unsafe_allow_html=True)

# Suggested questions (only show if chat is empty)
if not st.session_state.chat_history:
    st.markdown("<h4>Domande Frequenti:</h4>", unsafe_allow_html=True)
    cols = st.columns(2)
    suggested_questions = [
        "Come funziona l'IVA per i liberi professionisti?",
        "Quali detrazioni fiscali posso avere per i figli?",
        "Come gestire le fatture elettroniche?",
        "Cosa offre Fiscozen?"
    ]
    
    for i, question in enumerate(suggested_questions):
        col_idx = i % 2
        if cols[col_idx].button(question, key=f"suggested_{i}"):
            # Add user message to chat with custom avatar
            user_msg_key = f"user_msg_{len(st.session_state.chat_history)}"
            st.session_state.chat_history.append({
                "message": question,
                "is_user": True,
                "key": user_msg_key
            })
            
            # Set processing flag
            st.session_state.processing = True
            st.rerun()

# Input field for user queries
user_input = st.chat_input("Chiedimi qualcosa sul fisco italiano...")

# Process user input
if user_input and not st.session_state.processing:
    # Add user message to chat with custom avatar
    user_msg_key = f"user_msg_{len(st.session_state.chat_history)}"
    st.session_state.chat_history.append({
        "message": user_input,
        "is_user": True,
        "key": user_msg_key
    })
    
    # Set processing flag
    st.session_state.processing = True
    st.rerun()

# Process the response (after the rerun)
if st.session_state.processing:
    # Get the last user message
    last_user_message = next((msg["message"] for msg in reversed(st.session_state.chat_history) if msg["is_user"]), None)
    
    if last_user_message:
        # Generate response
        response = process_query(last_user_message)
        
        # Add bot message to chat
        bot_msg_key = f"bot_msg_{len(st.session_state.chat_history)}"
        st.session_state.chat_history.append({
            "message": response,
            "is_user": False,
            "key": bot_msg_key
        })
    
    # Reset processing flag
    st.session_state.processing = False
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Add a button to clear conversation history
col1, col2 = st.columns([4, 1])
with col2:
    if st.button("Ricomincia", help="Elimina la conversazione e ricomincia da capo"):
        st.session_state.chat_history = []
        st.session_state['memory'].clear()
        st.rerun() 