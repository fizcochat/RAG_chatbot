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

# Logo only, no title, larger and centered
st.image("images/fiscozen_logo.jpeg", width=200)

# Welcome Section
st.markdown("""
### Benvenuto nel Tax Assistant di Fiscozen!
Sono qui per aiutarti con le tue domande fiscali. Posso supportarti su:
""")

# List of features with checkmarks
st.markdown("""
- ✓ IVA e regime fiscale italiano
- ✓ Detrazioni e deduzioni fiscali
- ✓ Dichiarazione dei redditi
- ✓ Servizi offerti da Fiscozen
""")

# Restart button at the top
restart_clicked = st.button("↻ Ricomincia")
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
    st.write("Processing...")

# Suggested questions (only show if chat is empty)
if not st.session_state.chat_history:
    st.subheader("Domande Frequenti:")
    col1, col2 = st.columns(2)
    
    suggested_questions = [
        "Come funziona l'IVA per i liberi professionisti?",
        "Quali detrazioni fiscali posso avere per i figli?",
        "Come gestire le fatture elettroniche?",
        "Cosa offre Fiscozen?"
    ]
    
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
user_input = st.chat_input("Chiedimi qualcosa sul fisco italiano...")

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
        response = process_query(last_user_message)
        bot_msg_key = f"bot_msg_{len(st.session_state.chat_history)}"
        st.session_state.chat_history.append({
            "message": response,
            "is_user": False,
            "key": bot_msg_key
        })
    
    st.session_state.processing = False
    st.rerun() 