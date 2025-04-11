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

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Fiscozen Tax Chatbot",
    page_icon="💼",
    layout="centered"
)

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

# Initialize session state
if 'responses' not in st.session_state:
    st.session_state['responses'] = []
if 'requests' not in st.session_state:
    st.session_state['requests'] = []
if 'conversation_id' not in st.session_state:
    st.session_state['conversation_id'] = str(uuid.uuid4())

# Initialize relevance checker
try:
    if 'relevance_checker' not in st.session_state:
        st.session_state['relevance_checker'] = FastTextRelevanceChecker()
except Exception as e:
    st.error(f"Error initializing relevance checker: {e}")
    st.stop()

# Page header
st.title("Fiscozen Tax Chatbot 💼")
st.markdown("""
Welcome to Fiscozen's Tax Assistant! I can help you with:
- Questions about IVA (Italian VAT)
- Tax-related inquiries
- Fiscozen services
""")

# Chat interface
response_container = st.container()
input_container = st.container()

with input_container:
    user_input = st.chat_input("Ask your question here...")
    if user_input:
        with st.spinner("Processing your question..."):
            try:
                # Check relevance
                is_relevant, details = st.session_state['relevance_checker'].is_relevant(user_input)
                
                if not is_relevant:
                    response = "Mi dispiace, ma posso rispondere solo a domande relative a tasse, IVA e questioni fiscali. Posso aiutarti con domande su questi argomenti?"
                else:
                    # Process relevant query
                    conversation_string = get_conversation_string()
                    refined_query = query_refiner(conversation_string, user_input)
                    response = find_match(refined_query)
                    
                    # Update conversation memory
                    st.session_state['memory'].save_context(
                        {"input": user_input},
                        {"answer": response}
                    )
            except Exception as e:
                st.error(f"Error processing query: {e}")
                response = "Mi dispiace, si è verificato un errore. Per favore, riprova."
            
            # Store the conversation
            st.session_state['requests'].append(user_input)
            st.session_state['responses'].append(response)

# Display conversation history
with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            if i < len(st.session_state['requests']):
                message(st.session_state['requests'][i], is_user=True, key=f"user_msg_{i}")
            message(st.session_state['responses'][i], key=f"bot_msg_{i}")

# Add a button to clear conversation history
if st.button("Clear Conversation"):
    st.session_state['responses'] = []
    st.session_state['requests'] = []
    st.session_state['memory'].clear()
    st.session_state['conversation_id'] = str(uuid.uuid4()) 