"""
Fiscozen Tax Chatbot - Main Application
"""

import os
import sys
import streamlit as st
from bert import RelevanceChecker
from dotenv import load_dotenv
import re
import uuid
import time
import random
from streamlit_chat import message
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

# === Initialization Code (previously in run_prod.py) ===
def initialize_environment():
    # Load environment variables from .env if it exists
    if os.path.exists(".env"):
        load_dotenv()
        
    # Check for required API keys
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("PINECONE_API_KEY"):
        st.error("⚠️ API keys not found! Please set OPENAI_API_KEY and PINECONE_API_KEY in your environment or .env file.")
        st.stop()
    
    # Create model directories if they don't exist
    os.makedirs("bert/models/enhanced_bert", exist_ok=True)
    
    # Check if BERT model is initialized
    model_path = "bert/models/enhanced_bert"
    if not os.path.exists(os.path.join(model_path, "config.json")):
        st.warning("⚠️ BERT model not initialized. Attempting to initialize now...")
        try:
            # Try dedicated initialization script
            if os.path.exists("bert/initialize_model.py"):
                import subprocess
                subprocess.check_call([sys.executable, "bert/initialize_model.py"])
            else:
                # Fallback to manual initialization
                from transformers import BertTokenizer, BertForSequenceClassification
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
                model.save_pretrained(model_path)
                tokenizer.save_pretrained(model_path)
        except Exception as e:
            st.error(f"❌ Error initializing BERT model: {e}")
            st.info("Please run 'python bert/initialize_model.py' before starting the chatbot.")
            st.stop()

# Run initialization before the Streamlit app
initialize_environment()

# === Original main.py code continues below ===

# Text preprocessing function
def preprocess_text(text):
    """
    Clean and normalize text for better relevance detection
    
    Args:
        text: The input text to clean
        
    Returns:
        Cleaned and normalized text
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
relevance_checker = RelevanceChecker(model_path="bert/models/enhanced_bert")

# Remove the dropdown and set a fixed model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

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
    import uuid
    st.session_state['user_id'] = str(uuid.uuid4())

# Add debug mode to session state (disable for production)
if 'debug_mode' not in st.session_state:
    st.session_state['debug_mode'] = False

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

response_container = st.container()
textcontainer = st.container()

# Debug container (only shown in debug mode)
debug_container = st.container()

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
                    st.write(f"- IVA: {result['probabilities']['IVA']:.4f}")
                    st.write(f"- Fiscozen: {result['probabilities']['Fiscozen']:.4f}")
                    st.write(f"- Other: {result['probabilities']['Other']:.4f}")
            
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
