import streamlit as st
from streamlit_chat import message
import os
import re
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
# Import the conversation-aware relevance checking
from simple_integration import check_message_relevance, reset_conversation
from relevance import RelevanceChecker

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

# Remove the dropdown and set a fixed model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

if 'responses' not in st.session_state:
     st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

# Add user_id for conversation tracking
if 'user_id' not in st.session_state:
    import uuid
    st.session_state['user_id'] = str(uuid.uuid4())

# Add off-topic counter for explicit tracking in session state
if 'off_topic_count' not in st.session_state:
    st.session_state['off_topic_count'] = 0

# Add debug mode to session state (can be toggled in the UI if needed)
if 'debug_mode' not in st.session_state:
    st.session_state['debug_mode'] = True  # Enable debug by default for testing

# Add sidebar toggle for debug mode
with st.sidebar:
    st.session_state['debug_mode'] = st.checkbox("Enable Debug Mode", value=st.session_state['debug_mode'])
    
    if st.session_state['debug_mode']:
        st.write(f"Current off-topic count: {st.session_state['off_topic_count']}")
        st.write(f"User ID: {st.session_state['user_id']}")
        if st.button("Reset Conversation Tracking"):
            reset_conversation(st.session_state['user_id'])
            st.session_state['off_topic_count'] = 0
            st.write("Conversation tracking reset!")

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

# Debug container for showing relevance results
debug_container = st.container()

with textcontainer:
    query = st.chat_input("Type here...")
    if query:
        with st.spinner("Typing..."):
            # Store the original query for display
            original_query = query
            
            # Check relevance with conversation context - more aggressive thresholds
            relevance_result = check_message_relevance(
                message=query,
                user_id=st.session_state['user_id'],
                model_path="models/enhanced_bert",
                tax_threshold=0.5,  # Lower threshold to be more sensitive
                redirect_threshold=2  # Redirect after fewer off-topic messages
            )
            
            # Also update our explicit counter for redundancy
            if not relevance_result['is_relevant']:
                st.session_state['off_topic_count'] += 1
            else:
                st.session_state['off_topic_count'] = max(0, st.session_state['off_topic_count'] - 1)
            
            # Debug information if needed
            if st.session_state['debug_mode']:
                with debug_container:
                    st.write("## Debug Information")
                    st.write(f"**Original query:** {original_query}")
                    st.write(f"**Is relevant:** {relevance_result['is_relevant']}")
                    st.write(f"**Topic:** {relevance_result['topic']}")
                    st.write(f"**Tax probability:** {relevance_result['tax_related_probability']:.4f}")
                    st.write(f"**Confidence:** {relevance_result['confidence']:.4f}")
                    st.write(f"**Consecutive off-topic:** {relevance_result['consecutive_off_topic']}")
                    st.write(f"**Conversation drifting:** {relevance_result['is_drifting']}")
                    st.write(f"**Needs redirect:** {relevance_result['needs_redirect']}")
                    st.write(f"**Session off-topic count:** {st.session_state['off_topic_count']}")
                    
                    # Show probabilities
                    st.write("### Class Probabilities")
                    st.write(f"- IVA: {relevance_result['probabilities']['IVA']:.4f}")
                    st.write(f"- Fiscozen: {relevance_result['probabilities']['Fiscozen']:.4f}")
                    st.write(f"- Other: {relevance_result['probabilities']['Other']:.4f}")
            
            # Force redirect if either tracking method suggests it
            needs_redirect = relevance_result['needs_redirect'] or st.session_state['off_topic_count'] >= 2
            
            if needs_redirect:
                # Conversation has drifted too far off-topic
                response = (
                    "<span class='warning-text'>OFF-TOPIC CONVERSATION DETECTED:</span> "
                    "I notice our conversation has moved away from tax-related topics. "
                    "I'm specialized in Italian tax and Fiscozen-related matters only. "
                    "Let me redirect you to a Customer Success Consultant who can help with general inquiries."
                )
                # Reset conversation tracking after redirection
                reset_conversation(st.session_state['user_id'])
                st.session_state['off_topic_count'] = 0
                
            elif not relevance_result['is_relevant']:
                # Current message is off-topic but not enough to redirect yet
                response = (
                    "<span class='warning-text'>OFF-TOPIC DETECTED:</span> "
                    "I'm specialized in Italian tax matters and Fiscozen services. "
                    "Could you please ask something related to taxes, IVA, or Fiscozen?"
                )
                
            else:
                # Message is relevant - process normally
                conversation_string = get_conversation_string()
                refined_query = query_refiner(client, conversation_string, query)
                print("\nRefined Query:", refined_query)
                context = find_match(vectorstore, refined_query)
                response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
                
                # Add topic-specific context if we have high confidence
                if relevance_result['confidence'] > 0.6:
                    topic = relevance_result['topic']
                    if topic == "IVA" and "IVA" not in response.upper():
                        response = f"Regarding IVA (Italian VAT): {response}"
                    elif topic == "Fiscozen" and "Fiscozen" not in response:
                        response = f"About Fiscozen services: {response}"
            
            # Add conversation drift warning if drifting but not yet redirecting
            if relevance_result['is_drifting'] and not needs_redirect and relevance_result['is_relevant']:
                drift_note = (
                    "<br><br><span class='warning-text'>Note:</span> Our conversation seems to be moving away from tax topics. "
                    "I'm specialized in Italian tax matters and Fiscozen services."
                )
                response += drift_note
                    
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
    Process user input and generate response with conversation-aware topic tracking
    
    Args:
        user_input: The user's message
        conversation_id: Identifier for the conversation
        
    Returns:
        Response text
    """
    if not user_input:
        return "Please enter a valid question."
    
    # Check relevance with conversation context - more aggressive thresholds
    relevance_result = check_message_relevance(
        message=user_input,
        user_id=conversation_id,
        model_path="models/enhanced_bert",
        tax_threshold=0.5,  # Lower threshold to be more sensitive
        redirect_threshold=2  # Redirect after fewer off-topic messages
    )
    
    if relevance_result['needs_redirect']:
        # Conversation has drifted too far off-topic
        reset_conversation(conversation_id)  # Reset tracking
        return (
            "OFF-TOPIC CONVERSATION DETECTED: "
            "I notice our conversation has moved away from tax-related topics. "
            "I'm specialized in Italian tax and Fiscozen-related matters only. "
            "Let me redirect you to a Customer Success Consultant who can help with general inquiries."
        )
        
    elif not relevance_result['is_relevant']:
        # Current message is off-topic but not enough to redirect yet
        return (
            "OFF-TOPIC DETECTED: "
            "I'm specialized in Italian tax matters and Fiscozen services. "
            "Could you please ask something related to taxes, IVA, or Fiscozen?"
        )
    
    # Message is relevant - process normally
    conversation_string = get_conversation_string()
    refined_query = query_refiner(client, conversation_string, user_input)
    context = find_match(vectorstore, refined_query)
    response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{user_input}")
    
    # Add topic-specific context if we have high confidence
    if relevance_result['confidence'] > 0.6:
        topic = relevance_result['topic']
        if topic == "IVA" and "IVA" not in response.upper():
            response = f"Regarding IVA (Italian VAT): {response}"
        elif topic == "Fiscozen" and "Fiscozen" not in response:
            response = f"About Fiscozen services: {response}"
    
    # Add conversation drift warning if drifting but not yet redirecting
    if relevance_result['is_drifting'] and not relevance_result['needs_redirect']:
        drift_note = (
            "\n\nNote: Our conversation seems to be moving away from tax topics. "
            "I'm specialized in Italian tax matters and Fiscozen services."
        )
        response += drift_note
        
    return response
