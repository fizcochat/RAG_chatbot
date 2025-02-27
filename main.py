import streamlit as st
from streamlit_chat import message
import os
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

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

with textcontainer:
    query = st.chat_input("Type here...")
    if query:
        with st.spinner("Typing..."):
            conversation_string = get_conversation_string()
            refined_query = query_refiner(client, conversation_string, query)
            print("\nRefined Query:", refined_query)
            context = find_match(vectorstore, refined_query)
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)
        st.rerun()

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], 
                   avatar_style="no-avatar",
                   key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], 
                       is_user=True,
                       avatar_style="no-avatar",
                       key=str(i) + '_user')

# Initialize and train the model
def initialize_classifier():
    # Training data (same as in test file)
    queries = [
        # Fiscozen related
        "How do I contact Fiscozen support?",
        "What services does Fiscozen offer?",
        "Can Fiscozen help with my accounting?",
        "Fiscozen pricing plans",
        "How to register with Fiscozen",
        
        # IVA related
        "What is an IVA?",
        "How long does an IVA last?",
        "IVA payment terms",
        "Can I get an IVA if I'm self-employed?",
        "IVA debt minimum",
        
        # Tax related
        "How to pay taxes online?",
        "When is the tax deadline?",
        "Tax deductions for businesses",
        "VAT registration process",
        "Corporate tax rates",
        
        # Unrelated queries
        "What's the weather today?",
        "Best restaurants nearby",
        "How to make pasta",
        "Latest football scores",
        "Movie showtimes"
    ]
    
    labels = [0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3]
    
    # Create and train the model
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(queries)
    model = LogisticRegression(multi_class='ovr', max_iter=1000)
    model.fit(X, labels)
    
    return vectorizer, model

# Global variables for model and vectorizer
vectorizer, model = initialize_classifier()

def classify_query(query):
    """
    Classifies a query into one of four categories:
    0: Fiscozen related
    1: IVA related
    2: Tax related
    3: Unrelated
    """
    # Transform the query using the same vectorizer
    query_vector = vectorizer.transform([query])
    
    # Predict the category
    prediction = model.predict(query_vector)
    
    return prediction[0]

def get_response(query):
    # First classify the query
    category = classify_query(query)
    
    if not query.strip():
        return "Please ask a question."
    
    # Generate response based on category
    if category == 0:
        return "This is a Fiscozen-related query. Let me help you with that..."
    elif category == 1:
        return "This is related to Individual Voluntary Arrangement (IVA)..."
    elif category == 2:
        return "This is a tax-related query. Here's how you can proceed..."
    else:
        return "I cannot help with this query as it's not related to Fiscozen, IVA, or tax matters."
