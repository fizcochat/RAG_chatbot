import streamlit as st
from streamlit_chat import message
import os
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

# Load environment variables
load_dotenv()

st.subheader("Fisco-Chat")

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    st.error("Please set up your API keys in the .env file")
    st.stop()

# Initialize services with environment variables
vectorstore, client = initialize_services(OPENAI_API_KEY, PINECONE_API_KEY)

# Dropdown for selecting the chat LLM model
model_name = st.selectbox(
    "Choose the chat model for the LLM:",
    ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"],
    index=0
)

# Use the selected model from the dropdown
llm = ChatOpenAI(model_name=model_name, openai_api_key=OPENAI_API_KEY)

if 'responses' not in st.session_state:
     st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

system_msg_template = SystemMessagePromptTemplate.from_template(template="""
You are **Fisco-Chat**, the AI assistant for **Fiscozen**, a digital platform that simplifies VAT management for freelancers and sole proprietors in Italy. Your primary goal is to provide users with accurate and efficient tax-related assistance by retrieving information from the provided documentation before generating a response. Additionally, you serve as a bridge between:
- **AI-based assistance** (answering questions directly when the provided documents contain the necessary information),
- **CS Consultants** (for general customer support beyond your knowledge), and
- **Tax Advisors** (for complex tax matters requiring personalized expertise).

**Response Workflow:**
1. **Check Documentation First**
   - Before answering, always search the provided documentation for relevant information.
   - If the answer is found, summarize it clearly and concisely.
   - If the answer is partially found, provide the available information and suggest further steps.

2. **Determine the Best Course of Action**
   - If the user's question is fully covered in the documentation, respond confidently with the answer.
   - If the question is outside the scope of the documentation or requires case-specific advice:
     - **For general support (e.g., account issues, service-related questions):** Suggest redirecting to a **CS Consultant**.
     - **For tax-specific advice that requires a professional opinion:** Suggest scheduling an appointment with a **Tax Advisor** and provide instructions to do so.

**Tone & Interaction Guidelines:**
- Maintain a **professional, clear, and friendly** tone.
- Be **precise and concise** in your responses—users appreciate efficiency.
- Use simple language where possible to make complex tax topics easy to understand.
- If redirecting to a consultant or advisor, explain **why** the transfer is necessary.

**Limitations & Boundaries:**
- Do not make assumptions beyond the provided documentation.
- Do not offer legal, financial, or tax advice beyond the scope of Fiscozen’s services.
- If uncertain, guide the user toward professional assistance rather than providing speculative answers.
""")

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

response_container = st.container()
textcontainer = st.container()

with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            refined_query = query_refiner(client, conversation_string, query)
            st.subheader("Refined Query:")
            st.write(refined_query)
            context = find_match(vectorstore, refined_query)
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            st.write(st.session_state['responses'][i])  # Replace message with st.write
            if i < len(st.session_state['requests']):
                st.write(st.session_state["requests"][i])