"""
Fiscozen Tax Chatbot - Streamlit App
"""

import os
import streamlit as st
from streamlit_chat import message
import sys
import importlib
import logging

# Configure logging for the app
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Function to check and import dependencies
def import_dependencies():
    try:
        # Import FastText relevance checker (with fallback to keyword-only mode)
        from fast_text.relevance import FastTextRelevanceChecker
        st.session_state['relevance_checker'] = FastTextRelevanceChecker()
        logger.info("FastText relevance checker initialized")
    except ImportError:
        logger.warning("FastText module not available. Using keyword-based relevance checking.")
        # Define a simplified fallback relevance checker
        from fast_text.relevance import FastTextRelevanceChecker
        st.session_state['relevance_checker'] = FastTextRelevanceChecker()

    try:
        # Import core utilities
        from utils import initialize_services, find_match, query_refiner, get_conversation_string, translate_to_italian, translate_from_italian
        st.session_state['utils_imported'] = True
    except ImportError as e:
        st.error(f"Failed to import core utilities: {e}")
        st.stop()
    
    try:
        # Import LangChain components
        from langchain_openai import ChatOpenAI
        from langchain.memory import ConversationBufferMemory
        from langchain.schema import HumanMessage, AIMessage
        st.session_state['langchain_imported'] = True
    except ImportError:
        st.error("Failed to import LangChain components. Please install the required packages.")
        st.write("Try running: `pip install langchain langchain-openai`")
        st.stop()
    
    return True

# Only import the required modules when inside the Streamlit app
if 'utils_imported' not in st.session_state:
    import_dependencies()

# Now import core utilities - these imports will work because of the check above
from utils import initialize_services, find_match, query_refiner, get_conversation_string, translate_to_italian, translate_from_italian
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
import uuid
import time
import base64
from monitor.db_logger import init_db, log_event
from fast_text.trainer import train_fasttext_if_needed 

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD")

# Try to load from `.env` for local development
if not OPENAI_API_KEY or not PINECONE_API_KEY:
    if os.path.exists(".env"):
        from dotenv import load_dotenv
        load_dotenv()
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD")

# Try loading from Streamlit secrets if still missing
if not OPENAI_API_KEY or not PINECONE_API_KEY:
    if "OPENAI_API_KEY" in st.secrets and "PINECONE_API_KEY" in st.secrets:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
        PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
        DASHBOARD_PASSWORD = st.secrets.get("DASHBOARD_PASSWORD")

# Initialize database
init_db()

# Only train FastText model if needed, and only once when app starts
if 'fasttext_trained' not in st.session_state:
    try:
        train_fasttext_if_needed()
        st.session_state['fasttext_trained'] = True
    except Exception as e:
        st.session_state['fasttext_trained'] = False
        logger.warning(f"FastText training failed: {e}. Using keyword-based relevance checking only.")
    
# Initialize services
try:
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

# Function to process a query and get a response
def process_query(query, language="it"):
    try:
        # Translate the query to Italian if the user is using English
        original_query = query
        if language == "en":
            query = translate_to_italian(query)
            logger.info(f"Translated query: {query}")
        
        # Check relevance
        is_relevant, details = st.session_state['relevance_checker'].is_relevant(query)
        
        if not is_relevant:
            log_event("out_of_scope", query=original_query)
            
            if language == "it":
                response = "Mi dispiace, ma posso rispondere solo a domande relative a tasse, IVA e questioni fiscali. Posso aiutarti con domande su questi argomenti?"
            else:  # English
                response = "I'm sorry, but I can only answer questions about taxes, VAT, and fiscal matters. Can I help you with questions on these topics?"
        else:
            # Process relevant query
            conversation_string = get_conversation_string()
            refined_query = query_refiner(conversation_string, query)
            italian_response = find_match(refined_query)
            
            log_event("answered", query=original_query, response=italian_response)
            
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
        logging.error(f"ERROR | Query: {query} | Exception: {str(e)}")
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

# Initialize language selection in session state
if 'language' not in st.session_state:
    st.session_state.language = "it"  # Default to Italian

query_params = st.query_params
page = query_params.get("page", "chat")  # Default to "chat" if not set

# Function to convert image to base64 for HTML embedding
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


if page == "monitor":
    from monitor.db_logger import get_all_logs, export_logs_to_csv
    import pandas as pd
    import altair as alt
    from io import StringIO

    # Load dashboard password from .env or st.secrets (only if secrets exist)
    DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD")

    try:
        if not DASHBOARD_PASSWORD and "DASHBOARD_PASSWORD" in st.secrets:
            DASHBOARD_PASSWORD = st.secrets["DASHBOARD_PASSWORD"]
    except FileNotFoundError:
        pass  # Ignore if running locally without secrets.toml

    # Protect the dashboard tab with a password
    if "monitor_authenticated" not in st.session_state:
        password = st.sidebar.text_input("🔐 Inserisci password", type="password")
        
        if password and password == DASHBOARD_PASSWORD:
            st.session_state["monitor_authenticated"] = True
        elif password:
            st.warning("Accesso negato.")
            st.stop()

    # Still not authenticated? Stop app
    if not st.session_state.get("monitor_authenticated", False):
        st.stop()

    rows = get_all_logs()
    if not rows:
        st.warning("⚠️ Nessun dato nei log.")
        st.stop()

    df = pd.DataFrame(rows, columns=["id", "timestamp", "event", "query", "response", "feedback", "response_time"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    st.subheader("📌 Metriche chiave")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("✅ Risposte", df[df.event == "answered"].shape[0])
    col2.metric("❌ Fuori ambito", df[df.event == "out_of_scope"].shape[0])
    col3.metric("👍 Feedback positivo", df[df.feedback == "👍"].shape[0])
    col4.metric("👎 Feedback negativo", df[df.feedback == "👎"].shape[0])

    st.markdown("---")
    st.subheader("📈 Trend")

    df = df.dropna(subset=["timestamp"])
    if not df.empty:
        answered_over_time = df[df.event == "answered"].groupby(pd.Grouper(key="timestamp", freq="15min")).size().reset_index(name="count")
        out_of_scope_over_time = df[df.event == "out_of_scope"].groupby(pd.Grouper(key="timestamp", freq="15min")).size().reset_index(name="count")
        feedback_over_time = df[df.event == "feedback"].groupby([pd.Grouper(key="timestamp", freq="15min"), "feedback"]).size().unstack(fill_value=0).reset_index()
        response_times = df.dropna(subset=["response_time"])

        st.altair_chart(
            alt.Chart(answered_over_time).mark_line(point=True).encode(
                x="timestamp:T", y="count:Q", tooltip=["timestamp:T", "count:Q"]
            ).properties(title="✅ Risposte nel tempo").interactive(),
            use_container_width=True
        )

        st.altair_chart(
            alt.Chart(out_of_scope_over_time).mark_line(point=True, color="orange").encode(
                x="timestamp:T", y="count:Q", tooltip=["timestamp:T", "count:Q"]
            ).properties(title="❌ Fuori ambito nel tempo").interactive(),
            use_container_width=True
        )

        feedback_melted = feedback_over_time.melt(id_vars="timestamp", var_name="Feedback", value_name="Count")
        st.altair_chart(
            alt.Chart(feedback_melted).mark_bar().encode(
                x="timestamp:T", y="Count:Q", color="Feedback:N", tooltip=["timestamp:T", "Feedback:N", "Count:Q"]
            ).properties(title="👍👎 Feedback nel tempo").interactive(),
            use_container_width=True
        )

        if not response_times.empty:
            st.line_chart(response_times.set_index("timestamp")["response_time"].rename("Tempo di risposta (s)"))

    st.markdown("---")
    st.subheader("📄 Esplora log")
    st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)

    # Download CSV
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button("📥 Scarica log CSV", data=csv_buffer.getvalue(), file_name="fiscozen_chatbot_logs.csv", mime="text/csv")

else:
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
            avatar_style="no-avatar"  # This disables the avatar
        )

    # Show feedback only for the last assistant message
    if st.session_state.chat_history:
        last_bot_msg = next(
            (msg for msg in reversed(st.session_state.chat_history) if not msg["is_user"]),
            None
        )
        if last_bot_msg:
            feedback_key = f"feedback_{last_bot_msg['key']}"
            feedback = st.radio(
                "Ti è stata utile questa risposta?" if st.session_state.language == "it" else "Was this response helpful?",
                ["👍", "👎"],
                index=None,
                key=feedback_key,
                horizontal=True
            )
            if feedback:
                last_user_msg = next(
                    (msg["message"] for msg in reversed(st.session_state.chat_history) if msg["is_user"]),
                    ""
                )
                st.session_state['pending_feedback'] = {
                    "query": last_user_msg,
                    "response": last_bot_msg["message"],
                    "feedback": feedback
                }

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

    # Process user input (with pending feedback handling)
    if user_input and not st.session_state.processing:
        # If there's pending feedback, log it now
        if 'pending_feedback' in st.session_state:
            fb = st.session_state.pop('pending_feedback')
            log_event("feedback", query=fb['query'], response=fb['response'], feedback=fb['feedback'])

        # Store new user input
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
            start_time = time.time()
            response = process_query(last_user_message, st.session_state.language)

            # Log performance metrics
            duration = time.time() - start_time
            log_event("perf", query=last_user_message, response_time=duration)

            bot_msg_key = f"bot_msg_{len(st.session_state.chat_history)}"
            st.session_state.chat_history.append({
                "message": response,
                "is_user": False,
                "key": bot_msg_key
            })

            # Clear previous feedback selections
            for key in list(st.session_state.keys()):
                if key.startswith("feedback_"):
                    del st.session_state[key]

            # Prepare new pending feedback (to be logged later)
            st.session_state["pending_feedback"] = {
                "query": last_user_message,
                "response": response,
                "feedback": None  # Will be updated when user makes a choice
            }

        st.session_state.processing = False
        st.rerun()
