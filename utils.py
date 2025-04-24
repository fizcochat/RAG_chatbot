from langchain_openai import OpenAIEmbeddings
import openai
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import streamlit as st
from monitor.db_logger import log_event
import requests

from ingestion.config import MODEL, INDEX, PINECONE_ENV

# Load the environment variables from the .env file
def initialize_services(openai_api_key, pinecone_api_key):
    # Set OpenAI API key
    openai.api_key = openai_api_key

    # Initialize OpenAI Embeddings model
    model = OpenAIEmbeddings(model=MODEL,openai_api_key=openai.api_key)

    # Initialize Pinecone with API key
    pc = Pinecone(api_key=pinecone_api_key, environment=PINECONE_ENV)
    index = pc.Index(INDEX)

    # Set up Pinecone VectorStore
    vectorstore = PineconeVectorStore(index, model, "text")

    # Initialize OpenAI client
    client = openai.OpenAI(api_key=openai.api_key)

    return vectorstore, client

def find_match(vectorstore, query):
    result = vectorstore.similarity_search(query, k=5)
    
    # Create a short preview of the chunks for monitoring 
    retrieved_chunks = [doc.page_content if hasattr(doc, "page_content") else str(doc) for doc in result]
    chunk_previews = [chunk[:100] + "..." if len(chunk) > 100 else chunk for chunk in retrieved_chunks]
    log_event("rag_success", query=query, response="\n---\n".join(chunk_previews), feedback=None)

    return str(result)

def query_refiner(client, conversation, query):
    # Take only the last 2 exchanges from the conversation
    conversation_lines = conversation.split('\n')[-4:]
    shortened_conversation = '\n'.join(conversation_lines)
    
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=f"""Given the following user query and conversation log, formulate a more detailed and specific question that would help retrieve the most relevant information from a knowledge base about Italian taxation and VAT management.

The refined question should:
- Be more specific than the original query
- Include relevant technical terms (e.g., proper tax terminology)
- Maintain the original intent
- Be longer and more detailed than the original query
- Focus exclusively on Italian tax matters and Fiscozen services

CONVERSATION LOG: 
{shortened_conversation}

Original Query: {query}

Refined Query:""",
        temperature=0.7,
        max_tokens=150,  # Reduced from 256
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].text

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i+1] + "\n"
    return conversation_string

def fetch_external_knowledge(query):
    """
    Fetch relevant data from external knowledge sources based on query content.
    Returns formatted data that can be added to the context.
    """
    knowledge = []
    api_types_used = []
    
    # Define API endpoints
    endpoints = {
        "locations": "https://fizcochat.github.io/partita_iva_api/api/locations.json",
        "professions": "https://fizcochat.github.io/partita_iva_api/api/professions.json",
        "tax_regimes": "https://fizcochat.github.io/partita_iva_api/api/tax-regimes.json"
    }
    
    # Initialize cache in session state if it doesn't exist
    if 'api_cache' not in st.session_state:
        st.session_state.api_cache = {}
    
    # Location keywords
    locations = ["Milan", "Rome", "Florence", "Remote", "Naples", "Turin"]
    if any(location.lower() in query.lower() for location in locations):
        try:
            # Use cached data if available
            if "locations" in st.session_state.api_cache:
                data = st.session_state.api_cache["locations"]
            else:
                response = requests.get(endpoints["locations"])
                response.raise_for_status()
                data = response.json()
                # Cache the data
                st.session_state.api_cache["locations"] = data
            
            # Find which location was mentioned
            for location in locations:
                if location.lower() in query.lower() and location in data:
                    location_info = "\n".join([f"- {tip}" for tip in data[location]])
                    knowledge.append(f"Location-specific information for {location}:\n{location_info}")
                    api_types_used.append("locations")
                    log_event("external_api", query=query, api_type="locations")
        except Exception as e:
            print(f"Error fetching location data: {e}")
    
    # Profession keywords
    professions = ["Designer", "Software", "Photographer", "Digital Marketer", "Yoga", "Translator"]
    if any(profession.lower() in query.lower() for profession in professions):
        try:
            # Use cached data if available
            if "professions" in st.session_state.api_cache:
                data = st.session_state.api_cache["professions"]
            else:
                response = requests.get(endpoints["professions"])
                response.raise_for_status()
                data = response.json()
                # Cache the data
                st.session_state.api_cache["professions"] = data
            
            # Find which profession was mentioned
            for key, profession in [("Freelance Designer", "Designer"), ("Software Consultant", "Software"), 
                               ("Photographer", "Photographer"), ("Digital Marketer", "Digital Marketer"),
                               ("Yoga Instructor", "Yoga"), ("Translator", "Translator")]:
                if profession.lower() in query.lower() and key in data:
                    profession_info = "\n".join([f"- {tip}" for tip in data[key]])
                    knowledge.append(f"Profession-specific information for {key}:\n{profession_info}")
                    api_types_used.append("professions")
                    log_event("external_api", query=query, api_type="professions")
        except Exception as e:
            print(f"Error fetching profession data: {e}")
    
    # Tax regime keywords
    tax_regimes = ["Forfettario", "Ordinario", "Regime dei Minimi", "Regime Ordinario Semplificato"]
    if any(regime.lower() in query.lower() for regime in tax_regimes):
        try:
            # Use cached data if available
            if "tax_regimes" in st.session_state.api_cache:
                data = st.session_state.api_cache["tax_regimes"]
            else:
                response = requests.get(endpoints["tax_regimes"])
                response.raise_for_status()
                data = response.json()
                # Cache the data
                st.session_state.api_cache["tax_regimes"] = data
            
            # Find which tax regime was mentioned
            for regime in tax_regimes:
                if regime.lower() in query.lower() and regime in data:
                    regime_info = "\n".join([f"- {tip}" for tip in data[regime]])
                    knowledge.append(f"Tax regime information for {regime}:\n{regime_info}")
                    api_types_used.append("tax_regimes")
                    log_event("external_api", query=query, api_type="tax_regimes")
        except Exception as e:
            print(f"Error fetching tax regime data: {e}")
    
    return "\n\n".join(knowledge)