from langchain_openai import OpenAIEmbeddings
import openai
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import streamlit as st

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
        max_tokens=256,
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