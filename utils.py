from langchain_openai import OpenAIEmbeddings
import openai
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import streamlit as st
from monitor.db_logger import log_event

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
        prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{shortened_conversation}\n\nQuery: {query}\n\nRefined Query:",
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