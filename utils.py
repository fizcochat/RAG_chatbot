from langchain_openai import OpenAIEmbeddings
import openai
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import streamlit as st

# Load the environment variables from the .env file
def initialize_services(openai_api_key, pinecone_api_key):
    """Initialize OpenAI and Pinecone services"""
    # Set OpenAI API key
    openai.api_key = openai_api_key

    # Initialize OpenAI Embeddings model
    model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai.api_key)

    # Initialize Pinecone with API key
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index("ragtest")

    # Set up Pinecone VectorStore
    vectorstore = PineconeVectorStore(index, model, "text")

    # Initialize OpenAI client
    client = openai.OpenAI(api_key=openai.api_key)

    return vectorstore, client

def find_match(query, k=2):
    """Find similar documents in the vector store and format them into a readable response"""
    try:
        # Get vectorstore from session state
        if 'vectorstore' not in st.session_state:
            raise ValueError("Vector store not initialized")
        vectorstore = st.session_state['vectorstore']
        
        # Perform similarity search
        docs = vectorstore.similarity_search(query, k=k)
        
        # Extract and clean the content
        contents = []
        for doc in docs:
            # Clean up the content
            content = doc.page_content
            if content.startswith("page_content='"):
                content = content[13:-1]  # Remove page_content=' and final '
            contents.append(content)
        
        # Get OpenAI client from session state
        if 'openai_client' not in st.session_state:
            raise ValueError("OpenAI client not initialized")
        client = st.session_state['openai_client']
        
        # Use GPT to generate a coherent response from the documents
        combined_content = "\n".join(contents)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a tax assistant for Fiscozen. Create a clear, concise response in Italian based on the provided information. Focus on answering the user's question directly and professionally. Do not mention that you're using any source documents."},
                {"role": "user", "content": f"Query: {query}\n\nInformation:\n{combined_content}"}
            ]
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in find_match: {e}")
        return "Mi dispiace, non sono riuscito a trovare una risposta pertinente. Posso aiutarti con qualcos'altro?"

def query_refiner(conversation, query):
    """Refine the query based on conversation context"""
    try:
        # Get OpenAI client from session state
        if 'openai_client' not in st.session_state:
            raise ValueError("OpenAI client not initialized")
        client = st.session_state['openai_client']
        
        # Take only the last 2 exchanges from the conversation
        conversation_lines = conversation.split('\n')[-4:] if conversation else []
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
    except Exception as e:
        print(f"Error in query_refiner: {e}")
        return query  # Return original query if refinement fails

def get_conversation_string(conversation_id=None):
    """Get the conversation history as a string"""
    try:
        conversation_string = ""
        if 'responses' in st.session_state and 'requests' in st.session_state:
            for i in range(len(st.session_state['responses'])-1):
                conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
                conversation_string += "Bot: " + st.session_state['responses'][i+1] + "\n"
        return conversation_string
    except Exception as e:
        print(f"Error in get_conversation_string: {e}")
        return ""