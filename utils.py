from langchain_openai import OpenAIEmbeddings
import openai
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.schema import HumanMessage, AIMessage
import streamlit as st
import logging

# Load the environment variables from the .env file
def initialize_services(openai_api_key, pinecone_api_key):
    """Initialize OpenAI and Pinecone services"""
    # Set OpenAI API key
    openai.api_key = openai_api_key

    # Initialize OpenAI Embeddings model
    model = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai.api_key)

    # Initialize Pinecone with API key
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index("fiscozen")

    # Set up Pinecone VectorStore
    vectorstore = PineconeVectorStore(index, model, "text")

    # Initialize OpenAI client
    client = openai.OpenAI(api_key=openai.api_key)

    return vectorstore, client

def translate_to_italian(text, client=None):
    """Translate text from English to Italian using OpenAI."""
    if not text:
        return ""
        
    try:
        # Get OpenAI client from session state if not provided
        if client is None:
            if 'openai_client' not in st.session_state:
                raise ValueError("OpenAI client not initialized")
            client = st.session_state['openai_client']
        
        # Use OpenAI to translate
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional translator. Translate the following text from English to Italian, maintaining the same tone and style. Only return the translated text without any additional commentary."},
                {"role": "user", "content": text}
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Translation error (to Italian): {e}")
        return text  # Return original text if translation fails

def translate_from_italian(text, client=None):
    """Translate text from Italian to English using OpenAI."""
    if not text:
        return ""
        
    try:
        # Get OpenAI client from session state if not provided
        if client is None:
            if 'openai_client' not in st.session_state:
                raise ValueError("OpenAI client not initialized")
            client = st.session_state['openai_client']
        
        # Use OpenAI to translate
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional translator. Translate the following text from Italian to English, maintaining the same tone and style. Only return the translated text without any additional commentary."},
                {"role": "user", "content": text}
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Translation error (from Italian): {e}")
        return text  # Return original text if translation fails

def find_match(query, k=2):
    """Find similar documents in the vector store and format them into a readable response"""
    try:
        # Get vectorstore from session state
        if 'vectorstore' not in st.session_state:
            raise ValueError("Vector store not initialized")
        vectorstore = st.session_state['vectorstore']

        # Perform similarity search
        docs = vectorstore.similarity_search(query, k=k)
        
        logging.info(f"RAG | Query: {query} | Retrieved Chunks: {len(docs)}")
        chunk_preview = [doc.page_content[:60].replace("\n", " ") for doc in docs]
        logging.info(f"RAG | Chunks Preview: {chunk_preview}")
        
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
            model="gpt-4",  # Upgrade to GPT-4 to match new version
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
        
        # Get conversation history from memory
        if 'memory' in st.session_state:
            memory_buffer = st.session_state['memory']
            chat_history = memory_buffer.load_memory_variables({})
            history_str = ""
            
            # Format the conversation history
            if "chat_history" in chat_history and chat_history["chat_history"]:
                for message in chat_history["chat_history"]:
                    if isinstance(message, HumanMessage):
                        history_str += f"Human: {message.content}\n"
                    elif isinstance(message, AIMessage):
                        history_str += f"Assistant: {message.content}\n"
        else:
            # Fallback to basic conversation string if memory not available
            history_str = conversation
        
        # Create the system prompt for query refinement
        system_prompt = """Given a conversation history and user query, formulate a question that would be most relevant to provide a helpful answer from our knowledge base about Italian taxes, IVA, and fiscal matters."""
        
        # Create the user prompt with conversation and query
        user_prompt = f"""CONVERSATION HISTORY:
{history_str}

CURRENT QUERY: {query}

Please consider:
1. Previous context from the conversation
2. Specific tax-related terms mentioned
3. Any clarifications or follow-up questions needed

REFINED QUERY:"""
        
        # Use chat completions API instead of completions
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=256
        )
        
        refined_query = response.choices[0].message.content.strip()
        print(f"Original query: {query}")
        print(f"Refined query: {refined_query}")
        return refined_query
        
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