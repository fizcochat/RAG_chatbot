from langchain_openai import OpenAIEmbeddings
import openai
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import streamlit as st
import re
from dotenv import load_dotenv
import os
from mock_api import MockAPI

# Load environment variables from .env file (won't override existing ones)
load_dotenv()

# Initialize the mock API instance
mock_api = MockAPI()

# Load the environment variables from the .env file
def initialize_services(openai_api_key, pinecone_api_key):
    # Set OpenAI API key
    openai.api_key = openai_api_key

    # Initialize OpenAI Embeddings model
    model = OpenAIEmbeddings(model="text-embedding-ada-002",openai_api_key=openai.api_key)

    # Initialize Pinecone with API key
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index("ragtest")

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

def detect_api_request(query):
    """
    Detects if the user query should trigger a call to the mock API.
    This function analyzes the user's query to determine if specific data needs to be retrieved.
    It acts as a natural language router to appropriate API endpoints.
    
    Args:
        query: The user's query string
        
    Returns:
        A tuple of (should_call_api, endpoint, parameters) or (False, None, None) if no API call needed
    """
    # Convert query to lowercase for easier matching
    query_lower = query.lower()
    
    # Check for invoice information requests
    # This section identifies invoice-related queries to call the external invoice API
    invoice_match = re.search(r'invoice(?:s)?\s+(?:information|data|details)?', query_lower) or re.search(r'my\s+invoice(?:s)?', query_lower)
    if invoice_match or "invoic" in query_lower:
        # Check for specific invoice ID
        invoice_id_match = re.search(r'invoice\s+(?:number|#|id|)\s*(\d+)', query_lower)
        if invoice_id_match:
            return True, "invoices", {"invoice_id": invoice_id_match.group(1)}
            
        # Check for specific customer ID
        customer_id_match = re.search(r'customer\s+(?:id|number)\s*(\d+)', query_lower)
        if customer_id_match:
            return True, "invoices", {"customer_id": customer_id_match.group(1)}
        
        # No specific parameters, return all invoices
        return True, "invoices", {}
    
    # Check for tax rate information requests
    # This section handles various tax rate queries using internal mock data
    if re.search(r'(tax|vat) rates?', query_lower) or re.search(r'(income tax|tax brackets)', query_lower):
        if "income" in query_lower or "bracket" in query_lower:
            return True, "tax_rates", {"type": "income_tax_brackets"}
        elif "reduced" in query_lower:
            return True, "tax_rates", {"type": "vat_reduced"}
        elif "super" in query_lower:
            return True, "tax_rates", {"type": "vat_super_reduced"}
        elif "standard" in query_lower:
            return True, "tax_rates", {"type": "vat_standard"}
        else:
            return True, "tax_rates", {}
    
    # Check for deadline information requests
    # This section handles tax deadline queries using internal mock data
    if re.search(r'(deadlines?|due dates?)', query_lower):
        if "vat" in query_lower and "quarterly" in query_lower:
            return True, "deadlines", {"type": "quarterly_vat"}
        elif "vat" in query_lower:
            return True, "deadlines", {"type": "vat_payment"}
        elif "annual" in query_lower or "tax return" in query_lower:
            return True, "deadlines", {"type": "annual_tax_return"}
        else:
            return True, "deadlines", {}
    
    # Check for user profile information requests
    # This section handles user profile queries using internal mock data
    # Look for words like "my profile", "my account", etc. along with a possible user ID
    user_id_match = re.search(r'user (?:id|profile) (\d+)', query_lower)
    if re.search(r'(my profile|my account|my information)', query_lower) or user_id_match:
        user_id = "123456"  # Default user ID for demonstration
        if user_id_match:
            user_id = user_id_match.group(1)
        return True, "user_profile", {"user_id": user_id}
    
    # No API call needed - will rely solely on RAG for response
    return False, None, None

def enrich_response_with_api_data(query, context, api_data=None):
    """
    Enhances the context with API data when relevant.
    This function integrates API responses into the RAG context before generating the final response.
    It formats different types of API data appropriately to ensure natural-sounding responses.
    
    Args:
        query: The original user query
        context: The context from vector search
        api_data: Optional API response data
        
    Returns:
        Enhanced context with API data incorporated if available
    """
    if not api_data:
        return context
    
    # Only add API data if the API call was successful
    if api_data.get("status") == 200:
        data = api_data.get("data", {})
        
        # Format the data based on the type of data received
        # This section handles invoice data from the external API
        if isinstance(data, list) and len(data) > 0 and "invoiceNumber" in data[0]:
            # We have invoice data as a list - format it in a user-friendly way
            formatted_data = "Invoice Information:\n"
            for i, invoice in enumerate(data, 1):
                formatted_data += f"Invoice #{i}:\n"
                formatted_data += f"  - Invoice Number: {invoice.get('invoiceNumber', 'N/A')}\n"
                formatted_data += f"  - Customer: {invoice.get('customerName', 'N/A')}\n"
                formatted_data += f"  - Amount: €{invoice.get('amount', 'N/A')}\n"
                formatted_data += f"  - Date: {invoice.get('date', 'N/A')}\n"
                formatted_data += f"  - Status: {invoice.get('status', 'N/A')}\n"
                if i < len(data):
                    formatted_data += "\n"
                
            # Add the formatted invoice data to the context
            enhanced_context = (
                f"{context}\n\n"
                f"Additional specific information from our database:\n"
                f"{formatted_data}"
            )
            
        elif isinstance(data, dict) and "invoiceNumber" in data:
            # We have a single invoice - format it in a user-friendly way
            formatted_data = "Invoice Information:\n"
            formatted_data += f"  - Invoice Number: {data.get('invoiceNumber', 'N/A')}\n"
            formatted_data += f"  - Customer: {data.get('customerName', 'N/A')}\n"
            formatted_data += f"  - Amount: €{data.get('amount', 'N/A')}\n"
            formatted_data += f"  - Date: {data.get('date', 'N/A')}\n"
            formatted_data += f"  - Status: {data.get('status', 'N/A')}\n"
            
            # Add the formatted invoice data to the context
            enhanced_context = (
                f"{context}\n\n"
                f"Additional specific information from our database:\n"
                f"{formatted_data}"
            )
        else:
            # Default formatting for other types of data (tax rates, deadlines, user profiles)
            enhanced_context = (
                f"{context}\n\n"
                f"Additional specific information from our database:\n"
                f"{str(data)}"
            )
            
        return enhanced_context
    
    # If API call failed, return original context without enhancement
    return context