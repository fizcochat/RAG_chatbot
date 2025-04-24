"""
RAG Chatbot API Service
This implements a Flask REST API for the RAG chatbot system.
"""

import os
import time
import json
import dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import test_utils
from fast_text.relevance import FastTextRelevanceChecker

# Load environment variables
dotenv.load_dotenv()

# Initialize global variables to store services
global_store = {
    'vectorstore': None,
    'openai_client': None,
    'relevance_checker': None,
    'conversation_history': {},  # Store conversation history by session_id
    'last_request_time': {},     # Track last request time for each session
}

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Session timeout (30 minutes)
SESSION_TIMEOUT = 30 * 60  # 30 minutes in seconds

# Function to initialize services
def initialize_services():
    """Initialize OpenAI and Pinecone services."""
    # Get API keys from environment
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    
    if not openai_api_key or not pinecone_api_key:
        app.logger.error("API keys not found in environment variables")
        return
    
    # Initialize services
    try:
        global_store['vectorstore'], global_store['openai_client'] = test_utils.initialize_services(
            openai_api_key=openai_api_key,
            pinecone_api_key=pinecone_api_key
        )
        app.logger.info("Services initialized successfully")
        
        # Try to initialize FastText relevance checker
        try:
            global_store['relevance_checker'] = FastTextRelevanceChecker()
            app.logger.info("FastText relevance checker initialized successfully")
        except Exception as e:
            app.logger.warning(f"FastText relevance checker could not be initialized: {e}")
            app.logger.warning("Will skip relevance checking - all queries will be processed")
    
    except Exception as e:
        app.logger.error(f"Error initializing services: {e}")

# Initialize services when the app starts
with app.app_context():
    initialize_services()
    
# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify the API is running."""
    status = "healthy" if global_store.get('vectorstore') and global_store.get('openai_client') else "degraded"
    components = {
        "rag": global_store.get('vectorstore') is not None,
        "openai": global_store.get('openai_client') is not None,
        "fasttext": global_store.get('relevance_checker') is not None
    }
    
    return jsonify({
        "status": status,
        "components": components,
        "timestamp": time.time()
    })

# Clean expired sessions
def clean_expired_sessions():
    """Remove expired conversation sessions."""
    current_time = time.time()
    expired_sessions = []
    
    for session_id, last_time in global_store['last_request_time'].items():
        if current_time - last_time > SESSION_TIMEOUT:
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        if session_id in global_store['conversation_history']:
            del global_store['conversation_history'][session_id]
        del global_store['last_request_time'][session_id]

    return len(expired_sessions)

# Chat endpoint
@app.route('/api/chat', methods=['POST'])
def chat():
    """Process a chat request and return a response."""
    # Clean expired sessions
    expired_count = clean_expired_sessions()
    if expired_count > 0:
        app.logger.info(f"Cleaned {expired_count} expired sessions")
    
    # Get request data
    data = request.json
    if not data:
        return jsonify({"error": "No request data provided"}), 400
    
    # Extract parameters using the new format
    query = data.get('message')  # Changed from 'query' to 'message'
    session_id = data.get('session_id', 'default')
    language_code = data.get('language', 'it').lower()  # Using 'it' or 'en' format
    
    # Map language code to full language name
    language = 'italian'
    if language_code == 'en':
        language = 'english'
    
    if not query:
        return jsonify({"error": "No message provided"}), 400
    
    # Update session last request time
    global_store['last_request_time'][session_id] = time.time()
    
    # Create conversation history for this session if it doesn't exist
    if session_id not in global_store['conversation_history']:
        global_store['conversation_history'][session_id] = []
    
    # Process the query
    try:
        # Check relevance if FastText is available
        is_relevant = True
        relevance_details = {}
        
        if global_store.get('relevance_checker'):
            is_relevant, relevance_details = global_store['relevance_checker'].is_relevant(query)
        
        # If query is not relevant to tax/fiscal matters
        if global_store.get('relevance_checker') and not is_relevant:
            response = "Mi dispiace, ma posso rispondere solo a domande relative a tasse, fiscalità e servizi Fiscozen. Puoi provare a porre una domanda su questi argomenti?"
            # Use simplified response format for Streamlit integration
            return jsonify({
                "session_id": session_id,
                "response": response
            })
        
        # Translate to Italian if query is in English
        original_query = query
        if language == 'english':
            query = test_utils.translate_to_italian(query, global_store['openai_client'])
        
        # Use conversation history for context in follow-up questions
        if global_store['conversation_history'][session_id]:
            query = test_utils.query_refiner(global_store['conversation_history'][session_id], query)
        
        # Get response from RAG system
        response = test_utils.find_match(query)
        
        # Translate back to English if the original query was in English
        if language == 'english':
            response = test_utils.translate_from_italian(response, global_store['openai_client'])
            
        # Add to conversation history
        global_store['conversation_history'][session_id].append({
            "user": original_query,
            "assistant": response
        })
        
        # Limit conversation history to last 10 exchanges
        if len(global_store['conversation_history'][session_id]) > 10:
            global_store['conversation_history'][session_id] = global_store['conversation_history'][session_id][-10:]
        
        # Return simplified response format for Streamlit integration
        return jsonify({
            "session_id": session_id,
            "response": response
        })
            
    except Exception as e:
        app.logger.error(f"Error processing query: {e}")
        return jsonify({
            "error": "An error occurred while processing your request",
            "details": str(e)
        }), 500

# Clear conversation history endpoint
@app.route('/api/clear', methods=['POST'])
def clear_conversation():
    """Clear the conversation history for a session."""
    data = request.json
    if not data:
        return jsonify({"error": "No request data provided"}), 400
    
    session_id = data.get('session_id', 'default')
    
    if session_id in global_store['conversation_history']:
        global_store['conversation_history'][session_id] = []
    
    return jsonify({
        "status": "success",
        "message": f"Conversation history cleared for session {session_id}"
    })

# Main entry point
if __name__ == '__main__':
    # Get host IP for local network access
    host = '0.0.0.0'  # Listen on all network interfaces
    port = int(os.environ.get('PORT', 8080))

    # Start the server
    app.run(host=host, port=port, debug=True) 