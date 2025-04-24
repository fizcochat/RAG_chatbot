#!/bin/bash

# Script to start the RAG chatbot API service on the local network

# Get the IP address
IP_ADDRESS=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -n 1)

echo "======================================================"
echo "   Starting RAG Chatbot API on http://$IP_ADDRESS:8080"
echo "======================================================"
echo ""
echo "API endpoints available:"
echo "  POST  http://$IP_ADDRESS:8080/api/chat    - Main chatbot endpoint"
echo "        Parameters:"
echo "          - message: The user's question (required)"
echo "          - session_id: Unique session ID (optional)"
echo "          - language: 'it' or 'en' for language (optional, default: it)"
echo ""
echo "  GET   http://$IP_ADDRESS:8080/api/health  - API health check"
echo ""
echo "  POST  http://$IP_ADDRESS:8080/api/clear   - Clear conversation history"
echo "        Parameters:"
echo "          - session_id: Session ID to clear (optional)"
echo ""
echo "Example Streamlit integration format:"
echo "  Request:"
echo "    {\"message\": \"Come funziona il regime forfettario?\", \"session_id\": \"user123\", \"language\": \"it\"}"
echo ""
echo "  Response:"
echo "    {\"session_id\": \"user123\", \"response\": \"Il regime forfettario è...\"}"
echo ""
echo "For example, you can call the API with curl:"
echo "  curl -X POST http://$IP_ADDRESS:8080/api/chat \\"
echo "       -H \"Content-Type: application/json\" \\"
echo "       -d '{\"message\":\"Come funziona il regime forfettario?\", \"session_id\":\"user123\", \"language\":\"it\"}'"
echo ""
echo "Press Ctrl+C to stop the server"
echo "======================================================"

# Install dependencies if needed
pip install -r requirements.txt

# Create models directory if it doesn't exist
mkdir -p fast_text/models

# Start the API server using gunicorn on all network interfaces
export PYTHONPATH=$PWD
echo "Starting API server..."
gunicorn --bind 0.0.0.0:8080 --workers 1 --threads 8 --timeout 0 main:app 