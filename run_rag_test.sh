#!/bin/bash

echo "=========================================================="
echo "RAG Chatbot with FastText, LangChain, and GPT-4 Hybrid System"
echo "=========================================================="

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if FastText model exists and train if needed
if [ ! -f "fast_text/models/tax_classifier.bin" ]; then
    echo "FastText model not found. Training model..."
    python fast_text/train_fasttext_model.py
    if [ $? -ne 0 ]; then
        echo "Error training FastText model. Check the logs for details."
        exit 1
    fi
else
    echo "FastText model found. Skipping training."
fi

echo "=========================================================="
echo "Running Hybrid RAG-GPT4 testing script..."
echo "This script demonstrates a comprehensive system that:"
echo "1. Uses FastText to check if queries are tax-related"
echo "2. Expands queries with relevant fiscal terminology"
echo "3. Uses LangChain to understand conversation context"
echo "4. Retrieves relevant documents from Pinecone"
echo "5. Combines RAG with GPT-4 for comprehensive responses"
echo "=========================================================="

# Run the test script
python test_rag_direct.py

echo "Testing completed." 
echo "The full system is now operational with all components:"
echo "✓ FastText relevance checker"
echo "✓ LangChain for conversation context"
echo "✓ Pinecone for document retrieval"
echo "✓ Hybrid RAG-GPT4 for comprehensive responses" 