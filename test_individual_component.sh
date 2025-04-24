#!/bin/bash

echo "=========================================================="
echo "Individual Component Testing for RAG Chatbot System"
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

# Run the component tests
echo "=========================================================="
echo "Running individual component tests..."
echo "This will test each component separately:"
echo "1. FastText relevance checker"
echo "2. LangChain query refinement"
echo "3. Document retrieval from Pinecone"
echo "4. Translation functionality"
echo "5. Hybrid RAG-GPT4 responses"
echo "=========================================================="

# Run the test script
python test_components.py

echo "Individual component testing completed."
echo "You can now run the full system test with: ./run_rag_test.sh" 