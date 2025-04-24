# FiscoChat: AI-Powered Tax Assistant

FiscoChat is an intelligent chatbot designed to assist freelancers and sole proprietors in Italy with tax and VAT-related questions. Built with advanced RAG (Retrieval-Augmented Generation) technology, it combines the power of GPT-4 with domain-specific knowledge to provide accurate, contextually relevant tax information.

## Key Features

- **RAG Implementation**: Retrieves relevant tax information from a Pinecone vector database for accurate, knowledge-grounded responses
- **Multilingual Support**: Seamlessly operates in both Italian and English with automatic translation
- **FastText Topic Filtering**: Ensures queries are relevant to Italian tax topics before processing
- **Conversation Memory**: Maintains chat history for coherent multi-turn dialogues
- **Interactive UI**: Clean Streamlit interface with suggested questions and language selection
- **Monitoring Dashboard**: Analytics dashboard to track chatbot performance and user feedback
- **Docker Support**: Easy deployment via containerization
- **Cross-Platform Compatibility**: Works on macOS (Intel & Apple Silicon), Windows, and Linux

## Features

- **Document Processing Pipeline**: Extracts text from PDFs, processes/chunks, and stores in Pinecone
- **Semantic Search**: Retrieves relevant document chunks based on natural language queries
- **FastText Topic Filtering**: Ensures queries are relevant to Italian tax topics before processing
- **Advanced Prompting**: Uses retrieval-augmented generation (RAG) with contextual information
- **PDF Handling**: Views source documents and highlights relevant sections

## Architecture Overview

FiscoChat employs a sophisticated architecture:

1. **User Interface**: Streamlit-based interface with support for both Italian and English
2. **Query Processing Pipeline**:
   - Language detection and translation (if needed)
   - FastText relevance check to ensure query is tax-related
   - Context-aware query refinement using conversation history
   - Document retrieval from Pinecone vector database
   - Response generation with GPT-4
   - Translation back to user's language (if needed)
3. **Monitoring System**: SQLite-based logging and Streamlit dashboard for performance tracking

## Pinecone Integration

FiscoChat uses [Pinecone](https://www.pinecone.io/) as its vector database to power the RAG system:

- **Centralized Knowledge Base**: All tax documents are embedded and stored in Pinecone, eliminating the need to distribute large document datasets with the code repository
- **Semantic Search**: Pinecone enables fast, accurate retrieval of relevant context based on the semantic meaning of user queries
- **Scalable Performance**: Handles thousands of document chunks with millisecond query times
- **Consistent Knowledge Access**: Multiple instances of FiscoChat can access the same knowledge base using just an API key

The system is designed to work with a pre-populated Pinecone index containing all the tax-related documents, allowing for a lightweight repository deployment while maintaining robust knowledge retrieval capabilities.

## Hybrid RAG + GPT-4 Architecture

This system uses a sophisticated hybrid approach for answering tax-related questions:

1. **FastText Relevance Filtering**: First, user queries are analyzed with a FastText model to determine if they're related to tax topics.

2. **RAG-based Knowledge Retrieval**: For relevant questions, we search our Pinecone vector database for documents containing specific information related to the query.

3. **Smart Fallback System**: When relevant information is not found in our knowledge base, instead of returning "I don't know", the system automatically:
   - Detects that the RAG system doesn't have sufficient information
   - Seamlessly falls back to GPT-4 with specialized tax domain knowledge
   - Provides a substantive response based on general tax expertise

4. **Follow-up Question Processing**: The system maintains conversation context to understand and properly handle follow-up questions.

This hybrid approach combines the precision of RAG (Retrieval Augmented Generation) with the breadth of GPT-4's knowledge, providing the best of both worlds:
- RAG ensures answers are grounded in verified information when available
- GPT-4 fallback ensures users still get helpful responses even when specific information isn't in our database

### Testing the System

The included test script `test_rag_local.py` verifies all components of the system:

```bash
# Run the comprehensive test suite
./test_rag_local.py
```

The tests verify:
- FastText relevance detection
- Pinecone connectivity
- RAG-based responses
- GPT-4 fallback functionality
- Follow-up question handling

### Deployment

To deploy this system to Google Cloud Run:

1. Configure your environment variables:
   ```bash
   export OPENAI_API_KEY=your_openai_key
   export PINECONE_API_KEY=your_pinecone_key
   ```

2. Edit the deployment script with your Google Cloud project ID:
   ```bash
   # In deploy_to_cloud.sh
   PROJECT_ID="your-project-id"
   ```

3. Run the deployment script:
   ```bash
   ./deploy_to_cloud.sh
   ```

## Getting Started

### Prerequisites
- Python 3.8+
- OpenAI API key
- Pinecone API key

### Installation

Below are the step-by-step instructions for installing and running FiscoChat on different platforms:

#### MacOS (Intel and Apple Silicon)

```bash
# Clone the repository
git clone https://github.com/yourusername/fiscochat.git
cd fiscochat

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install FastText separately (important)
pip install fasttext

# Create and configure your .env file
cat > .env << EOL
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
DASHBOARD_PASSWORD=your_dashboard_password_here
EOL

# Train the FastText model and run the chatbot
python run_fiscozen.py --train-pinecone
```

#### Windows

```bash
# Clone the repository
git clone https://github.com/yourusername/fiscochat.git
cd fiscochat

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install FastText (may require Visual C++ Build Tools)
pip install fasttext-wheel

# Create and configure your .env file with your preferred editor
# Add the following lines:
# OPENAI_API_KEY=your_openai_api_key_here
# PINECONE_API_KEY=your_pinecone_api_key_here
# DASHBOARD_PASSWORD=your_dashboard_password_here

# Train the FastText model and run the chatbot
python run_fiscozen.py --train-pinecone
```

#### Linux (Ubuntu/Debian)

```bash
# Install required system dependencies
sudo apt-get update
sudo apt-get install -y build-essential python3-dev python3-venv

# Clone the repository
git clone https://github.com/yourusername/fiscochat.git
cd fiscochat

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install FastText separately
pip install fasttext

# Create and configure your .env file
cat > .env << EOL
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
DASHBOARD_PASSWORD=your_dashboard_password_here
EOL

# Train the FastText model and run the chatbot
python run_fiscozen.py --train-pinecone
```

### Troubleshooting

#### FastText Installation Issues

If you encounter issues with FastText installation:

**On macOS:**
```bash
# Apple Silicon fallback
pip install fasttext-wheel --no-build-isolation
```

**On Windows:**
```bash
# Install Visual C++ Build Tools first, then:
pip install fasttext-wheel
```

**On Linux:**
```bash
# Make sure you have the required build tools
sudo apt-get install -y build-essential python3-dev
pip install fasttext
```

#### Missing FastText Model

If the FastText model training fails with "Training scripts not found" error:

1. Create the directories:
   ```bash
   mkdir -p fast_text/models
   ```

2. Run the correct training script:
   ```bash
   python fast_text/train_with_pinecone.py
   ```

### Running the Application

After installation, you can start the chatbot by running:

```bash
python run_fiscozen.py
```

Access the web interface at: http://localhost:8501

## Training the FastText Model with Pinecone Data

FiscoChat includes a FastText model for classifying whether queries are tax-related. This model is trained using data stored in your Pinecone vector database.

You can explicitly trigger training:

```bash
# Train the model
python run_fiscozen.py --train-pinecone
```

Once training is complete, the model will be saved to `fast_text/models/tax_classifier.bin` and used automatically by the application.

## Updating Knowledge Base

To add new documents to the chatbot's knowledge:

1. Place documents in the `data_documents/` folder
2. Run the ingestion script:
   ```bash
   python ingestion/main.py
   ```

## Docker Deployment

### Building the Docker image
```bash
docker build -t fiscochat .
```

### Running the container
```bash
docker run -d -p 8501:8501 \
  -e OPENAI_API_KEY=your_openai_api_key \
  -e PINECONE_API_KEY=your_pinecone_api_key \
  -e DASHBOARD_PASSWORD=your_dashboard_password \
  fiscochat
```

## Technical Details

### Components

- **app.py**: Main Streamlit application and UI
- **utils.py**: Core utilities for RAG implementation, translation, and query processing
- **fast_text/**: Topic relevance classifier
- **monitor/**: Monitoring dashboard and logging functionality
- **ingestion/**: Document processing and vector database management
- **fast_text/train_with_pinecone.py**: Script for training FastText using Pinecone data
- **run_fiscozen.py**: Launcher script with automatic dependency management
- **main.py**: Core setup functions and utilities

### Dependencies

- **LangChain**: Framework for RAG implementation
- **OpenAI**: GPT-4 for response generation and embeddings
- **Pinecone**: Vector database for document retrieval
- **FastText**: ML model for topic classification
- **Streamlit**: Web interface and dashboard

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- OpenAI for GPT models
- Pinecone for vector database technology
- FastText for text classification

# FiscoChat API: Tax Chatbot Deployment

This is a deployment package for the FiscoChat Tax Chatbot API.

## Features

- **RAG Implementation**: Retrieves relevant tax information from a Pinecone vector database
- **Multilingual Support**: Seamlessly operates in both Italian and English with automatic translation
- **FastText Topic Filtering**: Ensures queries are relevant to Italian tax topics
- **Conversation Memory**: Maintains chat history for coherent multi-turn dialogues
- **Docker Support**: Easy deployment via containerization
- **Cloud-Ready**: Pre-configured for Google Cloud Run and App Engine

## Quickstart

### Local Development

```bash
# Clone the repository
git clone <repository-url>
cd fiscochat

# Make scripts executable
chmod +x run-local.sh run-docker.sh

# Option 1: Run locally with Python
./run-local.sh

# Option 2: Run with Docker
./run-docker.sh
```

### Cloud Deployment

#### Deploy to Google Cloud Run

```bash
# Build and deploy with a single command
gcloud run deploy fiscochat-api \
  --source . \
  --platform managed \
  --region europe-west1 \
  --memory 2Gi \
  --set-env-vars="OPENAI_API_KEY=your_key,PINECONE_API_KEY=your_key"
```

## Documentation

For detailed deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).

## Troubleshooting

If you encounter issues with the local setup:

1. Make sure your API keys are set in the `.env` file
2. Install gunicorn manually: `pip install gunicorn`
3. Check that all dependencies are installed: `pip install -r deployment-requirements.txt`

For Docker issues, ensure that both Docker and Docker Compose are installed and running on your system.

## API Endpoints

- `POST /api/chat`: Send a message to the chatbot
- `GET /api/history`: Get conversation history
- `GET /api/health`: Health check endpoint
- `GET /`: API documentation
