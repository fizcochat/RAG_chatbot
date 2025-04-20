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

## Getting Started

### Prerequisites
- Python 3.8+
- OpenAI API key
- Pinecone API key

### Installation

#### Easy Installation (Cross-Platform)

We provide a cross-platform installation script that handles all dependencies for your specific system:

```bash
# Clone the repository
git clone https://github.com/yourusername/fiscochat.git
cd fiscochat

# Run the easy installer
python install_dependencies.py

# Start the chatbot
python run_fiscozen.py
```

The installer will automatically detect your operating system and install the appropriate dependencies, including handling platform-specific requirements for FastText.

#### Manual Installation

If you prefer to install dependencies manually:

1. **Set up environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```

2. **Install dependencies based on your platform**

   **For macOS (Apple Silicon):**
   ```bash
   pip install -r requirements.txt
   pip install fasttext  # Apple Silicon compatible version
   ```

   **For macOS (Intel):**
   ```bash
   pip install -r requirements.txt
   pip install fasttext
   ```

   **For Windows:**
   ```bash
   pip install -r requirements.txt
   # For Windows, you might need to install Visual C++ Build Tools first
   # Visit https://visualstudio.microsoft.com/visual-cpp-build-tools/
   pip install fasttext
   ```

   **For Linux:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install build-essential python3-dev
   pip install -r requirements.txt
   pip install fasttext
   ```

3. **Configure API keys**
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   DASHBOARD_PASSWORD=your_dashboard_password
   ```

### Running the Application

You can run FiscoChat in two recommended ways:

#### Option 1: Two-Step Process (recommended)
This approach provides more control over the setup process:

```bash
# Step 1: Train the FastText model using Pinecone data
python run_fiscozen.py --train-pinecone

# Step 2: Run the chatbot
python run_fiscozen.py
```

This workflow:
- First trains the FastText model using data stored in Pinecone 
- Then launches the chatbot with the trained model
- Gives you clear visibility into each step of the process

#### Option 2: Directly with Streamlit
For a more streamlined approach:

```bash
streamlit run app.py
```

With this method:
- The application automatically detects if the FastText model is missing
- If missing, it provides a button to train the model using Pinecone data
- After training completes, the app automatically reloads with the trained model

Both methods utilize the same Pinecone-powered knowledge base, so you don't need to have document files locally on your machine.

### Troubleshooting

#### FastText Installation Issues

If you encounter issues with FastText:

**On macOS (Apple Silicon):**
```bash
pip install fasttext
```

If that doesn't work:
```bash
pip install fasttext-wheel --no-build-isolation
```

**On Windows:**
FastText requires C++ build tools. If you encounter issues:
1. Install Visual C++ Build Tools from Microsoft
2. Or use the fallback keyword-based relevance filtering (automatic)

**On Linux:**
```bash
# Ensure build tools are installed
sudo apt-get install build-essential python3-dev  # Ubuntu/Debian
sudo yum install gcc gcc-c++ python3-devel        # CentOS/RHEL
```

#### LangChain Installation Issues

If you encounter issues with LangChain:
```bash
pip install langchain langchain-core langchain-openai
```

### Accessing the Monitoring Dashboard

The monitoring dashboard provides insights into chatbot performance:
- Query volume and types (answered vs. out-of-scope)
- User feedback statistics
- Response times
- Detailed logs of all interactions

To access:
1. Add `?page=monitor` to the URL: http://localhost:8501/?page=monitor
2. Enter the dashboard password configured in your `.env` file

## Training the FastText Model with Pinecone Data

FiscoChat includes a FastText model for classifying whether queries are tax-related. This model is trained using data stored in your Pinecone vector database, which makes it easy to share the same knowledge base across multiple installations without distributing large datasets.

### Automatic Training

When you first run the application, it will check if the FastText model exists. If not:

- If using `python run_fiscozen.py`, it will prompt you to train the model
- If using `streamlit run app.py`, it will display a "Train model using Pinecone data" button

### Manual Training

You can explicitly trigger training at any time:

```bash
# Train using the runner script
python run_fiscozen.py --train-pinecone

# Or use the dedicated training script directly
python fast_text/train_with_pinecone.py
```

### How Training Works

The training process:
1. Connects to your Pinecone index
2. Extracts tax-related examples from the stored documents
3. Generates balanced non-tax examples
4. Trains and saves the FastText model
5. Tests the model with sample queries

Once training is complete, the model will be saved to `fast_text/models/tax_classifier.bin` and used automatically by the application.

## Updating Knowledge Base

To add new documents to the chatbot's knowledge:

1. Place documents in the `data_documents/` folder
2. Run the ingestion script to process and upload to Pinecone:
   ```bash
   python ingestion/main.py
   ```

The script automatically:
- Compares files with the current index
- Embeds new documents using OpenAI
- Updates the Pinecone vector index

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

### Dependencies

- **LangChain**: Framework for RAG implementation
- **OpenAI**: GPT-4 for response generation and embeddings
- **Pinecone**: Vector database for document retrieval
- **FastText**: ML model for topic classification
- **Streamlit**: Web interface and dashboard

### FastText Model for Tax Relevance

The chatbot uses a fine-tuned FastText model to determine if a user query is tax-related:

- Classifies text into "Tax" and "Other" categories
- Trained on tax examples extracted from Pinecone
- Achieves high accuracy on test data 
- Helps filter out non-tax related queries to keep the chatbot focused on its domain expertise

The model is automatically loaded when the application starts. If the model file doesn't exist, you'll be prompted to train it using Pinecone data.

## Testing

Use the test plan outlined in `test_plan.md` to verify functionality:
- Basic functionality tests
- Multilingual support tests
- Topic filtering tests
- Monitoring functionality tests
- RAG system tests
- Error handling and edge cases
- Cross-platform compatibility

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- OpenAI for GPT models
- Pinecone for vector database technology
- FastText for text classification
