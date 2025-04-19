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

You can run FiscoChat in several ways:

#### Option 1: Using the launcher script (recommended)
```bash
python run_fiscozen.py
```

This script will automatically:
- Check and install any missing dependencies
- Handle platform-specific configurations
- Start the Streamlit server with the correct settings

#### Option 2: Directly with Streamlit
```bash
streamlit run app.py
```

The application will:
- Install any missing dependencies
- Train or load the FastText model (with fallback to keyword-based filtering if FastText isn't available)
- Set up required directories
- Launch the Streamlit interface at http://localhost:8501

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

### Dependencies

- **LangChain**: Framework for RAG implementation
- **OpenAI**: GPT-4 for response generation and embeddings
- **Pinecone**: Vector database for document retrieval
- **FastText**: ML model for topic classification
- **Streamlit**: Web interface and dashboard

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
