# Fisco-Chat

Fisco-Chat is an AI-powered chatbot designed to assist freelancers and sole proprietors in Italy with tax and VAT management, leveraging LangChain, OpenAI's LLM models, and FastText for topic relevance filtering.

## Features

- **Multilingual Support**: Available in both Italian and English with seamless translation
- **RAG Implementation**: Retrieves information from vector databases for accurate, context-aware responses
- **FastText Topic Filtering**: Ensures responses stay focused on Italian tax-related topics
- **GPT-4 Integration**: Leverages OpenAI's advanced LLMs for natural conversations
- **Context Awareness**: Maintains chat history for coherent dialogue
- **User-Friendly Interface**: Intuitive Streamlit-based UI with suggested questions

## Prerequisites

Before running the project, ensure you have the following installed:
- Python 3.8+
- pip (Python package manager)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root and add the following keys:
```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```
Replace `your_openai_api_key` and `your_pinecone_api_key` with your actual API keys.

### 5. Run the Application

You can run the application in two ways:

#### Option A: Direct Streamlit Execution (Recommended)
```bash
streamlit run app.py
```

#### Option B: Using the Launch Script
```bash
python run_fiscozen.py
```

The application will automatically:
1. Check and install required dependencies
2. Set up necessary directories
3. Train or load the FastText model for topic filtering
4. Launch the Streamlit interface

Once started, access the chatbot at http://localhost:8501/

### 6. Access and Run the Private Docker Image

- Accept the Invitation to Access the Docker Image: Check your GitHub Notifications or email for an invitation to access the Fiscochat package. Click Accept Invitation to gain access.
- Generate a GitHub Personal Access Token (PAT):
  - Log in to GitHub.
  - Click your profile picture → Settings.
  - Go to Developer Settings → Personal Access Tokens → Fine-grained Tokens.
  - Click Generate New Token and configure the following: Repository Access: Select the repository containing the Fiscochat image. Expiration: Choose an expiration date or select No expiration. Permissions: read:packages (Required to pull Docker images)
  - Click Generate Token and copy the token.
- Open your terminal and type:
```bash
echo YOUR_GITHUB_PAT | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
```
- Pull the docker image:
```bash
docker pull ghcr.io/fizcochat/fizcochat/rag_chatbot:latest
```
- Run the docker container:
```bash  
docker run -d -p 8501:8501 -e OPENAI_API_KEY=your_openai_api_key -e PINECONE_API_KEY=your_pinecone_api_key ghcr.io/fizcochat/fizcochat/rag_chatbot:latest
```
- Now you can access the chatbot at http://localhost:8501/

## Documentation

For detailed information about specific features:

- [Multilingual Support](DOCUMENTATION/multilingual.md) - How the chatbot handles multiple languages
- [Product Vision](DOCUMENTATION/Product%20Vision%20Board.pdf) - Overall product vision and roadmap
- [Project Proposal](DOCUMENTATION/Project%20Proposal%20Document.pdf) - Original project proposal and scope

## How It Works

### Language Processing Flow

1. **User Input**: The system accepts queries in either Italian or English
2. **Language Detection**: Automatically processes based on user's language selection
3. **For English Users**:
   - Translates the query to Italian
   - Processes using the Italian tax knowledge base
   - Translates responses back to English
4. **Topic Filtering**: FastText model ensures queries are relevant to Italian taxation
5. **Context Management**: Maintains conversation history for coherent dialogue
6. **Response Generation**: Uses retrieved information to generate accurate, contextual answers

### Key Components

- **FastText Model**: Filters queries to ensure they're tax-related
- **OpenAI GPT-4**: Powers the core language understanding and generation
- **Pinecone Vector Database**: Stores and retrieves relevant tax information
- **Streamlit UI**: Provides an intuitive, user-friendly interface

## Troubleshooting

- **Missing API Keys**: Ensure your `.env` file is correctly set up.
- **Dependency Errors**: Run `pip install -r requirements.txt` again to ensure all dependencies are installed.
- **Streamlit Not Found**: Run `pip install streamlit` inside your virtual environment.
- **Language Issues**: If translation doesn't work, check your OpenAI API key and quota.
- **FastText Model Missing**: If the chatbot complains about a missing model, ensure you have write permissions in the fasttext/models directory.

## Contributing

If you'd like to contribute, feel free to open issues or submit pull requests.

## License

This project is licensed under [MIT License](LICENSE).
