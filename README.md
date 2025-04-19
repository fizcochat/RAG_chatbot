# Fisco-Chat

Fisco-Chat is an AI-powered chatbot designed to assist freelancers and sole proprietors in Italy with VAT management, leveraging LangChain and OpenAI's LLM models. 

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
```bash
streamlit run main.py
```

### 6. Open Monitoring Dashboard
This project includes a real-time monitoring dashboard built with Streamlit and SQLite for local inspection of chatbot performance:
- Tracks answered and out-of-scope queries
- Collects user feedback
- Measures response times

To run it locally in parallel with the chatbot:
```bash
streamlit run monitor/monitor_dashboard.py --server.port 8502
```
The dashboard auto-refreshes every 5 seconds and reads from a local SQLite database (monitor/logs.db). Logs are stored automatically as the chatbot is used.

### 7.  Access and Run the Private Docker Image

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
  docker pull ghcr.io/fizcochat/fizcochat/rag_chatbot:c2b27447fa48add23da1b773804833be4615e251
  ```
- Run the docker container:
```bash  
  docker run -d -p 8501:8501 -e OPENAI_API_KEY=your_openai_api_key -e PINECONE_API_KEY=your_pinecone_api_key ghcr.io/fizcochat/fizcochat/rag_chatbot:latest
```
- Now you can access the chatbot at http://localhost:8501/

### 8. Adding new data to the chatbot
To update the knowledge base with new documents:
1. Add your files to the folder:
    data_documents/

2. Run the ingestion script to automatically process and upload them to Pinecone:
   python ingestion/main.py

This script compares the files with the index saved in a local JSON, embeds any new documents using OpenAI embeddings, and uploads them to your configured Pinecone index. The chatbot will then be able to retrieve answers from the updated content immediately.

## Features
- Uses OpenAI's LLMs (GPT-3.5, GPT-4, GPT-4o, etc.)
- Retrieves information from vector databases for accurate responses
- Implements a structured workflow to guide users to AI assistance, CS consultants, or tax advisors
- Maintains chat history for context-aware responses

## Troubleshooting
- **Missing API Keys**: Ensure your `.env` file is correctly set up.
- **Dependency Errors**: Run `pip install -r requirements.txt` again to ensure all dependencies are installed.
- **Streamlit Not Found**: Run `pip install streamlit` inside your virtual environment.

## Contributing
If you'd like to contribute, feel free to open issues or submit pull requests.

## License
This project is licensed under [MIT License](LICENSE).
