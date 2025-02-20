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
```

You can now copy and paste this directly into your `README.md` file. Let me know if you need any modifications! 🚀