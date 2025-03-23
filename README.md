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

# Tax Chatbot Relevance Detection

This module adds relevance detection to your tax chatbot, warning users when they go off-topic.

## Files

- `bert_classifier.py` - Core BERT model implementation
- `relevance.py` - Relevance checking functionality
- `document_trainer.py` - Extract training data from documents
- `simple_integration.py` - Easy integration for existing chatbots

## Quick Start

### 1. Train the model

Train using PDF documents in your data folder:

```bash
python document_trainer.py
```

This will:
- Extract text from PDF documents in the `data_documents` folder
- Generate training samples from the text
- Add pre-defined questions for each category
- Train the BERT model
- Save the model to `models/bert_classifier`

### 2. Integration

The easiest way to integrate is with `simple_integration.py`:

```python
from simple_integration import check_message_relevance

# In your message handling function
def handle_message(message, user_id):
    # Check relevance
    relevance = check_message_relevance(message, user_id)
    
    if not relevance['is_relevant']:
        # Message is off-topic
        warning = relevance['warning']
        
        # Check if we should redirect after multiple off-topic messages
        if relevance['should_redirect']:
            # Redirect logic here
            return "I notice we've gone off-topic. Let me redirect you to general support."
        else:
            # Just warn the user
            return f"I'm specialized in tax matters. {warning}"
    
    # Message is relevant - process with your existing logic
    if relevance['topic'] == 'IVA':
        # IVA-specific handling
        pass
    elif relevance['topic'] == 'Fiscozen':
        # Fiscozen-specific handling
        pass
    else:
        # General tax handling
        pass
```

## Advanced Usage

For more control, use the `RelevanceChecker` class directly:

```python
from relevance import RelevanceChecker

# Initialize
checker = RelevanceChecker(model_path="models/bert_classifier")

# Check relevance
relevance = checker.check_relevance("What is IVA tax?")
```

## Categories

The model classifies messages into three categories:

1. `IVA` - Questions about IVA tax
2. `Fiscozen` - Questions about Fiscozen services
3. `Other Tax Matter` - Other tax-related questions

Messages with low confidence in all categories are flagged as off-topic.
