# Fiscozen Tax Chatbot

A specialized chatbot for Italian tax matters, with a focus on IVA regulations and Fiscozen services.

## Features

- **Tax-Specific Knowledge**: Specializes in Italian tax matters, particularly IVA regulations
- **Smart Relevance Detection**: Uses a BERT-based classifier to determine if queries are tax-related
- **Memory and Context**: Maintains conversation history for contextual responses
- **Off-Topic Handling**: Redirects users to customer support for non-tax queries
- **Responsive UI**: Built with Streamlit for a clean user interface

## System Requirements

- Python 3.8+
- API Keys for OpenAI and Pinecone
- Optional: CUDA-compatible GPU for faster inference

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/fizcochat/RAG_chatbot.git
   cd RAG_chatbot
   ```

2. Set up environment variables in a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   ```

3. Initialize the BERT model (required before first run):
   ```
   python initialize_model.py
   ```
   This script will download the base BERT model and configure it for relevance checking.

4. Run the application:
   ```
   python run_prod.py
   ```

## Project Structure

- `main.py`: Main Streamlit application
- `utils.py`: Utility functions for the RAG system
- `relevance.py`: BERT-based relevance checker for tax queries
- `initialize_model.py`: Script to initialize the BERT model
- `run_prod.py`: Production deployment script

## Model Information

The chatbot uses a BERT model for determining query relevance. This model is not included in the repository due to its size (over 400MB), but is automatically downloaded and set up when you run `initialize_model.py`.

The model classifies queries into three categories:
- IVA (Italian VAT tax)
- Fiscozen (tax service related)
- Other (non-tax related)

## Usage

The chatbot processes user queries as follows:

1. Checks if the query is relevant to tax matters
2. If relevant, retrieves information from the knowledge base
3. If not relevant after three consecutive queries, suggests contacting customer support
4. Maintains conversation context for more natural interactions

## License

Proprietary and confidential. Unauthorized copying or distribution prohibited.

## Support

For issues, contact support@fiscozen.it
