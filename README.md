# Fiscozen Tax Chatbot

A production-ready chatbot for Fiscozen that specializes in Italian tax matters, IVA regulations, and Fiscozen services.

## Overview

This chatbot leverages RAG (Retrieval-Augmented Generation) to provide accurate responses to user queries about Italian taxes, with a specific focus on IVA (Italian VAT) and Fiscozen services. The system includes a BERT-based relevance detection mechanism that filters out off-topic queries.

## Features

- **Tax-Specific Knowledge**: Specialized in Italian tax regulations, IVA, and Fiscozen services
- **Smart Relevance Detection**: BERT-based classifier detects and filters off-topic queries
- **Memory and Context**: Maintains conversation context to provide coherent responses
- **Off-Topic Handling**: Redirects users to customer support for non-tax related inquiries
- **Responsive UI**: Built with Streamlit for a clean, modern interface

## System Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for faster inference)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd fiscozen-chatbot
   ```

2. **Set up environment variables:**
   Create a `.env` file with the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   ```

3. **Deploy in production:**
   ```bash
   python run_prod.py
   ```

## Project Structure

- `main.py`: The main Streamlit application
- `utils.py`: Utility functions for vector search and query refinement
- `relevance.py`: BERT-based relevance detection for tax-related queries
- `run_prod.py`: Production deployment script

## Usage

Once deployed, the chatbot will:

1. Check if user queries are relevant to tax matters
2. Provide tax-specific responses for relevant queries
3. Redirect off-topic conversations to customer support
4. Add relevant context based on the specific tax topic detected

## License

This project is proprietary and confidential.

## Contact

For support, please contact [Fiscozen Support](mailto:support@fiscozen.it).
