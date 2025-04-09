FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    gcc \
    g++ \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir streamlit

# Create necessary directories
RUN mkdir -p fast_text/models \
    data_documents \
    dtaa-documents \
    argilla_data_49 \
    argilla-data

# Copy the rest of the application
COPY . .

# Create a proper FastText model file
RUN echo "__label__IVA Come funziona l'IVA?" > fast_text/models/tax_classifier.txt && \
    echo "__label__Other Che tempo fa?" >> fast_text/models/tax_classifier.txt && \
    python -c "import fasttext; model = fasttext.train_supervised('fast_text/models/tax_classifier.txt'); model.save_model('fast_text/models/tax_classifier.bin')"

# Debug: Show directory structure and model file
RUN echo "Current directory:" && pwd && \
    echo "Directory contents:" && ls -la && \
    echo "fast_text directory contents:" && ls -la fast_text && \
    echo "fast_text/models directory contents:" && ls -la fast_text/models

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.address", "0.0.0.0", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]