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

# Create necessary directories and test model
RUN mkdir -p fast_text/models \
    data_documents \
    dtaa-documents \
    argilla_data_49 \
    argilla-data

# Copy the rest of the application
COPY . .

# Ensure the model directory exists and create a test model file
RUN mkdir -p /app/fast_text/models && \
    echo "__label__IVA 1.0" > /app/fast_text/models/tax_classifier.bin

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.address", "0.0.0.0", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]