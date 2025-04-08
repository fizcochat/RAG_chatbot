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

# Create necessary directories for test data
RUN mkdir -p fast_text/models data_documents dtaa-documents argilla_data_49 argilla-data

# Copy the rest of the application
COPY . .

# Create a dummy model file for testing if it doesn't exist
RUN if [ ! -f fast_text/models/tax_classifier.bin ]; then \
    echo "dummy model" > fast_text/models/tax_classifier.bin; \
    fi

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.address", "0.0.0.0", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]