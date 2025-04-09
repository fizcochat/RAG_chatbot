FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/fast_text/models \
    /app/data_documents \
    /app/dtaa-documents \
    /app/argilla_data_49 \
    /app/argilla-data

# Create training data and train model
COPY create_model.py .
RUN python create_model.py

# Set permissions
RUN chmod -R 755 /app/fast_text/models

# Copy application code
COPY . .

# Verify model file exists and has correct permissions
RUN ls -la /app/fast_text/models/tax_classifier.bin

# Set environment variables
ENV PYTHONPATH=/app

# Run tests
CMD ["pytest", "tests/"]