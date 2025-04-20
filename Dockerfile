FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Stage for dependency installation
FROM base AS dependencies

# Copy requirements file first to leverage Docker cache
COPY requirements.txt ./

# Install Python dependencies with better error handling
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir streamlit && \
    python -c "import platform; print(f'Installing dependencies for {platform.system()}')" && \
    # Attempt to install fasttext or continue on failure
    pip install --no-cache-dir fasttext || echo "FastText installation failed, will use fallback mode" && \
    pip install --no-cache-dir -r requirements.txt

# Final stage for application
FROM dependencies AS application

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p fast_text/models data_documents argilla_data_49 monitor/logs

# Create dummy model file for testing if it doesn't exist
RUN if [ ! -f fast_text/models/tax_classifier.bin ]; then \
        echo "Creating dummy model file for testing..." && \
        echo "dummy model" > fast_text/models/tax_classifier.bin; \
    fi

# Set environment variables
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Expose the Streamlit port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
