FROM python:3.11-slim

WORKDIR /app

# Install system dependencies and cleanup in one layer to reduce image size
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir streamlit pytest

# Copy the rest of the application
COPY . .

# Create a non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Expose the port Streamlit runs on
EXPOSE 8501

# Health check to ensure the application is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501

# Command to run the application
CMD ["streamlit", "run", "main.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501", \
     "--server.enableCORS", "false", \
     "--server.enableXsrfProtection", "false"]