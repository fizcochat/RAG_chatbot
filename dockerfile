# Use an official Python runtime as a base
FROM python:3.12

# Set the working directory inside the container
WORKDIR /app

# Copy all files from your local project into the container
COPY . .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit will run on
EXPOSE 8501

# Command to run your Streamlit app
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
