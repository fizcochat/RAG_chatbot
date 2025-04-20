import os
from dotenv import load_dotenv
import openai
from pinecone import Pinecone
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)

# Define constants
MODEL = "text-embedding-3-large"
INDEX_NAME = "fiscozen"
VECTOR_DIMENSION = 3072  # Dimension for text-embedding-3-large

# Create index if it doesn't exist
def ensure_index_exists():
    # List all indexes
    indexes = pc.list_indexes()
    index_names = [index.name for index in indexes]
    
    # Check if our index already exists
    if INDEX_NAME not in index_names:
        print(f"Index '{INDEX_NAME}' not found. Creating new index...")
        # Create a new index
        pc.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIMENSION,
            metric="cosine"
        )
        print(f"Index '{INDEX_NAME}' created successfully.")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")
    
    # Connect to the index
    return pc.Index(INDEX_NAME)

# Text preprocessing
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text

# Process PDF files
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(data)
    texts = [str(doc) for doc in documents]
    return texts

# Create embeddings
def create_embeddings(texts):
    embeddings_list = []
    for text in texts:
        response = openai.embeddings.create(input=text, model=MODEL)
        embedding = response.data[0].embedding
        embeddings_list.append(embedding)
    return embeddings_list

# Upload embeddings to Pinecone
def upsert_embeddings_to_pinecone(index, embeddings, ids, texts, batch_size=100):
    for i in range(0, len(embeddings), batch_size):
        batch_embeddings = embeddings[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_texts = texts[i:i + batch_size]
        metadata = []
        for id, text in zip(batch_ids, batch_texts):
            filename = id.split('_chunk_')[0]
            metadata.append({
                'text': preprocess_text(text),
                'source': filename
            })
        index.upsert(vectors=[(id, embedding, meta) for id, embedding, meta in zip(batch_ids, batch_embeddings, metadata)])

# Get all PDF files from both directories
def get_pdf_files(directories):
    pdf_files = []
    for directory in directories:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(directory, file))
    return pdf_files

# Main function
def upload_documents_to_pinecone():
    # Ensure index exists before trying to use it
    index = ensure_index_exists()
    
    directories = ["argilla_data_49", "data_documents"]
    pdf_files = get_pdf_files(directories)
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for file_path in pdf_files:
        print(f"Processing {file_path}...")
        texts = process_pdf(file_path)
        print(f"  Created {len(texts)} text chunks")
        
        print("  Generating embeddings...")
        embeddings = create_embeddings(texts)
        
        # Create unique IDs for each chunk
        ids = [f"{file_path}_chunk_{i}" for i in range(len(embeddings))]
        
        # Upload to Pinecone
        print("  Uploading to Pinecone...")
        upsert_embeddings_to_pinecone(index, embeddings, ids, texts)
        
        print(f"✅ Completed processing {file_path}")
    
    print("All documents processed and uploaded to Pinecone")

if __name__ == "__main__":
    upload_documents_to_pinecone()
