import os
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from config import PINECONE_ENV, INDEX

# Init Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"], environment=PINECONE_ENV)
index = pc.Index(INDEX)

# Create vectorstore (must match same embedding model used in ingestion)
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=os.environ["OPENAI_API_KEY"])
vectorstore = PineconeVectorStore(index, embedding_model, "text")

# Test query
query = "What is the IVA?"
results = vectorstore.similarity_search(query, k=5)

# Print results
for i, doc in enumerate(results):
    print(f"🔹 Result {i+1}")
    print(f"Content: {doc.page_content[:300]}...")
    print(f"Metadata: {doc.metadata}")
    print("-" * 50)
