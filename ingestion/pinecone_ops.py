import openai
import os
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from pdf_utils import preprocess_text
from config import MODEL, INDEX

openai.api_key = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX)

embed_model = OpenAIEmbeddings(model=MODEL, openai_api_key=openai.api_key)
vectorstore = PineconeVectorStore(index, embed_model, "text")

def create_embeddings(texts):
    embeddings_list = []
    for text in texts:
        response = openai.embeddings.create(input=text, model=MODEL)
        embedding = response.data[0].embedding  # Access the embedding correctly
        embeddings_list.append(embedding)
    return embeddings_list

# Define a function to upsert embeddings to Pinecone with metadata
def upsert_embeddings_to_pinecone(index, embeddings, ids, texts, file_hash, batch_size=100):
    for i in range(0, len(embeddings), batch_size):
        batch_embeddings = embeddings[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_texts = texts[i:i + batch_size]
        metadata = []
        for id, text in zip(batch_ids, batch_texts):
            # Extract filename from id (removing _chunk_X)
            filename = id.split('_chunk_')[0]
            name_parts = filename.split('.')[0].split(' ')
            plan_type = name_parts[0] if name_parts else ''
            plan_difficulty = name_parts[1] if len(name_parts) > 1 else ''
            metadata.append({
                'text': preprocess_text(text),
                'plan_type': plan_type,
                'plan_difficulty': plan_difficulty,
                'hash': file_hash
            })
        index.upsert(vectors=[(id, embedding, meta) for id, embedding, meta in zip(batch_ids, batch_embeddings, metadata)])

def delete_existing_chunks(index, ids):
    index.delete(ids=ids)

def get_all_vector_ids(index):
    stats = index.describe_index_stats()
    vector_ids = []
    for ns in stats.get("namespaces", {}).values():
        vector_ids.extend(ns.get("vector_count", 0))
    return vector_ids
