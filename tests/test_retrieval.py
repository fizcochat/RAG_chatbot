import os
import pytest
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from ingestion.config import PINECONE_ENV, INDEX, MODEL

@pytest.fixture(scope="module")
def vectorstore():
    # Load API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")

    # Init Pinecone & Index
    pc = Pinecone(api_key=pinecone_key, environment=PINECONE_ENV)
    index = pc.Index(INDEX)

    # Create LangChain vectorstore
    embed_model = OpenAIEmbeddings(model=MODEL, openai_api_key=openai_key)
    return PineconeVectorStore(index, embed_model, "text")

def test_vectorstore_retrieval(vectorstore):
    query = "What is the IVA?"
    results = vectorstore.similarity_search(query, k=5)

    assert results, "Should return at least one document"
    assert all(hasattr(doc, "page_content") for doc in results), "Each result should have content"
    
    print("\nTop match:")
    print(results[0].page_content[:300])
