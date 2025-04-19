# utils.py
"""
Utility helpers for service initialization and common RAG helpers.
The OpenAI embedding model has been replaced with a local, licence‑free
Sentence‑Transformers model so no additional API key is required.
"""

from __future__ import annotations

import os
import streamlit as st
import anthropic

from pinecone import Pinecone as PineconeClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone

# Constants provided by your ingestion/config.py module
from ingestion.config import INDEX, PINECONE_ENV


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def initialize_services(anthropic_api_key: str, pinecone_api_key: str):
    """Initialise the embedding model, Pinecone vector store and Anthropic client.

    Parameters
    ----------
    anthropic_api_key : str
        API key for Anthropic models.
    pinecone_api_key : str
        API key for Pinecone.

    Returns
    -------
    tuple
        (vectorstore, anthropic_client)
    """

    # 1. Embeddings ---------------------------------------------------------
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # 2. Pinecone client & vector store ------------------------------------
    pc = PineconeClient(api_key=pinecone_api_key, environment=PINECONE_ENV)
    index = pc.Index(INDEX)

    vectorstore = Pinecone.from_existing_index(
        index_name=INDEX,
        embedding=embeddings,
        text_key="text",
    )

    # 3. Anthropic client ---------------------------------------------------
    anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)

    return vectorstore, anthropic_client


def find_match(vectorstore, query: str) -> str:
    """Return the top‑k documents that are semantically similar to *query*."""
    result = vectorstore.similarity_search(query, k=5)
    return str(result)


def query_refiner(client: anthropic.Anthropic, conversation: str, query: str) -> str:
    """Rewrite *query* taking the last exchange into account for a better search."""
    # Keep only the last two exchanges (4 lines)
    conversation_lines = conversation.split("\n")[-4:]
    shortened_conversation = "\n".join(conversation_lines)

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=256,
        temperature=0.7,
        messages=[
            {
                "role": "user",
                "content": (
                    "Given the following user query and conversation log, formulate a question "
                    "that would be the most relevant to provide the user with an answer from a knowledge base.\n\n"  # noqa: E501
                    f"CONVERSATION LOG:\n{shortened_conversation}\n\n"
                    f"Query: {query}\n\nRefined Query:"
                ),
            }
        ],
    )
    return response.content[0].text


def get_conversation_string() -> str:
    """Convert the Streamlit conversation stored in session_state into plain text."""
    conversation_string = ""
    for i in range(len(st.session_state["responses"]) - 1):
        conversation_string += f"Human: {st.session_state['requests'][i]}\n"
        conversation_string += f"Bot: {st.session_state['responses'][i + 1]}\n"
    return conversation_string
