"""
vector_store.py

Responsible for creating embeddings via the Mistral API and storing /
retrieving them from an in-memory ChromaDB vector store.
"""

from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings

COLLECTION_NAME = "rag_documents"


class VectorStoreError(Exception):
    """Raised when something goes wrong creating or querying the vector store."""
    pass


def get_embeddings_model(
    api_key: str,
    model_name: str = "mistral-embed",
) -> MistralAIEmbeddings:
    """
    Build the Mistral embeddings client.

    Raises:
        VectorStoreError: if the API key is missing or the client can't be created.
    """
    if not api_key:
        raise VectorStoreError(
            "Missing MISTRAL_API_KEY. Please set it in your .env file or Streamlit Secrets."
        )

    try:
        return MistralAIEmbeddings(
            model=model_name,
            api_key=api_key,
        )
    except Exception as exc:
        raise VectorStoreError(
            f"Failed to initialize Mistral embeddings: {exc}"
        ) from exc


def build_vector_store(
    chunks: List[Document],
    embeddings: MistralAIEmbeddings,
    persist_dir: str = "",
    replace_existing: bool = True,
) -> Chroma:
    """
    Embed document chunks and create an in-memory Chroma vector store.

    Note:
        This version is intended for deployment on Streamlit Community Cloud.
        The vector store exists only for the current user session.
    """
    if not chunks:
        raise VectorStoreError(
            "No document chunks were provided to embed."
        )

    try:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
        )

        return vector_store

    except Exception as exc:
        raise VectorStoreError(
            f"Failed to build the vector store: {exc}"
        ) from exc


def load_vector_store(
    embeddings: MistralAIEmbeddings,
    persist_dir: str = "",
):
    """
    In-memory mode does not support loading an existing vector store.
    """
    return None