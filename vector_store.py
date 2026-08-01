"""
vector_store.py

Responsible for creating embeddings via the Mistral API and storing /
retrieving them from a persistent ChromaDB vector store.
"""

import os
import shutil
from typing import List, Optional

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
            "Missing MISTRAL_API_KEY. Please set it in your .env file."
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


def reset_persist_directory(persist_dir: str) -> None:
    """
    Remove any existing persisted ChromaDB data so a newly uploaded
    document fully replaces the previous knowledge base.
    """
    if os.path.exists(persist_dir):
        try:
            shutil.rmtree(persist_dir)
        except OSError as exc:
            raise VectorStoreError(
                f"Could not clear previous vector store: {exc}"
            ) from exc

    os.makedirs(persist_dir, exist_ok=True)


def build_vector_store(
    chunks: List[Document],
    embeddings: MistralAIEmbeddings,
    persist_dir: str,
    replace_existing: bool = True,
) -> Chroma:
    """
    Embed document chunks and store them in a persistent Chroma collection.
    """
    if not chunks:
        raise VectorStoreError(
            "No document chunks were provided to embed."
        )

    try:
        if replace_existing:
            reset_persist_directory(persist_dir)

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=persist_dir,
        )

        return vector_store

    except Exception as exc:
        raise VectorStoreError(
            f"Failed to build the vector store: {exc}"
        ) from exc


def load_vector_store(
    embeddings: MistralAIEmbeddings,
    persist_dir: str,
) -> Optional[Chroma]:
    """
    Load an existing persisted Chroma vector store from disk.
    """
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        return None

    try:
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
    except Exception as exc:
        raise VectorStoreError(
            f"Failed to load existing vector store: {exc}"
        ) from exc