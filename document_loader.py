"""
document_loader.py

Responsible for taking an uploaded file, saving it temporarily to disk,
extracting its text content, and splitting that text into chunks that
are ready to be embedded.
"""

import os
import tempfile
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)

# File extensions this app knows how to handle, mapped to their loader class.
SUPPORTED_LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
}


class UnsupportedFileTypeError(Exception):
    """Raised when the uploaded file extension isn't supported."""
    pass


class EmptyDocumentError(Exception):
    """Raised when a document is loaded but contains no usable text."""
    pass


def _get_file_extension(filename: str) -> str:
    return os.path.splitext(filename)[1].lower()


def load_document(uploaded_file) -> List[Document]:
    """
    Save a Streamlit UploadedFile to a temp path and load it into
    LangChain Document objects using the appropriate loader.

    Args:
        uploaded_file: a Streamlit UploadedFile object.

    Returns:
        A list of LangChain Document objects (one or more per page/section).

    Raises:
        UnsupportedFileTypeError: if the extension isn't supported.
        EmptyDocumentError: if no text could be extracted.
    """
    extension = _get_file_extension(uploaded_file.name)

    if extension not in SUPPORTED_LOADERS:
        supported = ", ".join(SUPPORTED_LOADERS.keys())
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{extension}'. Supported types: {supported}"
        )

    # Loaders in LangChain generally expect a file path, so we write the
    # uploaded bytes to a temporary file first.
    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        loader_class = SUPPORTED_LOADERS[extension]
        loader = loader_class(tmp_path)
        documents = loader.load()
    finally:
        # Always clean up the temp file, even if loading fails.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Filter out any completely empty documents and check we have content.
    non_empty_docs = [doc for doc in documents if doc.page_content.strip()]

    if not non_empty_docs:
        raise EmptyDocumentError(
            "No readable text was found in the uploaded document. "
            "It may be empty, scanned as an image, or corrupted."
        )

    # Attach the original filename to each chunk's metadata for source tracking.
    for doc in non_empty_docs:
        doc.metadata["source_file"] = uploaded_file.name

    return non_empty_docs


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    """
    Split loaded documents into smaller overlapping chunks suitable for
    embedding and retrieval.

    Args:
        documents: list of LangChain Document objects.
        chunk_size: max characters per chunk.
        chunk_overlap: overlap between consecutive chunks, to preserve context.

    Returns:
        A list of chunked Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise EmptyDocumentError("Document could not be split into usable chunks.")

    # Add a simple chunk index to metadata, useful for displaying sources.
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks


def process_uploaded_file(
    uploaded_file, chunk_size: int = 1000, chunk_overlap: int = 150
) -> List[Document]:
    """
    Convenience function that loads and splits an uploaded file in one call.
    """
    documents = load_document(uploaded_file)
    return split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
