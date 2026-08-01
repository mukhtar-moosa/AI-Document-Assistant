"""
app.py

Streamlit front-end for the AI Document Assistant (RAG application).
Handles file upload, triggers document processing, and provides a
chat-style Q&A interface backed by the Mistral API + ChromaDB.
"""

import os
import streamlit as st
from dotenv import load_dotenv

from document_loader import (
    process_uploaded_file,
    UnsupportedFileTypeError,
    EmptyDocumentError,
)
from vector_store import (
    get_embeddings_model,
    build_vector_store,
    VectorStoreError,
)
from rag_pipeline import (
    get_chat_model,
    answer_question,
    RAGPipelineError,
)

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
CHAT_MODEL_NAME = os.getenv("MISTRAL_CHAT_MODEL", "mistral-small-latest")
EMBED_MODEL_NAME = os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed")
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

SUPPORTED_TYPES = ["pdf", "txt", "docx"]

st.set_page_config(page_title="AI Document Assistant", page_icon="📚", layout="centered")


def init_session_state():
    defaults = {
        "vector_store": None,
        "chat_model": None,
        "document_name": None,
        "chat_history": [],  # list of {"question": ..., "answer": ..., "sources": [...]}
        "processing_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def process_document(uploaded_file):
    """Load, chunk, embed, and store the uploaded document. Updates session state."""
    st.session_state.processing_error = None

    if not MISTRAL_API_KEY:
        st.session_state.processing_error = (
            "Missing MISTRAL_API_KEY. Please add it to your .env file and restart the app."
        )
        return

    try:
        with st.spinner("Reading and splitting document..."):
            chunks = process_uploaded_file(uploaded_file)

        with st.spinner("Generating embeddings and storing in ChromaDB..."):
            embeddings = get_embeddings_model(MISTRAL_API_KEY, EMBED_MODEL_NAME)
            vector_store = build_vector_store(
                chunks, embeddings, PERSIST_DIR, replace_existing=True
            )

        with st.spinner("Connecting to Mistral..."):
            chat_model = get_chat_model(MISTRAL_API_KEY, CHAT_MODEL_NAME)

        # Success: replace previous knowledge base entirely.
        st.session_state.vector_store = vector_store
        st.session_state.chat_model = chat_model
        st.session_state.document_name = uploaded_file.name
        st.session_state.chat_history = []

    except UnsupportedFileTypeError as exc:
        st.session_state.processing_error = str(exc)
    except EmptyDocumentError as exc:
        st.session_state.processing_error = str(exc)
    except VectorStoreError as exc:
        st.session_state.processing_error = f"Vector store error: {exc}"
    except RAGPipelineError as exc:
        st.session_state.processing_error = f"Mistral API error: {exc}"
    except Exception as exc:  # noqa: BLE001 - final safety net for unexpected errors
        st.session_state.processing_error = f"Unexpected error while processing document: {exc}"


def render_sidebar():
    with st.sidebar:
        st.header("About")
        st.write(
            "This app lets you upload a document and ask questions about it. "
            "Answers are generated using Retrieval-Augmented Generation (RAG) "
            "with Mistral AI and ChromaDB, grounded strictly in your document's content."
        )
        st.divider()
        if st.session_state.document_name:
            st.success(f"Active document: **{st.session_state.document_name}**")
        else:
            st.info("No document loaded yet.")

        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


def render_upload_section():
    st.subheader("1. Upload your document")
    uploaded_file = st.file_uploader(
        "Choose a file (PDF, TXT, or DOCX)", type=SUPPORTED_TYPES
    )

    process_clicked = st.button("📥 Process Document", type="primary", disabled=uploaded_file is None)

    if process_clicked:
        if uploaded_file is None:
            st.warning("Please choose a file before processing.")
        else:
            process_document(uploaded_file)

    if st.session_state.processing_error:
        st.error(st.session_state.processing_error)
    elif st.session_state.document_name and process_clicked:
        st.success("Document processed successfully ✅")


def render_qa_section():
    st.subheader("2. Ask a question about your document")

    if st.session_state.vector_store is None:
        st.info("Upload and process a document above to start asking questions.")
        return

    with st.form(key="question_form", clear_on_submit=True):
        question = st.text_input("Your question", placeholder="What is this document about?")
        ask_clicked = st.form_submit_button("💬 Ask")

    if ask_clicked:
        if not question or not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Thinking..."):
                    result = answer_question(
                        st.session_state.vector_store,
                        st.session_state.chat_model,
                        question,
                    )
                st.session_state.chat_history.append(
                    {
                        "question": question,
                        "answer": result["answer"],
                        "sources": result["sources"],
                    }
                )
            except RAGPipelineError as exc:
                st.error(f"Error generating answer: {exc}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Unexpected error: {exc}")

    render_chat_history()


def render_chat_history():
    if not st.session_state.chat_history:
        return

    st.subheader("Conversation")
    # Show most recent exchange first.
    for exchange in reversed(st.session_state.chat_history):
        st.markdown(f"**🧑 You:** {exchange['question']}")
        st.markdown("**🤖 Answer:**")
        st.write(exchange["answer"])

        if exchange["sources"]:
            with st.expander("📄 Sources used for this answer"):
                for i, source in enumerate(exchange["sources"], start=1):
                    page_info = f", page {source['page'] + 1}" if source.get("page") is not None else ""
                    st.markdown(
                        f"**{i}. {source['source_file']}** "
                        f"(chunk {source['chunk_index']}{page_info})"
                    )
                    st.caption(source["preview"])
        st.divider()


def main():
    init_session_state()

    st.title("📚 AI Document Assistant")
    st.write(
        "Upload a document and ask questions about it. Answers are generated "
        "only from the content of your document."
    )

    render_sidebar()
    render_upload_section()
    st.divider()
    render_qa_section()


if __name__ == "__main__":
    main()
