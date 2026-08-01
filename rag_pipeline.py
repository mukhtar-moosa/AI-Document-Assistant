"""
rag_pipeline.py

Ties retrieval (ChromaDB) together with generation (Mistral chat model)
to answer questions strictly from the uploaded document's content.
"""

from typing import Dict, List, Optional

from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

NOT_FOUND_MESSAGE = "I could not find this information in the uploaded document."

# The system prompt is the main defense against hallucination: it forces
# the model to answer only from the given context and to explicitly say
# when the context doesn't contain the answer.
SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the
context provided below, which was retrieved from a document the user uploaded.

Rules you must follow strictly:
1. Only use information found in the context to answer.
2. If the context does not contain enough information to answer the question,
   respond with exactly: "{not_found_message}"
3. Do not use outside knowledge, do not guess, and do not make up information.
4. Keep your answer clear and concise, and quote or paraphrase the document
   where helpful.

Context:
{context}
"""


class RAGPipelineError(Exception):
    """Raised when the RAG pipeline fails to retrieve or generate an answer."""
    pass


def get_chat_model(api_key: str, model_name: str = "mistral-small-latest") -> ChatMistralAI:
    """
    Build the Mistral chat model client.

    Raises:
        RAGPipelineError: if the API key is missing or the client can't be created.
    """
    if not api_key:
        raise RAGPipelineError("Missing MISTRAL_API_KEY. Please set it in your .env file.")

    try:
        return ChatMistralAI(model=model_name, api_key=api_key, temperature=0.1)
    except Exception as exc:
        raise RAGPipelineError(f"Failed to initialize Mistral chat model: {exc}") from exc


def retrieve_relevant_chunks(
    vector_store: Chroma, question: str, k: int = 4
) -> List:
    """
    Retrieve the top-k most relevant chunks for a question from ChromaDB.
    """
    if not question or not question.strip():
        raise RAGPipelineError("Please enter a non-empty question.")

    try:
        return vector_store.similarity_search(question, k=k)
    except Exception as exc:
        raise RAGPipelineError(f"Failed to search the document: {exc}") from exc


def build_context_text(chunks: List) -> str:
    """Concatenate retrieved chunks into a single context string for the prompt."""
    return "\n\n---\n\n".join(chunk.page_content for chunk in chunks)


def generate_answer(
    chat_model: ChatMistralAI,
    question: str,
    context_chunks: List,
) -> str:
    """
    Send the retrieved context and question to the Mistral model and
    return a grounded answer.
    """
    if not context_chunks:
        return NOT_FOUND_MESSAGE

    context_text = build_context_text(context_chunks)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )

    chain = prompt | chat_model | StrOutputParser()

    try:
        answer = chain.invoke(
            {
                "context": context_text,
                "question": question,
                "not_found_message": NOT_FOUND_MESSAGE,
            }
        )
        return answer.strip()
    except Exception as exc:
        raise RAGPipelineError(f"Mistral API error while generating an answer: {exc}") from exc


def answer_question(
    vector_store: Chroma,
    chat_model: ChatMistralAI,
    question: str,
    k: int = 4,
) -> Dict:
    """
    Full RAG call: retrieve relevant chunks, then generate a grounded answer.

    Returns:
        A dict with keys "answer" and "sources" (list of source chunk info).
    """
    chunks = retrieve_relevant_chunks(vector_store, question, k=k)
    answer = generate_answer(chat_model, question, chunks)

    sources = [
        {
            "source_file": chunk.metadata.get("source_file", "unknown"),
            "chunk_index": chunk.metadata.get("chunk_index", "?"),
            "page": chunk.metadata.get("page", None),
            "preview": chunk.page_content[:200].strip() + (
                "..." if len(chunk.page_content) > 200 else ""
            ),
        }
        for chunk in chunks
    ]

    return {"answer": answer, "sources": sources}
