import os
import shutil
from typing import List
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.schema import Document
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from utils.pdf_loader import load_pdf_as_documents
from utils.text_splitter import chunk_document_text_small, chunk_document_text_large
from config import CHROMA_DIR

# --- HuggingFace Models ---
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_llm():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    text_gen_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512
    )

    return HuggingFacePipeline(pipeline=text_gen_pipeline)

llm = build_llm()

# --- Ingestion & Retrieval ---
def ingest_pdfs_and_get_retriever(file_paths: List[str], chunk_size: int, chunk_overlap: int):
    """
    Ingests PDFs and returns a ParentDocumentRetriever.
    First, it loads and splits documents into small and large chunks.
    Then, it creates a new ChromaDB vector store and populates it.
    """
    # Clean up previous Chroma DB to ensure a fresh start
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        print(f"✅ Cleared existing Chroma DB at {CHROMA_DIR}")

    all_docs = []
    for file_path in file_paths:
        docs = load_pdf_as_documents(file_path, chunk_size, chunk_overlap)
        if docs:
            all_docs.extend(docs)
    
    # Small chunks for retrieval, large chunks for context
    small_chunks = chunk_document_text_small(all_docs)
    large_chunks = chunk_document_text_large(all_docs)
    
    # Create the vector store
    # ⚠️ We are removing the DuckDB client settings to avoid the RuntimeError
    vectorstore = Chroma.from_documents(
        documents=small_chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR
    )
    
    # Create ParentDocumentRetriever
    store = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=chunk_document_text_small,
        parent_splitter=chunk_document_text_large
    )
    retriever.add_documents(large_chunks)
    
    return retriever

def format_citations(docs: List[Document]) -> List[dict]:
    """Formats a list of documents into a clean citation list."""
    citations = [
        {
            "source": d.metadata.get("source", "N/A"),
            "page": d.metadata.get("page", "N/A"),
            "snippet": d.page_content
        }
        for d in docs
    ]
    return citations

def _compose_prompt(question: str, contexts: List[Document]) -> str:
    """Composes a prompt for the LLM using the retrieved contexts."""
    context_text = "\n\n".join([
        f"[Source {i}] {c.metadata.get('source', 'N/A')}, p.{c.metadata.get('page', 'N/A')}:\n{c.page_content}"
        for i, c in enumerate(contexts, start=1)
    ])
    return (
        "You are Arc, an academic research assistant. Answer strictly from the provided sources. "
        "Include inline citations like (FileName.pdf, p. X). If you cannot find evidence, say so.\n\n"
        f"Question: {question}\n\nContext:\n{context_text}\nAnswer concisely with evidence."
    )

def answer_query(question: str, retriever) -> dict:
    """Answers a user query using the RAG pipeline."""
    relevant_docs = retriever.get_relevant_documents(question)
    citations = format_citations(relevant_docs)

    prompt = _compose_prompt(question, relevant_docs)
    response = llm(prompt)
    
    return {
        "answer": response["generated_text"] if isinstance(response, dict) else str(response),
        "citations": citations
    }

def summarize_documents(documents: List[Document]) -> str:
    """Summarizes a list of documents using the LLM."""
    text_combined = " ".join([d.page_content for d in documents])
    summary_prompt = f"Please provide a detailed summary of the following text:\n\n{text_combined}"
    response = llm(summary_prompt)
    return response["generated_text"] if isinstance(response, dict) else str(response)
