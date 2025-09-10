import streamlit as st
import tempfile
import os
from rag_pipeline import ingest_pdfs_and_get_retriever, answer_query, summarize_documents
from utils.pdf_loader import load_pdf_as_documents
from utils.ui_helpers import style_app, sidebar_instructions
from config import CHROMA_DIR

# --- Page Config & Styles ---
st.set_page_config(page_title="Arc – AI Research Companion", layout="wide", initial_sidebar_state="expanded")
style_app()

# --- Title ---
st.title("📚 Arc — AI Research Companion")
st.markdown("A private & powerful research assistant using RAG with HuggingFace + ChromaDB")

# --- Sidebar ---
with st.sidebar:
    sidebar_instructions()
    st.markdown("---")
    st.write(f"**Current Vector DB:** `{CHROMA_DIR}`")
    st.markdown("---")
    st.markdown("Powered by `google/flan-t5-small`")

# --- Section 1: PDF Ingestion ---
st.header("1) Upload & Ingest PDFs")
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
chunk_size = st.number_input("Chunk size (characters)", value=1000, min_value=100, max_value=3000)
chunk_overlap = st.number_input("Chunk overlap (characters)", value=150, min_value=0, max_value=1000)

if st.button("📥 Ingest Uploaded PDFs"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        with st.spinner("Processing and ingesting PDFs..."):
            try:
                # Save uploaded files to a temporary directory
                tmp_dir = tempfile.mkdtemp()
                file_paths = []
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(tmp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(file_path)

                ingested_retriever = ingest_pdfs_and_get_retriever(
                    file_paths=file_paths,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                st.session_state["retriever"] = ingested_retriever
                st.session_state["ingested_docs"] = load_pdf_as_documents(file_paths)
                st.success(f"✅ Successfully ingested {len(file_paths)} PDF(s) into ChromaDB!")

            except Exception as e:
                st.error(f"❌ Ingestion failed: {e}")
            finally:
                # Clean up temp files
                import shutil
                shutil.rmtree(tmp_dir)

# --- Section 2: Query Documents ---
st.header("2) Query Documents")
if "retriever" not in st.session_state:
    st.info("Please ingest documents first to enable querying.")
else:
    question = st.text_area("Ask a question about your documents:", placeholder="e.g., What are the key findings of the research on climate change?")
    if st.button("🔎 Get Answer"):
        if not question:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating answer..."):
                try:
                    response = answer_query(question, st.session_state["retriever"])
                    st.markdown("### 📝 Answer")
                    st.write(response["answer"])

                    st.markdown("---")
                    st.markdown("### 📑 Citations")
                    citations = response["citations"]
                    if not citations:
                        st.info("No relevant citations found.")
                    else:
                        for i, c in enumerate(citations, start=1):
                            st.markdown(f"**{i}. {c.get('source', 'N/A')} (Page {c.get('page', 'N/A')})**")
                            st.write(f"> {c.get('snippet', 'N/A')}")
                except Exception as e:
                    st.error(f"Query failed: {e}")

# --- Section 3: Additional Actions ---
st.header("3) Additional Actions")
if "ingested_docs" not in st.session_state:
    st.info("Please ingest documents first to enable summarization.")
else:
    if st.button("🧾 Summarize All Documents"):
        with st.spinner("Generating summary of ingested documents..."):
            try:
                all_docs = st.session_state["ingested_docs"]
                summary = summarize_documents(all_docs)
                st.markdown("### 📝 Summary")
                st.write(summary)
            except Exception as e:
                st.error(f"Summarization failed: {e}")

st.markdown("---")
st.caption("Arc • Academic Research with RAG • Powered by HuggingFace + ChromaDB")
