from typing import List
from pypdf import PdfReader
from langchain.docstore.document import Document
from .text_splitter import chunk_document_text_small
import os

def load_pdf_as_documents(file_paths: List[str], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """
    Loads one or more PDFs and returns a list of langchain Documents.
    Each document has metadata: {'source': source_name, 'page': page_num}.
    """
    all_docs = []
    for file_path in file_paths:
        source_name = os.path.basename(file_path)
        reader = PdfReader(file_path)
        docs = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            
            # Use the small chunking strategy for the base documents
            chunks = chunk_document_text_small(text, chunk_size, chunk_overlap)
            
            for chunk in chunks:
                metadata = {"source": source_name, "page": i}
                docs.append(Document(page_content=chunk, metadata=metadata))
        
        all_docs.extend(docs)
        
    return all_docs
