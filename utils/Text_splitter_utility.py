from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Child/Small chunker for retrieval
def chunk_document_text_small(text: str or List[Document], chunk_size: int=250, chunk_overlap: int=50) -> List[Document]:
    """
    Splits text or a list of documents into small, overlapping chunks for optimized retrieval.
    This is used for the "child" documents in the ParentDocumentRetriever.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if isinstance(text, str):
        return splitter.split_text(text)
    elif isinstance(text, list) and all(isinstance(d, Document) for d in text):
        return splitter.split_documents(text)
    return []

# Parent/Large chunker for context
def chunk_document_text_large(text: str or List[Document], chunk_size: int=1000, chunk_overlap: int=150) -> List[Document]:
    """
    Splits text or a list of documents into larger chunks for providing full context.
    This is used for the "parent" documents in the ParentDocumentRetriever.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if isinstance(text, str):
        return splitter.split_text(text)
    elif isinstance(text, list) and all(isinstance(d, Document) for d in text):
        return splitter.split_documents(text)
    return []
