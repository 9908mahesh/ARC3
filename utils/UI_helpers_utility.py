import streamlit as st

def style_app():
    """Applies custom CSS for a cleaner, branded look."""
    st.markdown(
        """
        <style>
        .reportview-container .main .block-container{padding:1rem 3rem;}
        .stButton>button {
            background-color: #0b5fff;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 10px 24px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #004dcc;
        }
        .stButton>button:active {
            background-color: #003399;
        }
        .stDownloadButton>button {
            background-color: #33aaff;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 10px 24px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .stDownloadButton>button:hover {
            background-color: #298cff;
        }
        </style>
        """, unsafe_allow_html=True
    )

def sidebar_instructions():
    """Renders the instructions in the Streamlit sidebar."""
    st.sidebar.header("How to use Arc")
    st.sidebar.markdown(
        """
        1.  **Upload PDFs:** Select one or more PDF files from your computer.
        2.  **Ingest:** Click `Ingest Uploaded PDFs` to process the documents. This will create a local vector database.
        3.  **Query:** Ask a question about the content of your PDFs and get a concise answer with citations.
        4.  **Summarize:** Generate a summary of all the ingested documents.
        """
    )
