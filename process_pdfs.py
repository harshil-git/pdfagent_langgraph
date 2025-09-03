
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PDF_STORAGE_PATH = '/Users/harshilgohil/python/Langgraph/Agentic_RAG/data'



def save_uploaded_file(uploaded_file):
    file_path = os.path.join(PDF_STORAGE_PATH, uploaded_file.name)
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdfs_from_directory(pdf_dir):
    """
    Loads and splits PDF documents from a local directory.
    """
    all_docs = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            path = os.path.join(pdf_dir, filename)
            loader = PyPDFLoader(path)
            docs = loader.load()
            all_docs.extend(docs)
    return all_docs

def create_chroma_db(docs, persist_directory="./chroma_db", collection_name="pdf_collection"):
    """
    Loads or creates a Chroma collection and adds new documents.
    """
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(persist_directory):
        # Load existing DB
        vectordb = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=persist_directory
        )
        print(f"Loaded existing ChromaDB from {persist_directory}")
    else:
        # Create new DB
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embedding_function,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        

def process_all_pdfs(uploaded_file):
    save_uploaded_file(uploaded_file)
    pdf_docs = load_pdfs_from_directory(PDF_STORAGE_PATH)
    create_chroma_db(pdf_docs)

