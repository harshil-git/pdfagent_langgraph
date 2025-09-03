
from langchain_chroma import Chroma

from re_ranker import ranker

from langchain_huggingface import HuggingFaceEmbeddings

def query_documents(question:str,persist_directory="./chroma_db", collection_name="pdf_collection", k=5):
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    persist_directory="./chroma_db"
    # Load existing ChromaDB
    db = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_directory
    )
    searched_para = db.similarity_search(question)
    retrieved_para = [doc.page_content for doc in searched_para]
    return retrieved_para 


