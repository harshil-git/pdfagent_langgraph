import streamlit as st
import requests
from process_pdfs import process_all_pdfs
from query_handler import query_documents

API_URL = "http://127.0.0.1:8000/" 

st.title("📄 PDF Search")

uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

if uploaded_file:
    process_all_pdfs(uploaded_file)
    st.success("File uploaded and vectors db created! Ready to query.")

question = st.text_input("Ask a question:")
if st.button("Submit"):
    if question:
        r = requests.post(f"{API_URL}/query", json={"question": question})
        res = r.json()
        st.write("### Answer:")
        st.write(res["answer"])
        st.write("### Metrics")
        st.json(res["metrics"])




