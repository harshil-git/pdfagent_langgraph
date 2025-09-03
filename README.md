# Langgraph PDF search agent

## Overview
- When you upload a pdf file it creates vector embeddings out of its content and store it in chromadb.
- When user hits query it goes to retrieve node in graph and retrieves relevant documents.
- Once relevant documents are retrieved it goes to rerank node in graph to rerank them.
- after reranking them it goes to generate node to generate answer.
- after generating answer it checks if answer is grounded in given documents.
- if its grounded then presents answer to user and reach end node. otherwise redirects to retrieve node again.

# Graph

<img width="156" height="531" alt="graph_1" src="https://github.com/user-attachments/assets/3da4ccfc-58d9-4648-bf28-babb7be015aa" />


# Demo

https://github.com/user-attachments/assets/5f828b2a-f57e-45a1-b9da-2a7ad864af77

# Technologies Used
- Langgraph
- Langchain
- gemini-2.0-flash (Gemini API)
- RAG
- Chromadb
