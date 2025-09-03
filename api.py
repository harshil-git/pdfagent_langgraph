from fastapi import FastAPI, UploadFile
import shutil, os, json

from graph import build_graph

app = FastAPI(title="PDF RAG QA API")
agent_graph = build_graph()


@app.post("/query")
async def query(data: dict):
    question = data.get("question")
    state = {"question": question, "attempts": 0}
    result = agent_graph.invoke(state)
    return {"answer": result["answer"], "metrics": result["metrics"]}

