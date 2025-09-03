from langgraph.graph import StateGraph, END
from langchain.schema import Document
from model import llm
from query_handler import query_documents
from re_ranker import ranker


# State definition
class QAState(dict):
    question: str
    documents: list
    answer: str
    metrics: dict
    attempts: int


# Nodes
def retrieve_node(state: QAState):
    docs = query_documents(state["question"],persist_directory="./chroma_db", collection_name="pdf_collection",k=5)
    return {**state, "documents": docs}


def rerank_node(state: QAState):
    docs = ranker(state["question"], state["documents"])
    return {**state, "documents": docs}


def generate_node(state: QAState):
    context = "\n\n".join([f"{d}" for d in state["documents"]])
    prompt = f"Think step by step and Answer strictly from the following documents:\n{context}\n\nQ: {state['question']}\nA:"
    response = llm.invoke(prompt)
    return {**state, "answer": response.content}


def self_check_node(state: QAState):
    judge_prompt = f"Does the answer below stay grounded in the provided documents? Answer YES/NO. \n\n Documents: {state['documents']} \n\nAnswer: {state['answer']}"
    judge = llm.invoke(judge_prompt)
    grounded = "YES" in judge.content.upper()
    return {**state, "metrics": {"grounded": grounded}, "attempts": state.get("attempts", 0) + 1}


# Graph build
def build_graph():
    g = StateGraph(QAState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("rerank", rerank_node)
    g.add_node("generate", generate_node)
    g.add_node("self_check", self_check_node)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "generate")
    g.add_edge("generate", "self_check")

    def conditional(state: QAState):
        if not state["metrics"]["grounded"] and state["attempts"] < 2:
            return "retrieve"
        return END

    g.add_conditional_edges("self_check", conditional, {"retrieve": "retrieve", END: END})
    return g.compile()
