import json
from typing import Any, Literal, TypedDict

import bm25s
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain.messages import SystemMessage
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS, DistanceStrategy
from langgraph.graph import END, START, MessagesState, StateGraph
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


class StateSchema(MessagesState):
    documents: list[Document]
    grade: str
    current_query: str


class GradeOutput(TypedDict):
    binary_score: str


llm = init_chat_model(
    "qwen3.5:9b",
    model_provider="ollama",
    base_url="http://192.168.88.179:11434",
    temperature=0,
)

embeddings = init_embeddings(
    "qwen3-embedding:8b",
    provider="ollama",
    base_url="http://192.168.88.179:11434",
)

reranker = HuggingFaceCrossEncoder(
    model_name="BAAI/bge-reranker-v2-m3",
    model_kwargs={"device": "cpu"},
)


def _vector_search(query: str) -> list[Document]:
    faiss_retriever = FAISS.load_local(
        "faiss_index",
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
        relevance_score_fn=None,
        distance_strategy=DistanceStrategy.EUCLIDEAN,
    ).as_retriever(k=10)

    docs = faiss_retriever.invoke(query)

    return docs


def _keyword_search(query: str) -> list[Document]:
    index_dir = "bm25_index"
    bm25_retriever = bm25s.BM25.load(index_dir, load_corpus=True, mmap=True)
    query_tokens = bm25s.tokenize(query, return_ids=False, stopwords="en")
    results = bm25_retriever.retrieve(query_tokens, k=10, return_as="documents")

    with open(f"{index_dir}/id_map.json") as f:
        id_map: list[dict[str, Any]] = json.load(f)

    docs: list[Document] = []
    for i in range(results.shape[1]):
        docs.append(
            Document(
                id=id_map[i]["id"],
                metadata=id_map[i]["metadata"],
                page_content=results[0][i]["text"],
            )
        )

    return docs


def route_query(state: StateSchema) -> StateSchema:
    current_query = (
        state["current_query"]
        if state.get("current_query")
        else state["messages"][-1].content
    )
    return StateSchema(current_query=current_query)


def hybrid_search(state: StateSchema) -> StateSchema:
    query = state["current_query"]
    vector_docs = _vector_search(query)
    keyword_docs = _keyword_search(query)
    docs = vector_docs + keyword_docs
    docs = list({doc.id: doc for doc in docs}.values())

    return StateSchema(documents=docs)


def rerank_documents(state: StateSchema) -> StateSchema:
    query = state["current_query"]
    docs = state["documents"]
    text_pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.score(text_pairs)
    reranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in reranked[:5]]

    return StateSchema(documents=top_docs)


def grade_documents(state: StateSchema) -> StateSchema:
    prompt = f"""You are a strict relevance grader evaluating whether a retrieved document is relevant to a user's question

**Retrieved document:**
{"\n\n".join([f"{doc.page_content}" for doc in state["documents"]])}

**User's question:**
{state["current_query"]}

**Instructions:**
- Carefully analyze the user question and infer its underlying semantic intent, even if it is implicit
- Compare the document against the question based on both keyword overlap and semantic meaning
- Consider the document relevant if it contains information that directly answers, partially answers, or clearly supports the intent of the question
- Do NOT require exact keyword matches; semantic similarity is sufficient
- Mark the document as NOT relevant if it is unrelated, too vague, or does not help answer the question
- Return a binary score:
  - yes: if the document is relevant
  - no: if the document is not relevant
- You MUST strictly follow the provided JSON schema
- Do NOT include any explanation or additional text""".strip()

    llm_with_structured = llm.with_structured_output(GradeOutput)
    grade_output = llm_with_structured.invoke(prompt)

    return StateSchema(grade=grade_output["binary_score"])


def docs_condition(state: StateSchema) -> Literal["generate_answer", "rewrite_query"]:
    return "generate_answer" if state["grade"] == "yes" else "rewrite_query"


def rewrite_query(state: StateSchema) -> StateSchema:
    prompt = f"""You are a semantic query rewriting assistant, your task is to improve a user's question by understanding its deeper intent

**User's question:**
{state["current_query"]}

**Instructions:**
- Examine the original question
- Infer hidden or implicit meaning behind the wording
- Resolve ambiguity if possible
- Rewrite the question so that it is clear, specific, and suitable for information retrieval tasks
- Do NOT change the original intent
- Do NOT add unrelated information
- Only the rewritten question""".strip()

    ai_response = llm.invoke(prompt)

    return StateSchema(current_query=ai_response.content)


def generate_answer(state: StateSchema) -> StateSchema:
    system_prompt = f"""You are a strict, citation-focused assistant for a private knowledge base

**Context:**
{"\n\n".join([f"{doc.page_content}" for doc in state["documents"]])}

**Instructions:**
- Use ONLY the provided context to answer the question
- Do NOT use any external knowledge, assumptions, or web information
- If the answer is not clearly supported by the context, respond with: "I don't know based on the provided documents"
- When answering, include citations using available metadata (e.g., file name, reference location)
- Ensure the answer is concise, accurate, and directly aligned with the context
- Do not fabricate or infer information beyond what is explicitly supported""".strip()

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    ai_response = llm.invoke(messages)

    return StateSchema(messages=[ai_response], current_query=None)


# Define the graph
graph = (
    StateGraph(StateSchema)
    # define nodes
    .add_node("route_query", route_query)
    .add_node("hybrid_search", hybrid_search)
    .add_node("rerank_documents", rerank_documents)
    .add_node("grade_documents", grade_documents)
    .add_node("rewrite_query", rewrite_query)
    .add_node("generate_answer", generate_answer)
    # define workflow
    .add_edge(START, "route_query")
    .add_edge("route_query", "hybrid_search")
    .add_edge("hybrid_search", "rerank_documents")
    .add_edge("rerank_documents", "grade_documents")
    .add_conditional_edges("grade_documents", docs_condition)
    .add_edge("rewrite_query", "route_query")
    .add_edge("generate_answer", END)
    # compile the graph
    .compile()
)
