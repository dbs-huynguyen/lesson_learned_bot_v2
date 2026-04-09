import json
from pathlib import Path
from typing import TypedDict

import faiss
import bm25s
import Stemmer
from langchain.embeddings import init_embeddings
from langgraph.graph import END, START, StateGraph
from langchain_community.vectorstores import FAISS, DistanceStrategy
from langchain_community.docstore.in_memory import InMemoryDocstore

from src.lib.parser import LessonsLearnedParser


class StateSchema(TypedDict):
    texts: list[str]
    metadata: list[dict]
    ids: list[str]


embeddings = init_embeddings(
    "qwen3-embedding:8b",
    provider="ollama",
    base_url="http://192.168.88.179:11434",
)


vectorstore = FAISS(
    embedding_function=embeddings,
    index=faiss.IndexFlatL2(len(embeddings.embed_query("hello world"))),
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
    relevance_score_fn=None,
    normalize_L2=False,
    distance_strategy=DistanceStrategy.EUCLIDEAN,
)

bm25_retriever = bm25s.BM25()

parser = LessonsLearnedParser()


def load_documents(state: StateSchema) -> StateSchema:
    file_paths: list[Path] = []
    documents = {"texts": [], "metadata": [], "ids": []}
    for file_path in sorted(Path("data").glob("BM.10.2.01.BISO - Bao cao HDKP va BHKN*.docx")):
        file_paths.append(file_path)
    for doc in parser.parser(file_paths):
        documents["texts"].append(doc["text"])
        documents["metadata"].append(doc["metadata"])
        documents["ids"].append(doc["id"])

    return StateSchema(**documents)


def embed(state: StateSchema) -> StateSchema:
    vectorstore.add_texts(state["texts"], state["metadata"], state["ids"])
    corpus_tokens = bm25s.tokenize(state["texts"], stopwords="en")
    bm25_retriever.index(corpus_tokens)
    return state


def store(state: StateSchema) -> StateSchema:
    vectorstore.save_local("faiss_index")

    bm25_retriever.save("bm25_index")
    with open("bm25_index/id_map.json", "w") as f:
        docs = []
        for id, metadata in zip(state["ids"], state["metadata"]):
            docs.append({"id": id, "metadata": metadata})
        json.dump(docs, f)

    return state


# Define the graph
graph = (
    StateGraph(StateSchema)
    # define nodes
    .add_node("load_documents", load_documents)
    .add_node("embed", embed)
    .add_node("store", store)
    # define workflow
    .add_edge(START, "load_documents")
    .add_edge("load_documents", "embed")
    .add_edge("embed", "store")
    .add_edge("store", END)
    # compile the graph
    .compile()
)
