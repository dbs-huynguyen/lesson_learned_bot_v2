import argparse
import json
import shutil
from pathlib import Path

import faiss
import bm25s
from langchain.embeddings import init_embeddings
from langchain_community.vectorstores import FAISS, DistanceStrategy
from langchain_community.docstore.in_memory import InMemoryDocstore

from src.lib.parser import LessonsLearnedParser


BASE_URL = "http://192.168.88.179:11434"
FAISS_INDEX_DIR = "faiss_index"
BM25_INDEX_DIR = "bm25_index"
FILE_GLOB = "BM.10.2.01.BISO - Bao cao HDKP va BHKN*.docx"

embeddings = init_embeddings("qwen3-embedding:8b", provider="ollama", base_url=BASE_URL)

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


def load_documents(input_dir: Path) -> dict[str, list]:
    file_paths: list[Path] = []
    documents = {"texts": [], "metadata": [], "ids": []}
    for file_path in sorted(input_dir.glob(FILE_GLOB)):
        file_paths.append(file_path)
    for doc in parser.parser(file_paths):
        documents["texts"].append(doc["text"])
        documents["metadata"].append(doc["metadata"])
        documents["ids"].append(doc["id"])

    return documents


def embed(documents: dict[str, list]) -> None:
    vectorstore.add_texts(documents["texts"], documents["metadata"], documents["ids"])
    corpus_tokens = bm25s.tokenize(documents["texts"], return_ids=False, stopwords="en")
    bm25_retriever.index(corpus_tokens)


def store(documents: dict[str, list]) -> None:
    if Path(FAISS_INDEX_DIR).exists():
        shutil.rmtree(FAISS_INDEX_DIR)
    if Path(BM25_INDEX_DIR).exists():
        shutil.rmtree(BM25_INDEX_DIR)
    
    vectorstore.save_local(FAISS_INDEX_DIR)

    bm25_retriever.save(BM25_INDEX_DIR, corpus=documents["texts"])
    with open(f"{BM25_INDEX_DIR}/id_map.json", "w") as f:
        docs = []
        for id, metadata in zip(documents["ids"], documents["metadata"]):
            docs.append({"id": id, "metadata": metadata})
        json.dump(docs, f)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "-d",
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing documents to index",
    )
    args = args_parser.parse_args()

    documents = load_documents(args.data_dir)
    embed(documents)
    store(documents)
