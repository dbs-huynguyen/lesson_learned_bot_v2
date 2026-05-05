#!/usr/bin/env python3
import os
import json
import shutil
import inspect
import argparse
from itertools import chain
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

import bm25s
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PayloadSchemaType

from src.lib.parser import LessonsLearnedParser, MyDocument


def setup_logging():
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File handler (có rotate)
    file_handler = RotatingFileHandler(
        "app.log",
        maxBytes=5_000_000,
        backupCount=3
    )
    file_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)

QDRANT_INDEX_DIR = os.getenv("QDRANT_INDEX_DIR", "qdrant_index")
BM25_INDEX_DIR = os.getenv("BM25_INDEX_DIR", "bm25_index")


embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
    base_url=os.getenv("EMBEDDING_BASE_URL"),
)
d = len(embeddings.embed_query("hello world"))
client = QdrantClient(path=QDRANT_INDEX_DIR)
if "bhkn" in [collection.name for collection in client.get_collections().collections]:
    client.delete_collection(collection_name="bhkn")
client.create_collection(
    collection_name="bhkn",
    vectors_config=VectorParams(size=d, distance=Distance.COSINE),
)
client.create_payload_index(
    collection_name="bhkn",
    field_name="doc_type",
    field_schema=PayloadSchemaType.KEYWORD
)
qdrant_store = QdrantVectorStore(
    client=client,
    collection_name="bhkn",
    embedding=embeddings,
)
bm25_retriever = bm25s.BM25()


def embed(documents: list[MyDocument]) -> None:
    logger.info(
        json.dumps(
                {
                    "event": inspect.currentframe().f_code.co_name,
                    "message": f"Embedding {len(documents)} documents...",
                },
                ensure_ascii=False,
            )
    )
    try:
        qdrant_store.add_documents(documents)
        corpus_tokens = bm25s.tokenize([doc.page_content for doc in documents])
        bm25_retriever.index(corpus_tokens)

        if Path(BM25_INDEX_DIR).exists():
            shutil.rmtree(BM25_INDEX_DIR)

        bm25_retriever.save(BM25_INDEX_DIR, corpus=[doc.model_dump() for doc in documents])
    finally:
        qdrant_store.client.close()


if __name__ == "__main__":
    logger.info("Application started")
    
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "-d",
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing documents to index",
    )
    args = args_parser.parse_args()

    # parser = WorkInstructionParser(args.data_dir)
    parser = LessonsLearnedParser(args.data_dir)

    documents = list(chain.from_iterable(parser()))
    embed(documents)

    logger.info(
        json.dumps(
                {
                    "event": __name__,
                    "message": f"{len(documents)} documents have been processed!",
                },
                ensure_ascii=False,
            )
    )
