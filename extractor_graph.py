#!/usr/bin/env python3
import os
import json
import inspect
import argparse
from itertools import chain
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PayloadSchemaType,
    Modifier,
    KeywordIndexParams,
    DatetimeIndexParams,
    MultiVectorConfig,
    MultiVectorComparator,
    HnswConfigDiff,
)

from src.lib.parser import LessonsLearnedParser, MyDocument


def setup_logging():
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File handler (có rotate)
    file_handler = RotatingFileHandler("app.log", maxBytes=5_000_000, backupCount=3)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)


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
    from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse

    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        base_url=os.getenv("EMBEDDING_BASE_URL"),
    )
    d = len(embeddings.embed_query("hello world"))
    client = QdrantClient(url=os.getenv("QDRANT_URL"))
    if "bhkn" in [
        collection.name for collection in client.get_collections().collections
    ]:
        client.delete_collection(collection_name="bhkn")
    client.create_collection(
        collection_name="bhkn",
        vectors_config={
            "dense": VectorParams(
                size=1024,
                distance=Distance.COSINE,
            ),
            # "multi": VectorParams(
            #     size=96,
            #     distance=Distance.COSINE,
            #     multivector_config=MultiVectorConfig(
            #         comparator=MultiVectorComparator.MAX_SIM,
            #     ),
            #     hnsw_config=HnswConfigDiff(m=0),  #  Disable HNSW for reranking
            # ),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
                modifier=Modifier.IDF,
            ),
        },
    )
    client.create_payload_index(
        collection_name="bhkn",
        field_name="project_name",
        field_schema=KeywordIndexParams(
            type=PayloadSchemaType.KEYWORD,
            on_disk=False,
            enable_hnsw=False,
        ),
    )
    client.create_payload_index(
        collection_name="bhkn",
        field_name="occurred_at",
        field_schema=DatetimeIndexParams(
            type=PayloadSchemaType.DATETIME,
            on_disk=False,
            enable_hnsw=False,
        ),
    )

    qdrant_store = QdrantVectorStore(
        client=client,
        collection_name="bhkn",
        embedding=embeddings,
        vector_name="dense",
        sparse_embedding=FastEmbedSparse(),
        sparse_vector_name="sparse",
        retrieval_mode=RetrievalMode.HYBRID,
    )
    try:
        qdrant_store.add_documents(documents)
    finally:
        qdrant_store.client.close()


def main(data_dir: Path):
    logger.info(
        json.dumps(
            {
                "event": inspect.currentframe().f_code.co_name,
                "message": "Application started",
            },
            ensure_ascii=False,
        )
    )

    # parser = WorkInstructionParser(data_dir)
    parser = LessonsLearnedParser(data_dir)

    documents = list(chain.from_iterable(parser()))
    embed(documents)

    logger.info(f"{len(documents)} documents have been processed!")


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

    main(args.data_dir)
