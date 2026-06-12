#!/usr/bin/env python3
import os
import json
import inspect
import argparse
import shutil
from functools import lru_cache
from itertools import chain
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

from sqlalchemy import create_engine, Column, String, Integer, DateTime, JSON, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from datetime import datetime

from langchain_community.embeddings import InfinityEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PayloadSchemaType,
    Modifier,
    KeywordIndexParams,
    UuidIndexParams,
    DatetimeIndexParams,
    MultiVectorConfig,
    MultiVectorComparator,
    HnswConfigDiff,
)

from src.lib.parser import LessonsLearnedParser, MyDocument

# SQLAlchemy Setup
Base = declarative_base()


class LessonLearnedModel(Base):
    __tablename__ = "lessons_learned"

    id = Column(String, primary_key=True)
    page_content = Column(Text, nullable=False)
    source = Column(String)
    doc_type = Column(String)
    project_name = Column(String)
    occurred_at = Column(String)
    chunk_type = Column(String)
    department = Column(String)

    def __repr__(self):
        return f"<LessonLearnedModel(id={self.id}, source={self.source}, doc_type={self.doc_type}, project_name={self.project_name}, occurred_at={self.occurred_at})>"


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


@lru_cache
def get_embeddings():
    return InfinityEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        infinity_api_url=os.getenv("EMBEDDING_BASE_URL"),
    )


@lru_cache
def get_qdrant_store(collection_name: str) -> QdrantVectorStore:
    """Tạo QdrantVectorStore với cache riêng cho mỗi collection_name

    Args:
        collection_name: Tên collection trong Qdrant

    Returns:
        QdrantVectorStore đã được cache cho collection_name tương ứng
    """
    # Lazy import to avoid slow PyTorch loading at module import time
    from langchain_qdrant import FastEmbedSparse

    client = QdrantClient(url=os.getenv("QDRANT_URL"))

    try:
        client.get_collection(collection_name)
        client.delete_collection(collection_name)
        logger.info(
            json.dumps(
                {
                    "event": inspect.currentframe().f_code.co_name,
                    "message": "Collection already exists, deleting and recreating...",
                },
                ensure_ascii=False,
            )
        )
    except Exception:
        logger.info(
            json.dumps(
                {
                    "event": inspect.currentframe().f_code.co_name,
                    "message": "Collection does not exist, creating new collection...",
                },
                ensure_ascii=False,
            )
        )
    finally:
        client.create_collection(
            collection_name=collection_name,
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

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=get_embeddings(),
        vector_name="dense",
        sparse_embedding=FastEmbedSparse(),
        sparse_vector_name="sparse",
        retrieval_mode=RetrievalMode.HYBRID,
    )


def create_session_maker() -> Session:
    # Tạo engine và session
    engine = create_engine(f"sqlite:///{os.getenv("SQLITE_DB_NAME")}", echo=False)
    # Xóa tất cả bảng cũ trước khi tạo lại
    LessonLearnedModel.__table__.drop(engine, checkfirst=True)
    QuestionModel.__table__.drop(engine, checkfirst=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def create_lessons_learned_collection(
    collection_name: str = "lessons_learned",
) -> QdrantVectorStore:
    qdrant_store = get_qdrant_store(collection_name)
    qdrant_store.client.create_payload_index(
        collection_name=collection_name,
        field_name="project_name",
        field_schema=KeywordIndexParams(
            type=PayloadSchemaType.KEYWORD,
            on_disk=False,
            enable_hnsw=False,
        ),
    )
    qdrant_store.client.create_payload_index(
        collection_name=collection_name,
        field_name="occurred_at",
        field_schema=DatetimeIndexParams(
            type=PayloadSchemaType.DATETIME,
            on_disk=False,
            enable_hnsw=False,
        ),
    )

    return qdrant_store


def save_to_qdrant(
    documents: list[MyDocument], lessons_learned_store: QdrantVectorStore
) -> None:
    lessons_learned_store.add_documents(documents)


def _save_lessons_learned_to_sqlite(
    documents: list[MyDocument], session: Session
) -> None:
    logger.info(
        json.dumps(
            {
                "event": inspect.currentframe().f_code.co_name,
                "message": f"Saving {len(documents)} lessons learned to SQLite database",
            },
            ensure_ascii=False,
        )
    )

    try:
        saved_count = 0
        for doc in documents:
            metadata = doc.metadata if hasattr(doc, "metadata") else {}

            content = (
                doc.page_content
                if hasattr(doc, "page_content")
                else getattr(doc, "text", "")
            )

            doc_model = LessonLearnedModel(
                id=doc.id,
                page_content=content,
                source=metadata.get("source", ""),
                doc_type=metadata.get("doc_type", ""),
                project_name=metadata.get("project_name", ""),
                occurred_at=metadata.get("occurred_at", ""),
                chunk_type=metadata.get("chunk_type", ""),
                department=metadata.get("department", ""),
            )

            # Merge để tránh duplicate nếu id đã tồn tại
            session.merge(doc_model)
            saved_count += 1

            # Commit theo batch để tăng hiệu suất
            if saved_count % 100 == 0:
                session.commit()
                logger.info(f"Committed {saved_count}/{len(documents)} documents")

        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Error saving learned lessons to SQLite: {e}")
    finally:
        session.close()


def save_to_sqlite(documents: list[MyDocument], session: Session) -> None:
    _save_lessons_learned_to_sqlite(documents, session)


def save_to_markdown(documents: list[MyDocument], output_dir: str) -> None:
    logger.info(
        json.dumps(
            {
                "event": inspect.currentframe().f_code.co_name,
                "message": f"Saving {len(documents)} documents to markdown files in: {output_dir}",
            },
            ensure_ascii=False,
        )
    )

    # Xóa thư mục cũ nếu tồn tại và tạo thư mục mới
    output_path = Path("data", "BHKN", output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for idx, doc in enumerate(documents, 1):
        try:
            # Lấy metadata
            metadata = doc.metadata if hasattr(doc, "metadata") else {}

            content = doc.page_content

            md_content = []
            md_content.append(f"{content.strip()}\n")

            source = metadata.get("source", "unknown")
            page = metadata.get("page_number", "unknown")
            file_path = output_path / f"{source.split('.')[0]}_page_{page}.md"
            with open(file_path, "a", encoding="utf-8") as f:
                f.write("\n".join(md_content))

            saved_count += 1

            if saved_count % 100 == 0:
                logger.info(f"Saved {saved_count}/{len(documents)} markdown files")

        except Exception as e:
            logger.error(f"Error saving document {idx} to markdown: {e}")
            continue

    logger.info(
        json.dumps(
            {
                "event": inspect.currentframe().f_code.co_name,
                "message": f"Successfully saved {saved_count} documents to {output_dir}",
            },
            ensure_ascii=False,
        )
    )


def main(data_dir: Path, storage_type: str, markdown_dir: str):
    logger.info(
        json.dumps(
            {
                "event": inspect.currentframe().f_code.co_name,
                "message": "Application started",
                "storage_type": storage_type,
            },
            ensure_ascii=False,
        )
    )

    # parser = WorkInstructionParser(data_dir)
    parser = LessonsLearnedParser(data_dir)

    lessons_learned_store = create_lessons_learned_collection()
    session = create_session_maker()

    total_docs = 0
    try:
        for docs in parser():
            if storage_type in ["qdrant", "both", "all"]:
                save_to_qdrant(
                    documents=docs, lessons_learned_store=lessons_learned_store
                )

            if storage_type in ["sqlite", "both", "all"]:
                save_to_sqlite(documents=docs, session=session)

            # if storage_type in ["markdown", "all"]:
            #     save_to_markdown(documents, markdown_dir)

            total_docs += len(docs)
    finally:
        lessons_learned_store.client.close()

    logger.info(f"{total_docs} documents have been processed!")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing documents to index",
    )
    args_parser.add_argument(
        "--storage-type",
        type=str,
        choices=["qdrant", "sqlite", "markdown", "both", "all"],
        default="all",
        help="Storage type: qdrant (vector DB), sqlite (relational DB), markdown (markdown files), both (qdrant+sqlite), or all (all types, default: qdrant)",
    )
    args_parser.add_argument(
        "--markdown-dir",
        type=str,
        default="output_md",
        help="Directory to save markdown files (default: output_md)",
    )
    args = args_parser.parse_args()

    main(args.data_dir, args.storage_type, args.markdown_dir)
