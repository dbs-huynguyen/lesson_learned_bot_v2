#!/usr/bin/env python3
import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, DatetimeRange

from src.lib.reranker import MyReranker

load_dotenv()


def main():
    from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        base_url=os.getenv("EMBEDDING_BASE_URL"),
    )
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
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
    reranker = MyReranker(
        base_url=os.getenv("RERANKER_BASE_URL"),
        model=os.getenv("RERANKER_MODEL"),
        top_n=10,
    )
    # print(qdrant_store.client.get_collection("bhkn"))
    compression_retriever = ContextualCompressionRetriever(
        base_retriever=qdrant_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 1000,
                # "filter": Filter(
                #     must=[
                #         # FieldCondition(
                #         #     key="metadata.doc_type",
                #         #     match=MatchValue(value="BHKN"),
                #         # ),
                #         FieldCondition(
                #             key="metadata.occurred_at",
                #             range=DatetimeRange(lte="2024-11-07T00:00:00"),
                #         ),
                #     ]
                # ),
            },
        ),
        base_compressor=reranker,
    )

    query = "AWS S3"

    print("=" * 10 + " Compression Retriever Results " + "=" * 10)
    docs = compression_retriever.invoke(query)
    for doc in sorted(docs, key=lambda x: (x.metadata["project_name"])):
        print(f"{doc.metadata['project_name']}: {doc.metadata['occurred_at']}")
        # print(f"{doc.metadata['section']}:")
        # print(f"{doc.page_content.strip()}\n\n")

    qdrant_store.client.close()


if __name__ == "__main__":
    main()
