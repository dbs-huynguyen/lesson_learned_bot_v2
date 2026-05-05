#!/usr/bin/env python3
import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


from src.lib.reranker import MyReranker
from src.lib.retriever import MyBM25Retriever


load_dotenv()

FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "faiss_index")
QDRANT_INDEX_DIR = os.getenv("QDRANT_INDEX_DIR", "qdrant_index")
BM25_INDEX_DIR = os.getenv("BM25_INDEX_DIR", "bm25_index")


embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
    base_url=os.getenv("EMBEDDING_BASE_URL"),
)
client = QdrantClient(path=QDRANT_INDEX_DIR)
qdrant_store = QdrantVectorStore(
    client=client,
    collection_name="bhkn",
    embedding=embeddings,
)

bm25_retriever = MyBM25Retriever(k=12)
# hybrid_retriever = EnsembleRetriever(
#     retrievers=[bm25_retriever, faiss_retriever],
#     weights=[0.5, 0.5],
#     id_key="doc_id",
# )

# reranker = MyReranker(
#     base_url=os.getenv("RERANKER_BASE_URL"),
#     model=os.getenv("RERANKER_MODEL"),
#     top_n=4,
# )
# compression_retriever = ContextualCompressionRetriever(
#     base_retriever=hybrid_retriever,
#     base_compressor=reranker,
# )

from qdrant_client.http.models import Filter, FieldCondition, MatchValue, DatetimeRange

def main():
    # print(qdrant_store.client.get_collection("bhkn"))
    qdrant_retriever = qdrant_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 12,
            "filter": Filter(
                must=[
                    FieldCondition(
                        key="metadata.doc_type",
                        match=MatchValue(value="BHKN"),
                    ),
                    FieldCondition(
                        key="metadata.occurred_at",
                        range=DatetimeRange(lte="2024-11-07T00:00:00"),
                    ),
                ]
            ),
        },
    )
    query = "AWS S3"

    print("=" * 10 + " Compression Retriever Results " + "=" * 10)
    docs = qdrant_retriever.invoke(query)
    for doc in docs:
        print(doc.metadata)
        print(f"{doc.metadata['section']}:")
        print(f"{doc.page_content.strip()}\n\n")

    qdrant_store.client.close()


if __name__ == "__main__":
    main()
