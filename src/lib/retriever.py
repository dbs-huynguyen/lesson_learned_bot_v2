import os
from typing import Any
from pathlib import Path
from dotenv import load_dotenv
from pydantic import PrivateAttr

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

try:
    import bm25s
except ImportError:
    raise ImportError(
        "Could not import bm25s, please install with `pip install bm25s`."
    )


load_dotenv()


class MyBM25Retriever(BaseRetriever):
    k: int = 4
    filter: dict | None = None
    _bm25: bm25s.BM25 = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._bm25 = bm25s.BM25.load(
            os.getenv("BM25_INDEX_DIR", "bm25_index"),
            load_corpus=True,
            mmap=True,
        )
        self.filter = self.filter or None

    def _get_relevant_documents(self, query: str) -> list[Document]:
        tokenized_query = bm25s.tokenize(query, stopwords="en")
        corpus = list(self._bm25.corpus)

        if self.filter is not None:
            corpus = [doc for doc in corpus if all(doc["metadata"].get(k) == v for k, v in self.filter.items())]

        if len(corpus) == 0:
            return []

        docs = self._bm25.retrieve(tokenized_query, corpus=corpus, k=self.k, return_as="documents")
        return [Document(**doc) for doc in docs[0].tolist()]
