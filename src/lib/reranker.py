import requests
import logging
from httpx import URL
from typing import Any, Optional
from pydantic import ConfigDict

from langchain_core.documents import Document, BaseDocumentCompressor
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable


logger = logging.getLogger(__name__)


class MyReranker(BaseDocumentCompressor):
    base_url: str | URL
    model: str
    top_n: Optional[int] = None
    timeout: int = 30
    path: str | URL = "score"

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    def _enforce_trailing_slash(self, base_url: URL) -> URL:
        if base_url.raw_path.endswith(b"/"):
            return base_url
        return base_url.copy_with(raw_path=base_url.raw_path + b"/")

    def model_post_init(self, __context: Any) -> None:
        self.base_url = self._enforce_trailing_slash(URL(self.base_url))

    def _prepare_url(self, url: str) -> URL:
        merge_url = URL(url)
        if merge_url.is_relative_url:
            merge_raw_path = self.base_url.raw_path + merge_url.raw_path.lstrip(b"/")
            return self.base_url.copy_with(raw_path=merge_raw_path)

        return merge_url

    def _call_api(self, query: str, documents: list[Document]):
        response = requests.post(
            self._prepare_url(self.path),
            headers={
                "accept": "application/json",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "encoding_format": "float",
                "queries": query,
                "documents": [doc.page_content for doc in documents],
            },
            timeout=self.timeout,
        )

        response.raise_for_status()
        return response.json()

    def compress_documents(
        self, documents: list[Document], query: str, **kwargs
    ) -> list[Document]:
        if not documents:
            return []

        try:
            result = self._call_api(query, documents)

            scores = result.get("data", [])

            scored_docs = []
            for item in scores:
                idx = item["index"]
                score = item["score"]
                doc = documents[idx]
                doc.metadata["rerank_score"] = score
                scored_docs.append((doc, score))

            scored_docs.sort(key=lambda x: x[1], reverse=True)

            if self.top_n:
                scored_docs = scored_docs[: self.top_n]

            return [doc for doc, _ in scored_docs]
        except requests.RequestException as e:
            logger.error("Error occurred while calling the API: %s", e)
            if self.top_n:
                return documents[: self.top_n]
            return documents
