import asyncio
import typing as t
from tqdm.auto import tqdm
from dataclasses import dataclass, field

from langgraph_sdk import get_client
from langgraph_sdk.client import LangGraphClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.embeddings import InfinityEmbeddings
from langchain_ollama import ChatOllama
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse


@dataclass
class RAGAgent:
    _client: LangGraphClient | None = field(default=None, repr=False)

    async def async_invoke(self, input: dict) -> dict:
        """Async version of invoke for parallel processing."""
        if self._client is None:
            self._client = get_client(url="http://localhost:2024", timeout=120)

        prompt = input.get("question", "")
        resp = await self._client.runs.wait(
            thread_id=None,
            assistant_id="chat",
            input={"messages": [HumanMessage(prompt)]},
        )
        messages = resp.get("messages") or []
        docs: list[dict] = resp.get("context") or []

        try:
            response = AIMessage.model_validate(messages[-1])
            context = [Document(doc.get("page_content"), metadata=doc.get("metadata")) for doc in docs]
        except Exception as e:
            print(f"Error parsing AIMessage: {e}")
            response = AIMessage(content="")
            context = []

        return {
            "response": response.content,
            "context": context,
        }

    async def batch_invoke_async(self, inputs: t.List[dict], max_concurrent: int = 5) -> t.List[dict]:
        """
        Process multiple inputs in parallel with controlled concurrency.

        Args:
            inputs: List of input dicts, each containing a "question" key
            max_concurrent: Maximum number of concurrent requests (default: 5)

        Returns:
            List of results in the same order as inputs
        """
        if not inputs:
            return []

        semaphore = asyncio.Semaphore(max_concurrent)
        results = [None] * len(inputs)  # Pre-allocate to maintain order

        async def bounded_invoke(index: int, input_dict: dict, pbar: tqdm) -> None:
            """Process one input and update progress bar."""
            async with semaphore:
                try:
                    result = await self.async_invoke(input_dict)
                    results[index] = result
                except Exception as e:
                    print(f"Error processing question '{input_dict.get('question', '')}': {e}")
                    results[index] = {"response": "", "context": []}
                finally:
                    pbar.update(1)

        with tqdm(total=len(inputs), desc="Processing RAG queries", leave=True) as pbar:
            tasks = [bounded_invoke(i, input_dict, pbar) for i, input_dict in enumerate(inputs)]
            await asyncio.gather(*tasks)

        return results

    def batch_invoke(self, inputs: t.List[dict], max_concurrent: int = 5) -> t.List[dict]:
        """
        Synchronous wrapper for batch_invoke_async.

        Args:
            inputs: List of input dicts
            max_concurrent: Maximum concurrent requests

        Returns:
            List of results
        """
        return asyncio.run(self.batch_invoke_async(inputs, max_concurrent))

    def batch_invoke_with_executor(self, inputs: t.List[dict], max_workers: int = 5) -> t.List[dict]:
        """
        Alternative batch processing using ragas Executor for better progress tracking.

        Args:
            inputs: List of input dicts
            max_workers: Maximum number of concurrent workers

        Returns:
            List of results in the same order as inputs
        """
        from ragas.executor import Executor
        from ragas.run_config import RunConfig

        run_config = RunConfig(max_workers=max_workers)
        executor = Executor(
            desc="Processing RAG queries",
            show_progress=True,
            keep_progress_bar=False,
            raise_exceptions=False,
            run_config=run_config,
        )

        for input_dict in inputs:
            executor.submit(self.async_invoke, input_dict)

        return executor.results()


rag_pipeline = RAGAgent()

embeddings = InfinityEmbeddings(
    model="AITeamVN/Vietnamese_Embedding",
    infinity_api_url="http://192.168.88.179:2025",
)

llm = ChatOllama(
    model="qwen3.5:9b",
    base_url="http://192.168.88.179:11435",
    keep_alive=-1,
    seed=9999,
    num_ctx=32768,
    reasoning=False,
    temperature=0,
)
collection_name = "lessons_learned"
vectorstore = QdrantVectorStore(
    client=QdrantClient(url="http://192.168.88.179:6333"),
    collection_name=collection_name,
    embedding=embeddings,
    vector_name="dense",
    sparse_embedding=FastEmbedSparse(),
    sparse_vector_name="sparse",
    retrieval_mode=RetrievalMode.HYBRID,
)