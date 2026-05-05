import os
from typing import Any, Literal, TypedDict
from dotenv import load_dotenv

from langchain.messages import HumanMessage
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.documents import Document
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState, StateGraph
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, DatetimeRange
from dateparser import parse
from dateparser.search import search_dates

from src.lib.prompts import ANSWER_PROMPT, ROUTE_QUERY_PROMPT, ANSWER_WITH_RAG_PROMPT, EXTRACT_KEYWORD_PROMPT
from src.lib.reranker import MyReranker
from src.lib.retriever import MyBM25Retriever

load_dotenv()

FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "faiss_index")
QDRANT_INDEX_DIR = os.getenv("QDRANT_INDEX_DIR", "qdrant_index")
BM25_INDEX_DIR = os.getenv("BM25_INDEX_DIR", "bm25_index")


class StateSchema(MessagesState):
    documents: list[Document] | None
    task: Literal["BHKN", "ISO"]
    metadata_filter: dict[str, Any] | None
    doc_type: list[str] | None


class ClassificationOutput(TypedDict):
    label: Literal["DIRECT", "RAG"]


DOC_TYPE = Literal["BHKN", "HD", "QT", "QĐ", "CS", "MT", "MTCV", "QH", "ST"]


class KeywordOutput(TypedDict):
    doc_type: DOC_TYPE


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

qdrant_retriever = qdrant_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 20},
)
bm25_retriever = MyBM25Retriever(k=20)
reranker = MyReranker(
    base_url=os.getenv("RERANKER_BASE_URL"),
    model=os.getenv("RERANKER_MODEL"),
    top_n=10,
)
compression_retriever = ContextualCompressionRetriever(
    base_retriever=EnsembleRetriever(
        retrievers=[bm25_retriever, qdrant_retriever],
        weights=[0.5, 0.5],
        id_key="doc_id",
    ),
    base_compressor=reranker,
)


model = ChatOllama(
    model=os.getenv("OLLAMA_LLM_MODEL"),
    base_url=os.getenv("OLLAMA_BASE_URL"),
    validate_model_on_init=False,
    client_kwargs={"timeout": 120},
    seed=9999,
    num_ctx=8192,
)
summarization_middleware = SummarizationMiddleware(
    model=model.bind(think=False, options={"num_predict": 1024, "temperature": 0.0}),
    trigger=[("tokens", 2000), ("messages", 8)],
    keep=("messages", 4), # Nguyên tắc: keep nên bằng ~50% trigger (tính theo messages), để có đủ buffer trước khi trigger lại.
)
answer = create_agent(
    model.bind(think=False, options={"num_predict": 1024, "temperature": 0.0}),
    system_prompt=ANSWER_PROMPT.format(),
    middleware=[summarization_middleware],
    name="answer_subgraph",
)
_answer_with_rag = create_agent(
    model.bind(think=False, options={"num_predict": 1024, "temperature": 0.0}),
    system_prompt=ANSWER_WITH_RAG_PROMPT.format(),
    middleware=[summarization_middleware],
    name="answer_with_rag_subgraph",
)


def prepare_thread(state: StateSchema) -> dict[str, Any]:
    return {
        "documents": None,
        "task": state.get("task"),
        "metadata_filter": state.get("metadata_filter"),
    }


def route_query(state: StateSchema) -> Literal["answer", "extract_keyword"]:
    return "extract_keyword"
    # resp = (
    #     model
    #     .with_structured_output(ClassificationOutput)
    #     .bind(think=False, options={"num_predict": 32, "temperature": 0.0})
    #     .invoke(ROUTE_QUERY_PROMPT.format(query=state["messages"][-1].content))
    # )
    # return "answer" if resp["label"] == "DIRECT" else "extract_keyword"


def learned_lessons_task(state: StateSchema) -> dict[str, Any]:
    return state


def extract_datetime_for_learned_lessons(state: StateSchema) -> dict[str, Any]:
    dates = parse(
        state["messages"][-1].content,
        settings={"RETURN_TIME_SPAN": True},
        languages=["vi"],
    )
    print(dates)
    return state


def extract_keyword_for_learned_lessons(state: StateSchema) -> dict[str, Any]:
    pass


def extract_keyword(state: StateSchema) -> dict[str, Any]:
    print(state.get("metadata_filter"))
    return state
    # resp = (
    #     model
    #     .with_structured_output(KeywordOutput)
    #     .bind(think=False, options={"num_predict": 32, "temperature": 0.0})
    #     .invoke(EXTRACT_KEYWORD_PROMPT.format(query=state["messages"][-1].content))
    # )
    # metadata_filter = {k: v for k, v in resp.items() if v is not None} or None
    # print(metadata_filter)
    # return {"metadata_filter": metadata_filter}


def hybrid_search(state: StateSchema) -> dict[str, Any]:
    if metadata_filter := state.get("metadata_filter"):
        metadata_filter = Filter(metadata_filter)
        qdrant_retriever = qdrant_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 20, "filter": metadata_filter},
        )
        bm25_retriever = MyBM25Retriever(k=20, filter=metadata_filter)
        retriever = ContextualCompressionRetriever(
            base_retriever=EnsembleRetriever(
                retrievers=[bm25_retriever, qdrant_retriever],
                weights=[0.5, 0.5],
                id_key="doc_id",
            ),
            base_compressor=reranker,
        )
    else:
        retriever = compression_retriever
        
    docs = retriever.invoke(state["messages"][-1].content)
    print(len(docs))
    return {"documents": docs}


def answer_with_rag(state: StateSchema) -> dict[str, Any]:
    user_prompt = "**CONTEXT:**\n{retrieved_docs}\n\n**QUESTION:**\n{query}".format(
        retrieved_docs="\n\n".join(
            [
                f"{doc.metadata['source']}#page={doc.metadata['page_number']}:  \n{doc.page_content.strip()}"
                for doc in state["documents"]
            ]
        ),
        query=state["messages"][-1].content,
    )

    messages = state["messages"][:-1] + [HumanMessage(content=user_prompt)]
    response = _answer_with_rag.invoke({"messages": messages})
    return response


# Define the graph
graph = (
    StateGraph(state_schema=StateSchema)
    # define nodes
    .add_node("prepare_thread", prepare_thread)
    .add_node("extract_keyword", extract_keyword)
    .add_node("hybrid_search", hybrid_search)
    .add_node("answer", answer)
    .add_node("answer_with_rag", answer_with_rag)
    # define workflow
    .set_entry_point("prepare_thread")
    .add_conditional_edges("prepare_thread", route_query)
    .add_edge("extract_keyword", "hybrid_search")
    .add_edge("hybrid_search", "answer_with_rag")
    .set_finish_point("answer")
    .set_finish_point("answer_with_rag")
    # compile the graph
    .compile(name="main_graph")
)
