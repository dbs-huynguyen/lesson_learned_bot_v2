import pytz
from datetime import datetime
from operator import itemgetter, add
from typing import Any, TypedDict, Optional, Annotated, Union

from qdrant_client.http.models import Filter
from langchain_core.documents import Document
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph

from src.agent.common import (
    get_qdrant_store,
    get_reranker,
    keyword_extraction_agent,
    date_extraction_agent,
    to_qdrant_filter,
    ExtractionDate,
    ExtractionKeyword,
    MetadataFilter,
)


class InputSchema(TypedDict):
    query: str
    new_query: str
    top_k: int
    score_threshold: float
    collection_name: str
    use_reranking: bool


class StateSchema(TypedDict):
    query: str
    new_query: str
    top_k: int
    score_threshold: float
    collection_name: str
    use_reranking: bool
    keyword_filter: Optional[MetadataFilter[ExtractionKeyword]]
    date_filter: Optional[MetadataFilter[ExtractionDate]]
    metadata_filter: Optional[Filter]
    retrieved_docs: list[Document]


class OutputSchema(TypedDict):
    final_docs: Annotated[list[Document], add]


def extract_keyword(state: InputSchema) -> dict[str, Any]:
    try:
        output = keyword_extraction_agent().invoke(state["query"])
        print(output)
        keyword_filter = MetadataFilter[ExtractionKeyword](must=[output])
    except Exception as e:
        print(str(e))
        keyword_filter = None

    return {"keyword_filter": keyword_filter}


def extract_date(state: InputSchema) -> dict[str, Any]:
    try:
        output = date_extraction_agent().invoke(
            dict(
                query=state["query"],
                now=datetime.now(tz=pytz.timezone("Asia/Ho_Chi_Minh")).strftime("%Y-%m-%d"),
            )
        )
        print(output)
        date_filter = MetadataFilter[ExtractionDate](must=[output])
    except Exception as e:
        print(str(e))
        date_filter = None

    return {"date_filter": date_filter}


def combine_filter(state: StateSchema) -> dict[str, Any]:
    if state["date_filter"] and state["keyword_filter"]:
        metadata_filter = MetadataFilter[Union[ExtractionKeyword, ExtractionDate]](
            must=state["keyword_filter"].must + state["date_filter"].must
        )
    elif state["date_filter"]:
        metadata_filter = state["date_filter"]
    elif state["keyword_filter"]:
        metadata_filter = state["keyword_filter"]
    else:
        metadata_filter = None

    if metadata_filter is not None:
        metadata_filter = to_qdrant_filter(metadata_filter)

    return {"metadata_filter": metadata_filter}


def hybrid_search(state: StateSchema) -> dict[str, Any]:
    print(state["metadata_filter"].model_dump_json(indent=2))
    results = get_qdrant_store(state["collection_name"]).similarity_search_with_score(
        query=state["new_query"],
        k=state["top_k"],
        filter=state["metadata_filter"],
        score_threshold=state["score_threshold"],
    )
    print("========== Search Results ==========")
    if state["collection_name"] == "questions":
        print(
            *(f"{doc.metadata['doc_id']}: {score}" for doc, score in results), sep="\n"
        )
    else:
        print(
            *(
                f"{doc.metadata['source']}#{doc.metadata['page_number']}: {score}"
                for doc, score in results
            ),
            sep="\n",
        )

    return {"retrieved_docs": list(map(itemgetter(0), results))}


def rerank_docs(state: StateSchema) -> dict[str, Any]:
    if state["collection_name"] == "lessons_learned":
        retrieved_docs = state["retrieved_docs"]
    else:
        qdrant_store = get_qdrant_store("lessons_learned")
        retrieved_docs = qdrant_store.get_by_ids(
            list(
                dict.fromkeys(
                    doc.metadata["lesson_learned_id"] for doc in state["retrieved_docs"]
                )
            )
        )
        print("========== Mapped Results ==========")
        print(
            *(
                f"{doc.metadata['source']}#{doc.metadata['page_number']}"
                for doc in retrieved_docs
            ),
            sep="\n",
        )

    if state["use_reranking"]:
        reranker = get_reranker()
        results = reranker.compress_documents_with_score(
            retrieved_docs, state["new_query"]
        )
        print("========== Reranked Results ==========")
        print(
            *(
                f"{doc.metadata['source']}#{doc.metadata['page_number']}: {score}"
                for doc, score in results
            ),
            sep="\n",
        )
        docs = list(map(itemgetter(0), results))
    else:
        docs = retrieved_docs

    writer = get_stream_writer()
    writer(
        {
            "type": "reasoning",
            "message": "Sắp xếp {} tài liệu theo mức độ liên quan.".format(len(docs)),
        }
    )

    return {"final_docs": docs}


# Define the graph
graph = (
    StateGraph(
        state_schema=StateSchema,
        input_schema=InputSchema,
        output_schema=OutputSchema,
    )
    # define nodes
    .add_node("extract_keyword", extract_keyword)
    .add_node("extract_date", extract_date)
    .add_node("combine_filter", combine_filter)
    .add_node("hybrid_search", hybrid_search)
    .add_node("rerank_docs", rerank_docs)
    # define workflow
    .set_entry_point("extract_keyword")
    .set_entry_point("extract_date")
    .add_edge("extract_keyword", "combine_filter")
    .add_edge("extract_date", "combine_filter")
    .add_edge("combine_filter", "hybrid_search")
    .add_edge("hybrid_search", "rerank_docs")
    .set_finish_point("rerank_docs")
    # compile the graph
    .compile(name="rag_agent_graph")
)
