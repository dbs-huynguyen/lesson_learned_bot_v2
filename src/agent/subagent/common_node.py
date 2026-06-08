import pytz
import operator
import time
from datetime import datetime
from typing import Any, Optional, TypedDict, Union, Annotated, Literal

from qdrant_client.http.models import Filter
from langchain.messages import AIMessage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.config import get_stream_writer
from langgraph.graph import MessagesState
from langgraph.types import Send

from src.agent.common import (
    decision_making_agent,
    keyword_extraction_agent,
    date_extraction_agent,
    build_relevant_docs_ctx,
    rewrite_query_agent,
    to_qdrant_filter,
    merge_documents,
    answer_agent,
    ExtractionDate,
    ExtractionKeyword,
    MetadataFilter,
    KeywordFilter,
    DateFilter,
)


class InputSchema(MessagesState):
    pass


class ContextSchema(TypedDict):
    top_k: int
    score_threshold: float
    system_prompt: ChatPromptTemplate


class StateSchema(MessagesState):
    documents: Annotated[dict[str, Document], merge_documents]
    date_filter: Optional[DateFilter]
    keyword_filter: Optional[KeywordFilter]
    metadata_filter: Optional[Filter]
    relevant_docs: str
    final_docs: Annotated[list[Document], operator.add]
    system_prompt: str
    top_k: Optional[int]
    score_threshold: Optional[float]
    collection_name: Optional[str]
    use_reranking: Optional[bool]
    query_retrieval: Optional[str]


class ContextSchema(TypedDict):
    top_k: int
    score_threshold: float
    system_prompt: ChatPromptTemplate


class OutputSchema(MessagesState):
    documents: Annotated[dict[str, Document], merge_documents]


def decide_retrieval(
    state: StateSchema,
) -> Literal["retrieve_documents", "answer_directly"]:
    should_retrieve = decision_making_agent().invoke(state["messages"][-1].content)
    print(f"Should retrieve: {should_retrieve}")
    return "retrieve_documents"

    if should_retrieve:
        return "retrieve_documents"
    else:
        return "answer_directly"


def answer_directly(state: StateSchema) -> dict[str, Any]:
    message = "Xin lỗi nhưng tôi không thể trả lời câu hỏi của bạn."
    return {"messages": [AIMessage(message)]}


def rewrite_query(state: StateSchema) -> Send:
    # new_query = rewrite_query_agent().invoke(state["messages"])
    # print("Original query:", state["messages"][-1].content)
    # print("New query:", new_query)
    return Send(
        "rag_agent",
        dict(
            query=state["messages"][-1].content,
            query_retrieval=state["query_retrieval"],
            top_k=state["top_k"],
            score_threshold=state["score_threshold"],
            collection_name=state["collection_name"],
            use_reranking=state["use_reranking"],
        ),
    )


def extract_date(state: StateSchema) -> dict[str, Any]:
    resp = date_extraction_agent().invoke(
        dict(
            query=state["messages"][-1].content,
            now=datetime.now(tz=pytz.timezone("Asia/Ho_Chi_Minh")).strftime("%Y-%m-%d"),
        )
    )

    date_filter = None
    if resp["parsing_error"] is None:
        date_filter = MetadataFilter[ExtractionDate](must=[resp["parsed"]])

    return {"date_filter": date_filter}


def extract_keyword(state: StateSchema) -> dict[str, Any]:
    resp = keyword_extraction_agent().invoke(dict(query=state["messages"][-1].content))

    keyword_filter = None
    if resp["parsing_error"] is None:
        keyword_filter = MetadataFilter[ExtractionKeyword](must=[resp["parsed"]])

    return {"keyword_filter": keyword_filter}


def combine_filter(state: StateSchema) -> dict[str, Any]:
    if state["date_filter"] and state["keyword_filter"]:
        metadata_filter = MetadataFilter[Union[ExtractionKeyword, ExtractionDate]](
            must=state["date_filter"].must + state["keyword_filter"].must
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


def merge_docs(state: StateSchema) -> dict[str, Any]:
    # docs = deduplicate(state["final_docs"])
    relevant_docs, documents = build_relevant_docs_ctx(state["final_docs"])
    return {"relevant_docs": relevant_docs, "documents": documents}


def answer(state: StateSchema) -> dict[str, Any]:
    response = answer_agent().invoke(
        input={"messages": state["messages"]},
        context={
            "system_prompt": state["system_prompt"].format(
                relevant_docs=state["relevant_docs"]
            )
        },
    )

    return {"messages": [response["messages"][-1]]}
