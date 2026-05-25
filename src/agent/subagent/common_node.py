import pytz
from datetime import datetime
from typing import Any, Optional, Union, Annotated, Literal

from qdrant_client.http.models import Filter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.config import get_stream_writer
from langgraph.graph import MessagesState

from src.agent.common import (
    decision_making_agent,
    keyword_extraction_agent,
    date_extraction_agent,
    to_qdrant_filter,
    merge_documents,
    answer_agent,
    ExtractionDate,
    ExtractionKeyword,
    MetadataFilter,
)
from src.lib.prompts import ANSWER_DIRECT_SYSTEM_PROMPT


class InputSchema(MessagesState):
    pass


class StateSchema(MessagesState):
    documents: Annotated[dict[str, Document], merge_documents]
    date_filter: Optional[MetadataFilter[ExtractionDate]]
    keyword_filter: Optional[MetadataFilter[ExtractionKeyword]]
    metadata_filter: Optional[Filter]
    system_prompt: Optional[ChatPromptTemplate]
    relevant_docs: Optional[str]


class OutputSchema(MessagesState):
    documents: Annotated[dict[str, Document], merge_documents]


def decide_retrieval(
    state: StateSchema,
) -> Literal["retrieve_documents", "answer_directly"]:
    writer = get_stream_writer()
    resp = decision_making_agent().invoke(state["messages"][-1].content)
    print(f"Should retrieve relevant documents: {resp.content}")
    if resp.content.strip().lower() == "yes":
        writer(
            {
                "type": "reasoning",
                "message": "Truy xuất tài liệu liên quan.",
            }
        )
        return "retrieve_documents"
    else:
        return "answer_directly"


def answer_directly(state: StateSchema) -> dict[str, Any]:
    return {"system_prompt": ANSWER_DIRECT_SYSTEM_PROMPT}


def extract_date(state: StateSchema) -> dict[str, Any]:
    # resp = date_extraction_agent().invoke(
    #     dict(
    #         query=state["messages"][-1].content,
    #         now=datetime.now(tz=pytz.timezone("Asia/Ho_Chi_Minh")).strftime("%Y-%m-%d"),
    #     )
    # )

    date_filter = None
    # if resp["parsing_error"] is None:
    #     date_filter = MetadataFilter[ExtractionDate](must=[resp["parsed"]])

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


def answer(state: StateSchema) -> dict[str, Any]:
    if "relevant_docs" in state and state["relevant_docs"]:
        system_prompt = state["system_prompt"].format(
            relevant_docs=state["relevant_docs"]
        )
    else:
        system_prompt = state["system_prompt"].format()
    response = answer_agent().invoke(
        input={"messages": state["messages"]},
        context={"system_prompt": system_prompt},
    )
    # [print(f"{msg.type.upper()}: {msg.content}\n") for msg in response["messages"]]

    return {"messages": [response["messages"][-1]]}
