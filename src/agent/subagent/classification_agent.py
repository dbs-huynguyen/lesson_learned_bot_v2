import pytz
from datetime import datetime
from typing import Any, Optional, Union

from langchain_core.documents import Document
from langgraph.graph import MessagesState, StateGraph
from qdrant_client.http.models import Filter

from src.agent.common import (
    llm_extract_keyword,
    llm_extract_datetime,
    get_qdrant_store,
    get_answer_agent,
    to_qdrant_filter,
    ExtractionDatetime,
    ExtractionKeyword,
    MetadataFilter,
    AgentType,
)
from src.lib.prompts import EXTRACT_KEYWORD_PROMPT, EXTRACT_DATE_PROMPT


class StateSchema(MessagesState):
    documents: list[Document] | None
    date_filter: Optional[MetadataFilter[ExtractionDatetime]]
    keyword_filter: Optional[MetadataFilter[ExtractionKeyword]]
    metadata_filter: Optional[Filter]
    context: Optional[str]


class OutputSchema(MessagesState):
    documents: Optional[list[dict[str, Document]]]


def extract_date(state: StateSchema) -> dict[str, Any]:
    resp = llm_extract_datetime().invoke(
        EXTRACT_DATE_PROMPT.format(
            query=state["messages"][-1].content,
            schema=ExtractionDatetime.model_json_schema(),
            now=datetime.now(tz=pytz.timezone("Asia/Ho_Chi_Minh")).strftime("%Y-%m-%d"),
        )
    )

    date_filter = None
    if resp["parsing_error"] is None:
        date_filter = MetadataFilter[ExtractionDatetime](must=[resp["parsed"]])

    return {"date_filter": date_filter}


def extract_keyword(state: StateSchema) -> dict[str, Any]:
    resp = llm_extract_keyword().invoke(
        EXTRACT_KEYWORD_PROMPT.format(
            query=state["messages"][-1].content,
            schema=ExtractionKeyword.model_json_schema(),
        )
    )

    keyword_filter = None
    if resp["parsing_error"] is None:
        keyword_filter = MetadataFilter[ExtractionKeyword](must=[resp["parsed"]])

    return {"keyword_filter": keyword_filter}


def combine_datetime_and_keyword(state: StateSchema) -> dict[str, Any]:
    if state["date_filter"] and state["keyword_filter"]:
        metadata_filter = MetadataFilter[Union[ExtractionKeyword, ExtractionDatetime]](
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


def hybrid_search(state: StateSchema) -> dict[str, Any]:
    print(state["metadata_filter"])
    qdrant_retriever = get_qdrant_store().as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 100,
            "filter": state["metadata_filter"],
        },
    )

    docs = qdrant_retriever.invoke(state["messages"][-1].content)
    print(len(docs))

    context = "\n\n".join(
        [
            (
                f"{doc.metadata['source']}#page={doc.metadata['page_number']}\n"
                f"_Ngày ghi nhận: {doc.metadata['occurred_at']}_\n"
                f"_Phần: {doc.metadata['section']}_\n"
                f"{doc.page_content.strip()}"
            )
            for doc in docs
        ]
    )

    if not context:
        context = "Không tìm thấy tài liệu phù hợp."

    documents = {
        f"{doc.metadata['source']}#page={doc.metadata['page_number']}": doc
        for doc in docs
    }

    return {"context": context, "documents": documents}


def answer(state: StateSchema) -> dict[str, Any]:
    response = get_answer_agent(AgentType.BASIC).invoke(
        {"messages": state["messages"]},
        context={"context": state["context"]},
    )
    [print(msg.type) for msg in response["messages"]]

    return {"messages": response["messages"]}


# Define the graph
graph = (
    StateGraph(state_schema=StateSchema, output_schema=OutputSchema)
    # define nodes
    .add_node("extract_keyword", extract_keyword)
    .add_node("extract_date", extract_date)
    .add_node("combine_datetime_and_keyword", combine_datetime_and_keyword)
    .add_node("hybrid_search", hybrid_search)
    .add_node("answer", answer)
    # define workflow
    .set_entry_point("extract_keyword")
    .set_entry_point("extract_date")
    .add_edge("extract_keyword", "combine_datetime_and_keyword")
    .add_edge("extract_date", "combine_datetime_and_keyword")
    .add_edge("combine_datetime_and_keyword", "hybrid_search")
    .add_edge("hybrid_search", "answer")
    .set_finish_point("answer")
    # compile the graph
    .compile(name="trend_agent_graph")
)
