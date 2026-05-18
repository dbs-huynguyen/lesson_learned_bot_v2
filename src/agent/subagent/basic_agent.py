import pytz
from datetime import datetime
from typing import Any, Optional, Union, Annotated, Literal

from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from qdrant_client.http.models import Filter
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langgraph.graph import MessagesState, StateGraph

from src.agent.common import (
    decision_making_agent,
    keyword_extraction_agent,
    date_extraction_agent,
    answer_agent,
    get_qdrant_store,
    get_reranker,
    to_qdrant_filter,
    merge_documents,
    ExtractionDate,
    ExtractionKeyword,
    MetadataFilter,
    AgentType,
)


class StateSchema(MessagesState):
    documents: Annotated[dict[str, Document], merge_documents]
    date_filter: Optional[MetadataFilter[ExtractionDate]]
    keyword_filter: Optional[MetadataFilter[ExtractionKeyword]]
    metadata_filter: Optional[Filter]
    context: Optional[str]


class OutputSchema(MessagesState):
    documents: Annotated[dict[str, Document], merge_documents]


def decide_retrieval(state: StateSchema) -> Literal["retrieve_documents", "answer"]:
    resp = decision_making_agent().invoke(state["messages"][-1].content)
    print(resp.content)
    if resp.content == "yes":
        return "retrieve_documents"
    else:
        return "answer"


def retrieve_documents(state: StateSchema) -> dict[str, Any]:
    return state


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


def hybrid_search_and_rerank(state: StateSchema) -> dict[str, Any]:
    print(state["metadata_filter"].model_dump_json(indent=2))
    retriever = ContextualCompressionRetriever(
        base_retriever=get_qdrant_store().as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 50,
                "filter": state["metadata_filter"],
                # "score_threshold": 0.3,
            },
        ),
        base_compressor=get_reranker(),
    )

    docs = retriever.invoke(state["messages"][-1].content)
    docs = docs[:50]
    print(len(docs))

    context = "\n\n".join(
        [
            (
                f"{doc.metadata['source']}\n"
                f"Phần: {doc.metadata['section']}\n"
                f"Trang: {doc.metadata['page_number']}\n"
                f"Ngày ghi nhận: {doc.metadata['occurred_at']}\n"
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
    response = answer_agent(AgentType.BASIC).invoke(
        {"messages": state["messages"]},
        context={"context": state["context"]},
    )
    [print(f"{msg.type.upper()}: {msg.content}\n") for msg in response["messages"]]

    return {"messages": [response["messages"][-1]]}


# Define the graph
graph = (
    StateGraph(state_schema=StateSchema, output_schema=OutputSchema)
    # define nodes
    .add_node("retrieve_documents", retrieve_documents)
    .add_node("extract_keyword", extract_keyword)
    .add_node("extract_date", extract_date)
    .add_node("combine_filter", combine_filter)
    .add_node("hybrid_search_and_rerank", hybrid_search_and_rerank)
    .add_node("answer", answer)
    # define workflow
    .set_conditional_entry_point(decide_retrieval)
    .add_edge("retrieve_documents", "extract_keyword")
    .add_edge("retrieve_documents", "extract_date")
    .add_edge("extract_keyword", "combine_filter")
    .add_edge("extract_date", "combine_filter")
    .add_edge("combine_filter", "hybrid_search_and_rerank")
    .add_edge("hybrid_search_and_rerank", "answer")
    .set_finish_point("answer")
    # compile the graph
    .compile(name="basic_agent_graph")
)
