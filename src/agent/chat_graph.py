import typing as t
from pathlib import Path
from operator import itemgetter

from qdrant_client.models import Filter
from langchain.messages import AnyMessage
from langchain_core.documents import Document
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, add_messages

from src.lib.prompts import SYSTEM_PROMPT
from src.agent.common import (
    MetadataFilter,
    get_base_llm,
    build_qdrant_filter,
    create_complement_extraction_agent,
    create_reranker,
    create_retrieval_decision_agent,
    create_router_agent,
    create_question_answering_agent,
    create_keyword_extraction_agent,
    create_date_extraction_agent,
    build_context,
    search_for_basic,
    search_for_others,
    merge_dicts,
)
from src.lib.utils import log_chat


class StateSchema(t.TypedDict):
    original_query: str
    retrieval_query: str
    collection_name: str
    top_k: int
    score_threshold: float
    use_reranking: bool
    system_prompt: str
    agent: str

    keyword_filter: MetadataFilter
    date_filter: MetadataFilter
    qdrant_filter: Filter
    retrieved_docs: list[Document]
    final_docs: list[Document]

    messages: t.Annotated[list[AnyMessage], add_messages]

    log_file: str


class InputSchema(t.TypedDict):
    messages: t.Annotated[list[AnyMessage], add_messages]


class OutputSchema(t.TypedDict):
    messages: t.Annotated[list[AnyMessage], add_messages]
    documents: t.Annotated[dict[str, Document], merge_dicts]
    context: list[Document]


def prepare_thread(state: InputSchema) -> dict:
    complement_extraction_agent = create_complement_extraction_agent()
    router_agent = create_router_agent()
    query = state["messages"][-1].content

    retrieval_query = complement_extraction_agent.invoke(query)
    print(f"Query: {query}")
    print(f"Query for retrieval: {retrieval_query}")

    agent = router_agent.invoke(query)
    print(f"Routed to agent: {agent}")

    updated = dict(
        original_query=query,
        retrieval_query=retrieval_query,
        collection_name="lessons_learned",
        top_k=1000 if agent == "trend_agent" else 20,
        score_threshold=0.1 if agent == "trend_agent" else 0.3,
        use_reranking=agent != "trend_agent",
        system_prompt=SYSTEM_PROMPT.get(agent, SYSTEM_PROMPT["default"]),
        agent=agent,
    )

    if agent == "trend_agent":
        updated.update(dict(score_threshold=0))

    writer = get_stream_writer()
    writer(
        dict(
            type="reasoning",
            message=f"Truy vấn là ***{agent.replace('_agent', ' query').lower()}***.",
        )
    )

    LOG_FILE = Path("logs/chat_logs.jsonl")
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    updated["log_file"] = LOG_FILE

    return updated


def should_retrieve(
    state: StateSchema,
) -> t.Literal["extract_keyword", "extract_date", "answer_directly"]:
    # retrieval_decision_agent = create_retrieval_decision_agent()

    # should_retrieve = retrieval_decision_agent.invoke(state["messages"][-1].content)
    # print(f"Should retrieve: {should_retrieve}")
    should_retrieve = True
    if should_retrieve:
        return ["extract_keyword", "extract_date"]
    else:
        return "answer_directly"


def extract_keyword(state: StateSchema) -> dict:
    try:
        keyword_extraction_agent = create_keyword_extraction_agent()
        result = keyword_extraction_agent.invoke(state["original_query"])
        print(f"Extracted keyword: {result}")
        keyword_filter = MetadataFilter(must=[result])
    except Exception as e:
        print(str(e))
        keyword_filter = None

    updated = dict(keyword_filter=keyword_filter)

    return updated


def extract_date(state: StateSchema) -> dict:
    try:
        date_extraction_agent = create_date_extraction_agent()
        result = date_extraction_agent.invoke(state["original_query"])
        print(f"Extracted date: {result}")
        date_filter = MetadataFilter(must=[result])
    except Exception as e:
        print(str(e))
        date_filter = None

    updated = dict(date_filter=date_filter)

    return updated


def build_metadata_filter(state: StateSchema) -> dict:
    try:
        if state["date_filter"] and state["keyword_filter"]:
            qdrant_filter = state["keyword_filter"] + state["date_filter"]
        elif state["date_filter"]:
            qdrant_filter = state["date_filter"]
        elif state["keyword_filter"]:
            qdrant_filter = state["keyword_filter"]
        else:
            qdrant_filter = None
    except Exception as e:
        print(str(e))
        qdrant_filter = None

    qdrant_filter = build_qdrant_filter(qdrant_filter)
    print(f"Built qdrant filter: {qdrant_filter}")

    updated = dict(qdrant_filter=qdrant_filter)

    return updated


def retrieve_with_hybrid_search(state: StateSchema) -> dict:
    if state["agent"] == "trend_agent":
        results = search_for_others(
            filter=state["qdrant_filter"],
            collection_name=state["collection_name"],
            top_k=state["top_k"],
        )
    else:
        results = search_for_basic(
            query=state["retrieval_query"],
            filter=state["qdrant_filter"],
            collection_name=state["collection_name"],
            top_k=state["top_k"],
            score_threshold=state["score_threshold"],
        )

    writer = get_stream_writer()
    writer(
        dict(
            type="reasoning",
            message="Tìm thấy {} tài liệu.".format(len(results)),
        )
    )

    updated = dict(retrieved_docs=list(map(itemgetter(0), results)))

    return updated


def rerank_docs(state: StateSchema) -> dict:
    retrieved_docs = state["retrieved_docs"]

    if state["use_reranking"]:
        reranker = create_reranker()
        results = reranker.compress_documents_with_score(
            documents=retrieved_docs,
            query=state["retrieval_query"],
        )

        print("-" * 10 + " Reranked Results " + "-" * 10)
        print(
            *(
                f"source={doc.metadata['source']}#page={doc.metadata['page_number']}: {score}"
                for doc, score in results
            ),
            sep="\n",
        )
        docs = list(map(itemgetter(0), results))
    else:
        docs = retrieved_docs

    writer = get_stream_writer()
    writer(
        dict(
            type="reasoning",
            message="Xếp hạng lại {} tài liệu.".format(len(docs)),
        )
    )

    updated = dict(final_docs=docs)

    return updated


def answer(state: StateSchema) -> dict:
    relevant_docs = build_context(state["final_docs"])
    # print("-" * 10 + " Context " + "-" * 10)
    # print(relevant_docs)
    documents = {
        f"{doc.metadata.get('source', '')}#page={doc.metadata.get('page_number', '')}": doc
        for doc in state["final_docs"]
    }
    system_prompt = state["system_prompt"].format(relevant_docs=relevant_docs)
    question_answering_agent = create_question_answering_agent()

    response = question_answering_agent.invoke(
        input=dict(messages=state["messages"]),
        context=dict(system_prompt=system_prompt),
    )

    updated = dict(
        messages=[response["messages"][-1]],
        documents=documents,
        context=state["final_docs"],
    )

    log_chat(
        state["log_file"],
        question=state["original_query"],
        answer=response["messages"][-1].content,
        retrieval_docs=state["final_docs"],
    )

    return updated


def answer_directly(state: StateSchema) -> dict:
    ai = get_base_llm().invoke(
        (
            "Hãy đọc lại câu dưới đây mà không giải thích:\n\n"
            "Hệ thống chưa nhận diện được bài học kinh nghiệm từ câu hỏi của bạn. "
            "Vui lòng thử lại bằng cách nhập **tên hệ thống/mô-đun**, thông tin về **loại sự cố**, hoặc các từ khóa về nguyên nhân "
            "(ví dụ: quá tải database, lỗi logic, tràn bộ nhớ)..."
        )
    )

    updated = dict(messages=[ai])

    log_chat(state["log_file"], question=state["original_query"], answer=ai.content)

    return updated


# Define the graph
graph = (
    StateGraph(
        state_schema=StateSchema, input_schema=InputSchema, output_schema=OutputSchema
    )
    # define nodes
    .add_node("prepare_thread", prepare_thread)
    .add_node("extract_keyword", extract_keyword)
    .add_node("extract_date", extract_date)
    .add_node("build_metadata_filter", build_metadata_filter)
    .add_node("retrieve_with_hybrid_search", retrieve_with_hybrid_search)
    .add_node("rerank_docs", rerank_docs)
    .add_node("answer", answer)
    .add_node("answer_directly", answer_directly)
    # define workflow
    .set_entry_point("prepare_thread")
    .add_conditional_edges("prepare_thread", should_retrieve)
    .add_edge("extract_keyword", "build_metadata_filter")
    .add_edge("extract_date", "build_metadata_filter")
    .add_edge("build_metadata_filter", "retrieve_with_hybrid_search")
    .add_edge("retrieve_with_hybrid_search", "rerank_docs")
    .add_edge("rerank_docs", "answer")
    .set_finish_point("answer")
    .set_finish_point("answer_directly")
    # compile the graph
    .compile(name="main_graph")
)
