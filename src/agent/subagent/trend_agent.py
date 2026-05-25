from typing import Any

from langchain_classic.retrievers import ContextualCompressionRetriever
from langgraph.graph import StateGraph

from src.agent.common import get_qdrant_store, get_reranker
from src.agent.subagent.common_node import (
    StateSchema,
    OutputSchema,
    decide_retrieval,
    answer_directly,
    extract_date,
    extract_keyword,
    combine_filter,
    answer,
)
from src.lib.utils import build_relevant_docs_ctx
from src.lib.prompts import TREND_SYSTEM_PROMPT


def retrieve_documents(state: StateSchema) -> dict[str, Any]:
    return {"system_prompt": TREND_SYSTEM_PROMPT}


def hybrid_search_and_rerank(state: StateSchema) -> dict[str, Any]:
    print(state["metadata_filter"].model_dump_json(indent=2))
    retriever = ContextualCompressionRetriever(
        base_retriever=get_qdrant_store().as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 100,
                "filter": state["metadata_filter"],
            },
        ),
        base_compressor=get_reranker(top_n=100),
    )

    docs = retriever.invoke(state["messages"][-1].content)
    relevant_docs, documents = build_relevant_docs_ctx(docs)

    return {"relevant_docs": relevant_docs, "documents": documents}


# Define the graph
graph = (
    StateGraph(
        state_schema=StateSchema,
        output_schema=OutputSchema,
    )
    # define nodes
    .add_node("retrieve_documents", retrieve_documents)
    .add_node("answer_directly", answer_directly)
    .add_node("extract_keyword", extract_keyword)
    .add_node("extract_date", extract_date)
    .add_node("combine_filter", combine_filter)
    .add_node("hybrid_search_and_rerank", hybrid_search_and_rerank)
    .add_node("answer", answer)
    # define workflow
    .set_conditional_entry_point(decide_retrieval)
    .add_edge("retrieve_documents", "extract_keyword")
    .add_edge("retrieve_documents", "extract_date")
    .add_edge("answer_directly", "answer")
    .add_edge("extract_keyword", "combine_filter")
    .add_edge("extract_date", "combine_filter")
    .add_edge("combine_filter", "hybrid_search_and_rerank")
    .add_edge("hybrid_search_and_rerank", "answer")
    .set_finish_point("answer")
    # compile the graph
    .compile(name="trend_agent_graph")
)
