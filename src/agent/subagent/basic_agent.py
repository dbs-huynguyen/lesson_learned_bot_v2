from typing import Any

from langgraph.graph import StateGraph

from src.agent.subagent.rag_agent import graph as rag_agent
from src.agent.subagent.common_node import (
    StateSchema,
    OutputSchema,
    decide_retrieval,
    answer_directly,
    rewrite_query,
    merge_docs,
    answer,
)
from src.lib.prompts import BASIC_SYSTEM_PROMPT


def retrieve_documents(state: StateSchema) -> dict[str, Any]:
    return {
        "system_prompt": BASIC_SYSTEM_PROMPT,
        "top_k": 20,
        "score_threshold": 0.3,
        "collection_name": "lessons_learned",
        "use_reranking": True,
    }


# Define the graph
graph = (
    StateGraph(
        state_schema=StateSchema,
        output_schema=OutputSchema,
    )
    # define nodes
    .add_node("answer_directly", answer_directly)
    .add_node("retrieve_documents", retrieve_documents)
    .add_node("rag_agent", rag_agent)
    .add_node("merge_docs", merge_docs)
    .add_node("answer", answer)
    # define workflow
    .set_conditional_entry_point(decide_retrieval)
    .add_conditional_edges("retrieve_documents", rewrite_query)
    .add_edge("rag_agent", "merge_docs")
    .add_edge("merge_docs", "answer")
    .set_finish_point("answer_directly")
    .set_finish_point("answer")
    # compile the graph
    .compile(name="basic_agent_graph")
)
