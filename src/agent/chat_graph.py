from typing import Any, Literal, Optional

from langchain_core.documents import Document
from langgraph.graph import MessagesState, StateGraph

from src.agent.common import get_llm_classification, AgentClassification
from src.lib.prompts import ROUTE_QUERY_PROMPT
from src.agent.subagent import (
    basic_agent,
    trend_agent,
    statistics_agent,
    classification_agent,
)

AgentType = Literal[
    "basic_agent",
    "trend_agent",
    "statistics_agent",
    "classification_agent",
]


class InputSchema(MessagesState):
    task: Literal["BHKN", "ISO"]


class StateSchema(MessagesState):
    # task: Literal["BHKN", "ISO"]
    documents: Optional[list[dict[str, Document]]]


class OutputSchema(MessagesState):
    documents: Optional[list[dict[str, Document]]]


def prepare_thread(state: InputSchema) -> dict[str, Any]:
    return state


def route_query(state: StateSchema) -> AgentType:
    resp = get_llm_classification().invoke(
        ROUTE_QUERY_PROMPT.format(
            schema=AgentClassification.model_json_schema(),
            query=state["messages"][-1].content,
        )
    )

    agent = "basic_agent"
    if resp["parsing_error"] is None:
        agent = resp["parsed"].agent
    print(f"Routing to agent: {agent}")
    return agent


def answer(state: StateSchema) -> dict[str, Any]:
    return state


# Define the graph
graph = (
    StateGraph(state_schema=StateSchema, input_schema=InputSchema, output_schema=OutputSchema)
    # define nodes
    .add_node("prepare_thread", prepare_thread)
    .add_node("basic_agent", basic_agent)
    .add_node("trend_agent", trend_agent)
    .add_node("statistics_agent", statistics_agent)
    .add_node("classification_agent", classification_agent)
    .add_node("answer", answer)
    # define workflow
    .set_entry_point("prepare_thread")
    .add_conditional_edges("prepare_thread", route_query)
    .add_edge("basic_agent", "answer")
    .add_edge("trend_agent", "answer")
    .add_edge("statistics_agent", "answer")
    .add_edge("classification_agent", "answer")
    .set_finish_point("answer")
    # compile the graph
    .compile(name="main_graph")
)
