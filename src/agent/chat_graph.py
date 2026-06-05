from typing import Any, Literal

from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph

from src.agent.common import (
    task_classification_1_agent,
    task_classification_2_agent,
    task_classification_3_agent,
)
from src.agent.subagent import (
    basic_agent,
    trend_agent,
    statistics_agent,
    classification_agent,
)
from src.agent.subagent.common_node import InputSchema, StateSchema, OutputSchema

AgentType = Literal[
    "basic_agent",
    "trend_agent",
    "statistics_agent",
    "classification_agent",
]


def prepare_thread(state: InputSchema) -> dict[str, Any]:
    return state


def route_query(state: StateSchema) -> AgentType:
    writer = get_stream_writer()
    query = state["messages"][-1].content

    try:
        complement = task_classification_1_agent().invoke(query)
        print(f"Complement: {complement}")
        intent = task_classification_2_agent().invoke(complement)
        print(f"Intent: {intent}")
        if intent == "detail":
            agent = "basic_agent"
        else:
            agent = task_classification_3_agent().invoke(query)
    except Exception as e:
        print(str(e))
        agent = "basic_agent"

    print(f"Agent: {agent}")

    writer(
        {
            "type": "reasoning",
            "message": f"Chọn {agent.replace('_', ' ').title()} để trả lời câu hỏi.",
        }
    )

    return agent


def answer(state: StateSchema) -> dict[str, Any]:
    return state


# Define the graph
graph = (
    StateGraph(
        state_schema=StateSchema,
        input_schema=InputSchema,
        output_schema=OutputSchema,
    )
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
